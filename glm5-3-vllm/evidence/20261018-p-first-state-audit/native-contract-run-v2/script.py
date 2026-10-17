#!/usr/bin/env python3
"""Bounded offline audit of pinned producer-token state contracts, without device access.

Full native imports and definition-only native-method execution are reported separately.
Fixture objects are scalar/list recorders, not KV allocators, samplers or model simulators.
"""
import __future__
import argparse
import ast
import dataclasses
import datetime
import enum
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import types


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--known-result', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output/'script.py').write_bytes(source)
    pins = []
    for manifest in ('native-manifest.json', 'mooncake-manifest.json'):
        raw = (args.source_root/manifest).read_bytes()
        for row in json.loads(raw):
            if row.get('full_source') is False:
                continue
            path = args.source_root/row['path']
            data = path.read_bytes()
            assert digest(data) == row['sha256'], row['path']
            pins.append({'path': row['path'], 'sha256': row['sha256'],
                         'bytes': len(data), 'manifest': manifest})
    # Same file was captured independently on the initial image and D10.
    unique = {}
    for row in pins:
        assert row['path'] not in unique or unique[row['path']]['sha256'] == row['sha256']
        unique[row['path']] = row
    pins = list(unique.values())
    text = {row['path']: (args.source_root/row['path']).read_text() for row in pins}
    tree = {key: ast.parse(value) for key, value in text.items()}
    vllm = 'vllm/vllm/'
    moon = 'vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/'
    req_path = vllm+'v1/request.py'
    sched_path = vllm+'v1/core/sched/scheduler.py'
    output_path = vllm+'v1/core/sched/output.py'
    gpu_path = vllm+'v1/worker/gpu_model_runner.py'
    report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(), 'script_sha256': digest(source),
        'source_root': str(args.source_root), 'source_pins': pins, 'selected_definitions': [],
        'scope': 'Local files only. No SSH, device/model work, downloads or package installation. '
                 'Unmodified AST definitions compiled with postponed annotations, without module imports. '
                 'Request.__init__, ConstantList, SamplingParams, actual KV manager, worker InputBatch '
                 'and device MTP are NOT validated by the isolated calls below.'}
    report['dependencies'] = {name: bool(importlib.util.find_spec(name))
                              for name in ('torch', 'msgspec', 'vllm', 'transformers')}
    probe = subprocess.run([sys.executable, '-c',
        'import importlib.util; s=importlib.util.spec_from_file_location("native_request",'
        +repr(str(args.source_root/req_path))+'); m=importlib.util.module_from_spec(s); '
        's.loader.exec_module(m)'], capture_output=True, text=True, timeout=15)
    report['full_native_request_import'] = {'returncode': probe.returncode,
        'stderr': probe.stderr, 'passed': probe.returncode == 0}

    module = types.ModuleType('_p_first_native_definition_audit')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(dataclass=dataclasses.dataclass, enum=enum, threading=threading)

    def definition(path, name, parent=None):
        nodes = tree[path].body
        if parent:
            nodes = next(x for x in nodes if isinstance(x, ast.ClassDef) and x.name == parent).body
        node = next(x for x in nodes if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name == name)
        report['selected_definitions'].append({'source': path, 'name': name, 'parent': parent,
            'line': node.lineno, 'end_line': node.end_lineno,
            'ast_sha256': digest(ast.dump(node, include_attributes=False).encode())})
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec',
                     flags=__future__.annotations.compiler_flag), ns)
        return ns[name]

    Request = definition(req_path, 'Request')
    RequestStatus = definition(req_path, 'RequestStatus')
    NewRequestData = definition(output_path, 'NewRequestData')
    ReqMeta = definition(moon+'metadata.py', 'ReqMeta')
    BaseScheduler = definition(moon+'base_scheduler.py', 'MooncakeBaseConnectorScheduler')
    Receiver = definition(moon+'pull_worker.py', 'MooncakePullRecvingThread')
    stop = definition(vllm+'v1/core/sched/utils.py', 'check_stop')
    promote = definition(sched_path, '_update_waiting_for_remote_kv', 'Scheduler')
    promote_blocked = definition(sched_path, '_try_promote_blocked_waiting_request', 'Scheduler')
    async_path = vllm+'v1/core/sched/async_scheduler.py'
    async_class = next(x for x in tree[async_path].body if isinstance(x, ast.ClassDef)
                       and x.name == 'AsyncScheduler')
    async_update = next(x for x in async_class.body if isinstance(x, ast.FunctionDef)
                        and x.name == '_update_request_with_output')
    decrement_if = next(x for x in async_update.body if isinstance(x, ast.If)
                        and isinstance(x.test, ast.UnaryOp) and isinstance(x.test.operand, ast.Name)
                        and x.test.operand.id == 'is_stale')
    decrement_fragment = ast.Module(body=decrement_if.body, type_ignores=[])
    report['async_fragment'] = {'source': async_path, 'line': decrement_if.lineno,
        'ast_sha256': digest(ast.dump(decrement_if, include_attributes=False).encode()),
        'scope': 'Only the original post-completion placeholder decrement/assert is executed, '
                 'not AsyncScheduler superclass/model output handling.'}
    report['dto_contracts'] = {
        'NewRequestData_fields': [x.name for x in dataclasses.fields(NewRequestData)],
        'ReqMeta_fields': [x.name for x in dataclasses.fields(ReqMeta)],
        'native_new_request_has_initial_output_ids':
            'output_token_ids' in {x.name for x in dataclasses.fields(NewRequestData)},
        'native_transfer_reqmeta_has_P_first_token':
            'last_token_id' in {x.name for x in dataclasses.fields(ReqMeta)}}
    assert not report['dto_contracts']['native_new_request_has_initial_output_ids']
    assert not report['dto_contracts']['native_transfer_reqmeta_has_P_first_token']
    constructors = [x for x in ast.walk(tree[gpu_path]) if isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Name) and x.func.id == 'CachedRequestState']
    initial_ids = [kw.value for x in constructors for kw in x.keywords
                   if kw.arg == 'output_token_ids']
    assert len(initial_ids) == 1 and isinstance(initial_ids[0], ast.List) and not initial_ids[0].elts
    report['dto_contracts']['worker_new_request_output_ids_literal'] = []
    report['dto_contracts']['worker_constructor_line'] = constructors[0].lineno

    schedule = next(x for x in tree[sched_path].body if isinstance(x, ast.ClassDef)
                    and x.name == 'Scheduler')
    schedule = next(x for x in schedule.body if isinstance(x, ast.FunctionDef) and x.name == 'schedule')
    pad_if = next(x for x in ast.walk(schedule) if isinstance(x, ast.If)
                  and any(isinstance(y, ast.Assign) and any(isinstance(t, ast.Name)
                          and t.id == 'padded_num_tokens' for t in y.targets) for y in x.body))
    pad_assign = next(x for x in pad_if.body if isinstance(x, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'padded_num_tokens' for t in x.targets))
    report['pad_rule'] = {'condition_line': pad_if.lineno, 'assignment_line': pad_assign.lineno,
        'source': ast.get_source_segment(text[sched_path], pad_if.test),
        'rhs': ast.get_source_segment(text[sched_path], pad_assign.value),
        'scope': 'Native expression evaluated with recordedK5/C1 no-other-prefill conditions; '
                 'not a full scheduler or graph/iteration execution.'}

    known_raw = args.known_result.read_bytes()
    known = json.loads(known_raw)
    assert known['completed'] and known['passed']
    report['known_result'] = {'path': str(args.known_result), 'sha256': digest(known_raw)}
    counters = []

    def request_fixture(row, max_tokens=64):
        # No replacement Request implementation: these original native methods run
        # on manually initialized fixture fields because full imports are blocked.
        obj = object.__new__(Request)
        obj.request_id = row['request_id']
        obj.prompt_token_ids = row['P_response']['choices'][0]['prompt_token_ids'].copy()
        obj.num_prompt_tokens = len(obj.prompt_token_ids)
        obj._all_token_ids = obj.prompt_token_ids.copy()
        obj._output_token_ids = []
        obj.output_token_ids = obj._output_token_ids  # Fixture read view only, not ConstantList.
        obj._block_hasher = None
        obj.block_hashes = []
        obj.num_computed_tokens = obj.num_prompt_tokens
        obj.num_in_flight_tokens = 0
        obj.num_output_placeholders = 0
        obj.num_preemptions = 0
        obj.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        obj.max_tokens = max_tokens
        obj.pooling_params = None
        obj.stop_reason = None
        obj.sampling_params = types.SimpleNamespace(eos_token_id=None, stop_token_ids=[],
            min_tokens=0, repetition_detection=None)
        return obj

    def completed_receive(obj):
        recorded = []
        owner = types.SimpleNamespace(connector=object(), failed_recving_kv_req_ids=set(),
            finished_recving_kv_req_ids={obj.request_id},
            kv_cache_manager=types.SimpleNamespace(cache_blocks=lambda request, n: recorded.append(n)))
        promote(owner, obj)
        assert not owner.finished_recving_kv_req_ids
        return recorded

    for row in known['rows']:
        n = row['input_tokens']
        params = row['P_response']['kv_transfer_params']
        assert params['remote_num_prompt_tokens'] == n
        token_ids = [t for chunk in row['D_chunks'] for c in chunk.get('choices', [])
                     for t in c.get('token_ids', [])]
        assert len(token_ids) == 64 and token_ids[0] == params['last_token_id']
        policy = object.__new__(BaseScheduler)
        policy.need_truncate = False  # Observed P prompt length unchanged for these requests.
        assert policy._state_prefill_token_count(n) == n
        original = request_fixture(row)
        assert completed_receive(original) == [n] and original.num_computed_tokens == n-1
        adopted = request_fixture(row)
        adopted.append_output_token_ids(token_ids[0])
        assert completed_receive(adopted) == [n]
        assert (adopted.num_computed_tokens, adopted.num_tokens, adopted.num_output_tokens) == (n, n+1, 1)
        dto = NewRequestData(req_id=adopted.request_id,
            prompt_token_ids=adopted.prompt_token_ids.copy(), mm_features=[],
            sampling_params=adopted.sampling_params, pooling_params=None,
            block_ids=tuple(), num_computed_tokens=adopted.num_computed_tokens,
            lora_request=None)
        assert len(dto.prompt_token_ids) == n and not hasattr(dto, 'output_token_ids')
        pad = []
        for obj in (original, adopted):
            context = {'self': types.SimpleNamespace(num_spec_tokens=5, dynamic_sd_lookup=None,
                         num_sampled_tokens_per_step=1),
                       'num_new_tokens': obj.num_tokens-obj.num_computed_tokens,
                       'num_computed_tokens': obj.num_computed_tokens,
                       'prefill_scheduled': False, 'scheduled_running_reqs': []}
            assert eval(compile(ast.Expression(pad_if.test), sched_path, 'eval'), context)
            pad.append(eval(compile(ast.Expression(pad_assign.value), sched_path, 'eval'), context))
        assert pad == [6, 6]
        wrong_computed = n+1
        wrong_new_tokens = adopted.num_tokens-wrong_computed
        assert wrong_new_tokens == 0  # Knowing the ID does not compute its target KV.
        # Replay observed native suffix IDs, not generated/simulated model values.
        assert not stop(adopted, 8192)
        for token in token_ids[1:]:
            adopted.append_output_token_ids(token)
            finished = stop(adopted, 8192)
        assert finished and adopted.num_output_tokens == 64
        assert adopted.status == RequestStatus.FINISHED_LENGTH_CAPPED
        one = request_fixture(row, max_tokens=1)
        one.append_output_token_ids(token_ids[0])
        assert stop(one, 8192) and one.num_output_tokens == 1
        # The prompt-append workaround omits the first token from core output history.
        workaround = request_fixture(row)
        workaround.prompt_token_ids.append(token_ids[0])
        workaround.num_prompt_tokens += 1
        workaround._all_token_ids.append(token_ids[0])
        workaround.append_output_token_ids(token_ids[1:])
        assert workaround.num_output_tokens == 63 and not stop(workaround, 8192)
        counters.append({'input_tokens': n, 'request_id': row['request_id'],
            'observed_remote_prompt_tokens': params['remote_num_prompt_tokens'],
            'original_after_receive_computed': original.num_computed_tokens,
            'adopted_before_worker': {'computed': n, 'tokens': n+1, 'outputs': 1},
            'original_adopted_first_scheduled_rows': pad,
            'source_implied_first_query_positions': {'original': n-1, 'adopted': n},
            'wrong_computed_N_plus1_logical_new_tokens': wrong_new_tokens,
            'replayed_observed_total_output_ids': adopted.num_output_tokens,
            'max1_check_stop': True, 'prompt_append_without_budget_fix_stops_at_frontend64': False})

    # Negative controls use original native control code, not a replacement scheduler.
    first = known['rows'][0]
    external = request_fixture(first)
    try:
        exec(compile(decrement_fragment, async_path, 'exec'),
             {'request': external, 'new_token_ids': [first['P_last_token_id']]})
    except AssertionError:
        async_underflow_reproduced = True
    else:
        async_underflow_reproduced = False
    assert async_underflow_reproduced and external.num_output_placeholders == -1
    terminal = request_fixture(first, max_tokens=1)
    owner = types.SimpleNamespace(finished_recving_kv_req_ids={terminal.request_id},
        failed_recving_kv_req_ids=set(), connector=object(),
        kv_cache_manager=types.SimpleNamespace(cache_blocks=lambda request, n: None))
    finished_inside_hook = []
    def naive_terminal_hook(request):
        request.append_output_token_ids(first['P_last_token_id'])
        promote(owner, request)
        finished_inside_hook.append(stop(request, 8192))
    owner._update_waiting_for_remote_kv = naive_terminal_hook
    promoted = promote_blocked(owner, terminal)
    assert finished_inside_hook == [True] and promoted
    assert terminal.status == RequestStatus.WAITING
    report['negative_controls'] = {
        'external_token_as_native_async_completion_underflows_placeholders': async_underflow_reproduced,
        'max1_finished_inside_promotion_callback_overwritten_as_WAITING': True,
        'scope': 'Expected failures of naive insertion strategies. Callback KV operations '
                 'are recorders/no-ops; native stop/promotion/placeholder control statements '
                 'are unchanged. No resource release or worker execution is claimed.'}

    assert Receiver._expand_block_ids([3, 7], 1) == [3, 7]
    assert Receiver._expand_block_ids([3, 7], 2) == [6, 7, 14, 15]
    report['isolated_native_counter_replay'] = {'passed': True, 'requests': counters,
        'cache_blocks_callback': 'Argument recorder only; no KV bytes, allocation, hashing or readiness validated.',
        'sampler_parameters': 'Fixture data only; native SamplingParams/EOS masking not imported.',
        'dto_block_ids': 'Empty fixture, intentionally not a valid physical transfer descriptor.',
        'thread_state': 'Only original static block-expansion helper executed, no receiver thread started.'}
    report['missing_proofs'] = [
        'Full native package/Request construction blocked by missing compatible torch/msgspec/vllm dependencies.',
        'Actual per-layer/group handshake payloads, row contents, C8 scales/rope/indexer and MTP cache prefix were not captured.',
        'MTP proposer internals and its exact shifted-cache initialization source are not in the matching archived source set.',
        'Actual InputBatch initialization, async GPU scatter and new initial-output wire contract have not executed.',
        'Native frontend metrics/stats source and full dependency stack are not archived here; phase-accounting fix not validated.',
        'Recorded P min_tokens1 vs D min_tokens0 needs sampling-policy compatibility verification; matching observed IDs is finite evidence.'
    ]
    report['decision'] = ('Counter adoption can retain computedN while recording one known output, '
        'but current NewRequestData/worker initialization drops that output history. '
        'An explicit worker-state handoff is required. Under unchangedMTP5 padding, '
        'the initial batch is still6rows/request, so this is a one-token phase advance, '
        'not proof of deleting a model iteration or its drafter work. '
        'Physical KV/MTP-prefix validity remains conditional; no device implementation accepted.')
    report['completed'] = True
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'completed': True, 'full_native_import': report['full_native_request_import']['passed'],
                      'definition_only_replay': True, 'observed_requests': len(counters),
                      'decision': report['decision']}, indent=2))


if __name__ == '__main__':
    main()
