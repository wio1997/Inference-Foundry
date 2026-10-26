#!/usr/bin/env python3
"""Execute complete source methods with explicit CPU control-flow fixtures.

No torch/NPU import, graph capture, DMA, model inference, or timing. Fixtures
test dispatch/event-mailbox semantics; they cannot establish device ordering.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import itertools
import json
import pathlib
from enum import Enum
from types import SimpleNamespace as NS

R = pathlib.Path(__file__).resolve().parents[1]
O = R / 'runs/L026-R008-source-policy'
O.mkdir(parents=True, exist_ok=False)
S = R / 'sources/installed/python/sglang/srt'
protocol = json.loads((R / 'specs/L026-source-policy.json').read_text())
for name, expected in protocol['source_hashes'].items():
    assert hashlib.sha256((S / name).read_bytes()).hexdigest() == expected, name
bodies, cases = [], []
checks = 0


def check(condition, label):
    global checks
    checks += 1
    assert condition, label


def extract(path, owner, name, namespace):
    p = S / path
    text = p.read_text()
    tree = ast.parse(text)
    scope = tree if owner is None else next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
    )
    node = next(n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    original = ast.get_source_segment(text, node)
    bodies.append({'path': path, 'owner': owner, 'name': name,
                   'line': node.lineno, 'end_line': node.end_lineno,
                   'body_sha256': hashlib.sha256(original.encode()).hexdigest(),
                   'complete_body': True, 'decorators_removed': bool(getattr(node, 'decorator_list', []))})
    node = copy.deepcopy(node)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(p), 'exec'), namespace)
    return namespace[name]


class Flag:
    def __init__(self, value): self.value = value
    def get(self): return self.value


g = {'Enum': Enum}
SharedReadEnds = extract('layers/attention/base_attn_backend.py', None, 'SharedReadEnds', g)
g['SharedReadEnds'] = SharedReadEnds
resolve = extract('model_executor/runner/decode_cuda_graph_runner.py', 'DecodeCudaGraphRunner', '_resolve_shared_read_ends', g)
for declared, has_marker in itertools.product(SharedReadEnds, [False, True]):
    marker = object() if has_marker else None
    actual = resolve(NS(in_graph_metadata_prep_done=marker), NS(shared_read_ends=lambda fm: declared), 'fixture-mode')
    expected = SharedReadEnds.PRE_REPLAY if declared is SharedReadEnds.IN_REPLAY and not has_marker else declared
    check(actual is expected, 'generic resolver source contract')
    cases.append({'family': 'GENERIC_RESOLVER', 'declared': declared.name,
                  'has_marker': has_marker, 'actual': actual.name,
                  'NPU_execute_uses_this_method': False})

g['envs'] = NS(SGLANG_FORCE_COARSE_WAR_BARRIER=Flag(False))
war = extract('managers/scheduler.py', 'Scheduler', '_apply_war_barrier', g)
last_reader = extract('speculative/eagle_worker_v2.py', 'EAGLEWorkerV2', 'last_shared_read_runner', g)
for enabled, force, event_present, owns_draft in itertools.product([False, True], repeat=4):
    events = []
    target_event, draft_event = object(), object()
    target = NS(shared_read_done_event=target_event)
    draft = NS(shared_read_done_event=draft_event)
    worker = NS(_target_worker=NS(model_runner=target), _draft_worker=NS(draft_runner=draft) if owns_draft else None)
    selected = last_reader(worker)
    check(selected is (draft if owns_draft else target), 'EAGLE last owner')
    if not event_present: selected.shared_read_done_event = None
    before = selected.shared_read_done_event
    other = target if owns_draft else draft
    other_before = other.shared_read_done_event
    g['envs'].SGLANG_FORCE_COARSE_WAR_BARRIER.value = force
    forward_stream = object()
    schedule = NS(wait_event=lambda ev: events.append(('event', ev)),
                  wait_stream=lambda stream: events.append(('stream', stream)))
    war(NS(_war_barrier_enabled=enabled, model_worker=NS(last_shared_read_runner=selected),
           schedule_stream=schedule, forward_stream=forward_stream))
    expected = [] if not enabled else [('event', before)] if event_present and not force else [('stream', forward_stream)]
    check(events == expected, 'WAR wait selection')
    check(selected.shared_read_done_event is (None if enabled else before), 'mailbox consume only if enabled')
    check(other.shared_read_done_event is other_before, 'non-owner mailbox unchanged')
    cases.append({'family': 'WAR_MAILBOX', 'enabled': enabled, 'force_coarse': force,
                  'event_present': event_present, 'owner': 'draft' if owns_draft else 'target',
                  'wait': events[0][0] if events else 'none'})

external_calls = []
def fake_event(**kwargs):
    external_calls.append(kwargs)
    if event_type_error: raise TypeError('fixture external unsupported')
    return 'external-event-fixture'
g['is_cuda'] = lambda: cuda_fixture
make_event = extract('model_executor/runner_utils/shared_read_event.py', None, 'make_external_event', g)
for cuda_fixture, event_type_error in itertools.product([False, True], repeat=2):
    external_calls.clear()
    got = make_event(NS(Event=fake_event))
    check(got == ('external-event-fixture' if cuda_fixture and not event_type_error else None), 'external event gate')
    check(external_calls == ([{'external': True}] if cuda_fixture else []), 'non-CUDA never attempts external')
    cases.append({'family': 'EXTERNAL_EVENT', 'cuda_fixture': cuda_fixture,
                  'type_error': event_type_error, 'returned_marker': got is not None})

g['contextlib'] = contextlib
plan_stream, plan_ctx = object(), object()
g['torch'] = NS(get_device_module=lambda device: NS(Stream=lambda: plan_stream, stream=lambda stream: plan_ctx))
g['envs'].SGLANG_ENABLE_OVERLAP_PLAN_STREAM = Flag(False)
plan = extract('speculative/spec_utils.py', None, 'get_plan_stream', g)
for enabled in [False, True]:
    g['envs'].SGLANG_ENABLE_OVERLAP_PLAN_STREAM.value = enabled
    stream, ctx = plan('npu-fixture')
    check(stream is (plan_stream if enabled else None), 'plan-stream selection')
    check(ctx is plan_ctx if enabled else isinstance(ctx, contextlib.nullcontext), 'plan context selection')
    cases.append({'family': 'PLAN_STREAM', 'enabled': enabled})


class TensorFixture:
    """Only slicing/copy/list interfaces used below; no tensor computation."""
    def __init__(self, label, trace, values=(7, 9)): self.label, self.trace, self.values = label, trace, list(values)
    def __getitem__(self, key): return self
    def copy_(self, source): self.trace.append('copy:' + self.label); return self
    def cpu(self): self.trace.append('cpu:' + self.label); return self
    def tolist(self): return list(self.values)
    def __add__(self, value): return TensorFixture(self.label, self.trace, [x + value for x in self.values])


class PPProxyTensors:
    def __init__(self, tensors): self.tensors = tensors


class LogitsProcessorOutput:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


g.update(PPProxyTensors=PPProxyTensors, LogitsProcessorOutput=LogitsProcessorOutput)
g['is_deepseek_dsa'] = lambda hf: hf.dsa
g['is_deepseek_v4'] = lambda hf: hf.dsv4
npu_execute = extract('hardware_backend/npu/graph_runner/npu_graph_runner.py', 'NPUGraphRunner', 'execute', g)
for architecture, prep, output_kind, verify in itertools.product(['DSA', 'DSV4', 'OTHER'], ['load', 'preplanned', 'ragged'], ['proxy', 'logits'], [False, True]):
    trace = []
    output_tensor = TensorFixture('output', trace)
    output = PPProxyTensors({'x': output_tensor}) if output_kind == 'proxy' else LogitsProcessorOutput(next_token_logits=output_tensor, full_logits=None, hidden_states=output_tensor)
    before_event = object()
    runner = NS(ragged_verify_mode=prep == 'ragged', bs=2, raw_bs=2, raw_num_token=2,
                captured_req_width=4, _replay_graph_key='ragged-key', is_dllm=False,
                model_runner=NS(model_config=NS(hf_config=NS(dsa=architecture == 'DSA', dsv4=architecture == 'DSV4')),
                                spec_algorithm=NS(is_dflash=lambda: False), shared_read_done_event=before_event))
    runner.buffers = NS(**{name: TensorFixture(name, trace) for name in ['input_ids', 'positions', 'mrope_positions']})
    runner.load_batch = lambda *args: trace.append('load_batch-fixture')
    runner._make_graph_key = lambda bs: ('bucket', bs)
    runner._get_update_attr_name = lambda: 'fixture-attr'
    runner._get_update_attr_type = lambda: []
    runner._resolve_shared_read_ends = lambda *a: (_ for _ in ()).throw(AssertionError('unexpected generic resolver'))
    runner._publish_read_done = lambda *a: (_ for _ in ()).throw(AssertionError('unexpected generic publisher'))
    def replay(*args, **kwargs): trace.append('replay'); return output
    def update(*args, **kwargs): trace.append('replay_with_input_update'); return output
    runner.backend = NS(replay=replay, replay_with_input_update=update)
    fb = NS(needs_forward_metadata_init=lambda: prep == 'load',
            forward_mode=NS(is_target_verify=lambda: verify),
            input_ids=TensorFixture('live-input', trace), positions=TensorFixture('live-pos', trace),
            mrope_positions=TensorFixture('live-mrope', trace), seq_lens=TensorFixture('seq', trace))
    g['envs'].SGLANG_ENABLE_OVERLAP_PLAN_STREAM.value = True
    got = npu_execute(runner, fb)
    expected_replay = 'replay' if prep == 'ragged' or architecture != 'OTHER' else 'replay_with_input_update'
    check([x for x in trace if x.startswith('replay')] == [expected_replay], 'NPU replay branch')
    check(('load_batch-fixture' in trace) == (prep != 'preplanned'), 'NPU metadata planning choice')
    check(('copy:input_ids' in trace) == (prep == 'preplanned'), 'preplanned input copy')
    check(runner.model_runner.shared_read_done_event is before_event, 'NPU execute does not publish/consume marker')
    check(isinstance(got, PPProxyTensors if output_kind == 'proxy' else LogitsProcessorOutput), 'output contract')
    cases.append({'family': 'NPU_EXECUTE', 'architecture_fixture': architecture,
                  'prep_fixture': prep, 'output_fixture': output_kind, 'verify_fixture': verify,
                  'trace': trace, 'mailbox_unchanged': True,
                  'load_batch_and_backend_are_fixtures': True})

draft_path = S / 'hardware_backend/npu/graph_runner/eagle_draft_extend_npu_graph_runner.py'
draft_cls = next(n for n in ast.parse(draft_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'EAGLEDraftExtendNpuGraphRunner')
check(not any(isinstance(n, ast.FunctionDef) and n.name == 'replay' for n in draft_cls.body), 'NPU draft extend inherits replay')
check([ast.unparse(b) for b in draft_cls.bases] == ['EAGLEDraftExtendCudaGraphRunner'], 'draft extend parent')

result = {'status': 'PASS', 'semantic_checks': checks, 'cases': cases, 'case_count': len(cases),
          'complete_source_bodies_executed': bodies, 'source_hashes': protocol['source_hashes'],
          'source_commit': protocol['source_commit'], 'native_executed': False,
          'whole_modules_imported': False, 'torch_imported': False, 'model_requests': 0,
          'accepted_baseline': None, 'candidate_accepted': False, 'performance': 'NOT_MEASURED',
          'limitations': ['Exact source control-flow with explicit stream/event/tensor/load_batch/backend/output fixtures.',
                          'No NPU/graph capture/DMA/physical KV lifetime or performance proof.',
                          'NPU DFlash refresh, real logits, DLLM and inherited load_batch internals are not executed.',
                          'Draft extend inherited replay publisher is source-audited only, not executed here.',
                          'Generic IN_REPLAY-to-PRE_REPLAY branch excluded from tested NPU override, not fixed.']}
(O / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['status', 'semantic_checks', 'case_count', 'native_executed']}))
