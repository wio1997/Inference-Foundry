"""Exercise pinned A2 tail row-count policy on the frozen G804 scheduler steps."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional


RECEIPT_SHA = '7700cc13a2a38fbafb988e89308b3efccedde10055c70ff7a2ee87d3d7ad8be1'
SCHEDULER_SHA = 'f608e9d94d43153b62198781180b4ef987f6369721a1cc6f2e7ba9214be066c7'
BRIDGE_SHA = 'dcf76912c1af3b2ea521914de6d6bdf92ef6565ac4bd3e2e625b53f00a40995c'
MODEL = 'sglang/srt/models/deepseek_v4.py'
BACKEND = 'sglang/srt/layers/attention/deepseek_v4_backend.py'
CONFIG = 'sglang/srt/configs/deepseek_v41.py'
LOG_MARKER = 'G1464_SG_TAIL_PATH '
WINDOW = 128


def read_pinned(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f'Input hash mismatch: {path.name}')
    return json.loads(raw)


def load_receipt(path):
    packet = read_pinned(path, RECEIPT_SHA)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    record, = (line.removeprefix(LOG_MARKER) for line in packet['output'].splitlines()
               if line.startswith(LOG_MARKER))
    source = json.loads(record)
    assert source['epoch'] == 1317 and source['role'] == 'prefill'
    return source, packet['script_sha256']


def bind_frontend(repository, receipt):
    sources = receipt['sources']
    target = receipt['model_config_fields']
    assert target['hc_pre_from_prev_sublayer'] == {'root': None, 'text': None}
    for name in (MODEL, CONFIG):
        source = (repository / 'code' / name).read_bytes()
        assert sha256(source).hexdigest() == sources[name]['sha256']
    assert sources[CONFIG]['hc_pre_from_prev_sublayer'] == ['hc_pre_from_prev_sublayer = True']
    forward = sources[MODEL]['functions']['forward']['text']
    assert 'if not _is_cuda:' in forward and 'self._forward_impl(' in forward
    model_tree = ast.parse((repository / 'code' / MODEL).read_bytes())
    model_type, = (node for node in model_tree.body
                   if isinstance(node, ast.ClassDef) and node.name == 'DeepseekV4Model')
    implementation, = (node for node in model_type.body
                       if isinstance(node, ast.FunctionDef) and node.name == '_forward_impl')
    guarded = [node for node in implementation.body if isinstance(node, ast.If)
               and ast.unparse(node.test) == 'self.hc_pre_from_prev_sublayer']
    calls = [node for guard in guarded for statement in guard.body for node in ast.walk(statement)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == 'self._forward_layers_hc_pre_from_prev']
    assert len(calls) == 1
    tail_forward = sources[MODEL]['functions']['_forward_layers_hc_pre_from_prev']['text']
    assert 'self.late_layer_start is not None' in tail_forward
    assert 'forward_batch.forward_mode.is_extend_without_speculative()' in tail_forward
    assert target['candidate_source_layer_id']['text'] == max(target['kv_source_layer_ids']['text']) == 20


def load_pinned_policy(receipt):
    sources = receipt['sources']
    backend = sources[BACKEND]['global_functions']
    helper = sources[receipt['tail_helpers']]['functions']
    functions = {name: backend[name] for name in (
        '_late_layer_whole_extend', 'late_layer_whole_extend', 'late_layer_num_tokens')}
    functions['late_layer_tail_lens'] = helper['late_layer_tail_lens']
    # Only the frozen source functions above are loaded. Batch metadata is an isolated fixture.
    code = '\n\n'.join(item['text'] for item in functions.values())
    tree = ast.parse(code)
    assert [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] == list(functions)
    namespace = {
        'List': List, 'Optional': Optional, 'ForwardBatch': SimpleNamespace,
        'CaptureHiddenMode': SimpleNamespace(FULL='full'), 'SWA_WINDOW': WINDOW,
    }
    exec(compile(tree, '<fixed-A2-tail-policy>', 'exec'), namespace)
    return namespace['late_layer_num_tokens'], {name: item['lines'] for name, item in functions.items()}


def batch(lengths, *, pure_extend=True, capture='none', logprob_starts=None):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: pure_extend),
        extend_seq_lens_cpu=lengths, input_ids=SimpleNamespace(shape=(sum(lengths),)),
        capture_hidden_mode=capture, return_logprob=logprob_starts is not None,
        extend_logprob_start_lens_cpu=logprob_starts,
    )


def audit(repository, receipt, scheduler, bridge):
    bind_frontend(repository, receipt)
    count, lines = load_pinned_policy(receipt)
    logical = {item['logical']['iteration']: item['logical'] for item in scheduler}
    steps = bridge['active_steps']
    assert len(steps) == 65 and len(logical) == 198
    tail_rows = full_rows = 0
    for row in steps:
        step = logical[row['iteration']]
        requests = step['requests']
        assert all(req['context_phase'] for req in requests)
        lengths = [req['scheduled_query_tokens'] for req in requests]
        total = sum(lengths)
        ordinary = count(batch(lengths), dspark=True)
        full_capture = count(batch(lengths, capture='full'), dspark=True)
        assert total == row['scheduled_query_tokens']
        assert ordinary == full_capture < total
        assert row['device_tasks_by_query_rows'] == {str(total): 4, str(ordinary): 4}
        full_rows += total
        tail_rows += ordinary
    assert (full_rows, tail_rows) == (524272, 9847)

    scenarios = (
        ('ordinary_extend', batch([8192, 15]), True, 143),
        ('dspark_full_capture', batch([8192, 15], capture='full'), True, 143),
        ('full_capture_without_dspark', batch([8192, 15], capture='full'), False, 8207),
        ('one_prompt_logprob_reader', batch([8192, 400], logprob_starts=[8192, 399]), True, 528),
        ('no_prompt_row_requested', batch([8192, 400], logprob_starts=[8192, 400]), True, 256),
        ('non_extend_or_speculative_extend', batch([8192, 15], pure_extend=False), True, 8207),
    )
    checks = {}
    for name, metadata, dspark, expected in scenarios:
        actual = count(metadata, dspark=dspark)
        assert actual == expected, name
        checks[name] = actual
    return {
        'source_epoch': 1317, 'native_reference_capture': 'G804',
        'V41_config_default_selects_model_forward_containing_tail_path': True,
        'tail_path_still_requires_effective_feature_and_pure_extend': True,
        'layer20_full_source_KV_write_scope': 'Source contract from the linked G1463 live source check.',
        'source_functions': lines,
        'executed_reader_policy_controls': checks,
        'native_reference_nonzero_steps_with_same_ordinary_tail_row_count': len(steps),
        'native_reference_total_full_and_tail_rows': [full_rows, tail_rows],
        'SGLang_actual_NPU_tail_layout_KV_or_model_outputs_verified': False,
        'scope': ('CPU-only execution of fixed row-count functions with isolated batch metadata. '
                  'G804 is a vLLM workload and device reference. Actual Ascend tail execution still needs validation.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('prefill_source_receipt', type=Path)
    parser.add_argument('native_scheduler_rows', type=Path)
    parser.add_argument('native_step_bridge', type=Path)
    args = parser.parse_args()
    receipt, observer_sha = load_receipt(args.prefill_source_receipt)
    scheduler = read_pinned(args.native_scheduler_rows, SCHEDULER_SHA)
    bridge = read_pinned(args.native_step_bridge, BRIDGE_SHA)
    result = audit(args.repository, receipt, scheduler, bridge)
    result['private_source_receipt_sha256'] = RECEIPT_SHA
    result['observer_source_sha256'] = observer_sha
    result['runtime_source_sha256'] = {name: receipt['sources'][name]['sha256']
                                      for name in (MODEL, BACKEND, CONFIG, receipt['tail_helpers'])}
    result['native_reference_sha256'] = {'scheduler': SCHEDULER_SHA, 'indexer_step_bridge': BRIDGE_SHA}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
