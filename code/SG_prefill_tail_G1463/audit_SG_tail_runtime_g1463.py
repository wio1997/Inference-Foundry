"""Summarize the actual G1317 bounded-replay resolution and its source contract."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


GATE_MARKER = 'G1463_SG_ACTUAL_TAIL_GATE'
CONTRACT_MARKER = 'G1463_SG_TAIL_KV_CONTRACT'
MODEL_SOURCE = 'models/deepseek_v4.py'
GATE_SOURCE = 'arg_groups/deepseek_v4_hook.py'
BASE_BACKEND = 'layers/attention/deepseek_v4_backend.py'
REFERENCE = '5d5094d6c65990a23cad2abcafeb19876b8d4845'


def load_receipt(path, marker):
    raw = path.read_bytes()
    packet = json.loads(raw)
    assert packet['exitstatus'] == 0 and packet['outcome'] == 'finished'
    (record,) = (line.removeprefix(marker + ' ') for line in packet['output'].splitlines()
                 if line.startswith(marker + ' '))
    return json.loads(record), {
        'receipt_sha256': sha256(raw).hexdigest(),
        'observer_source_sha256': packet['script_sha256'],
    }


def summarize(repository, roles, contract):
    pinned = json.loads((repository / (
        'code/SG_PD_wait_recovery_G1317/G1317_runtime_source_index.json')).read_bytes())
    assert pinned['epoch'] == contract['epoch'] == 1317 and contract['role'] == 'prefill'
    source_index = {row['role_relative_path']: row for row in pinned['roles']['prefill']['pinned_role_files']}
    adapter_sources = {}
    for name, digest in contract['adapter_source_SHA256'].items():
        row = source_index[name]
        relative = row['exact_repo_path']
        assert sha256((repository / relative).read_bytes()).hexdigest() == row['sha256'] == digest
        adapter_sources[relative] = digest
    assert set(contract['adapter_source_SHA256']) == {'integration.py', 'npu_backend.py', 'pd_gate_bridge.py'}

    model_path = repository / 'code/sglang/srt/models/deepseek_v4.py'
    model_sha = sha256(model_path.read_bytes()).hexdigest()
    role_records = {}
    for role, receipt in roles.items():
        assert receipt['epoch'] == 1317 and receipt['role'] == role
        assert receipt['candidate_SHA256'] == pinned['roles'][role]['candidate_config_SHA256']
        assert not receipt['replay_flags_in_candidate_argv']
        assert receipt['log']['decoder_resolution_lines'] == ['False']
        assert receipt['runtime_selector']['sources'][MODEL_SOURCE]['sha256'] == model_sha
        source = receipt['runtime_selector']['sources'][GATE_SOURCE]
        gate = dict(source['selected'])
        assert 'get_platform().is_cuda or get_platform().is_hip' in gate[502]
        assert 'cfg.disaggregation_mode == "null" or get_platform().is_cuda' in gate[504]
        model_config = receipt['runtime_selector']['model_config']
        assert model_config['num_hidden_layers'] == 40 and model_config['sliding_window'] == 128
        assert max(model_config['kv_source_layer_ids']) == model_config['candidate_source_layer_id'] == 20
        role_records[role] = {
            'observed_at_utc': receipt['at'],
            'candidate_config_sha256': receipt['candidate_SHA256'],
            'startup_log_prefix_sha256': receipt['log']['start_log_prefix_sha256'],
            'startup_log_prefix_bytes': receipt['log']['start_log_prefix_bytes'],
            'tail_setting_from_startup_resolution': False,
            'explicit_bounded_replay_flags': [],
            'model_config_sha256': receipt['runtime_selector']['model_config_SHA256'],
            'replay_gate_source_sha256': source['sha256'],
            'existing_scoped_gate_overlay_sha256s': receipt['gate_overlay_SHA256s'],
        }
    prefill = roles['prefill']
    decode = roles['decode']
    base_backend = prefill['runtime_selector']['sources'][BASE_BACKEND]
    assert role_records['prefill']['model_config_sha256'] == role_records['decode']['model_config_sha256']
    assert role_records['prefill']['replay_gate_source_sha256'] == role_records['decode']['replay_gate_source_sha256']
    assert prefill['gate_overlay_SHA256s'] == decode['gate_overlay_SHA256s'] and len(prefill['gate_overlay_SHA256s']) == 1
    assert contract['sources'][BASE_BACKEND]['sha256'] == base_backend['sha256']
    assert contract['sources'][MODEL_SOURCE]['sha256'] == model_sha
    functions = contract['sources'][BASE_BACKEND]['functions']
    write = functions['write_tail_source_kv']['source']
    assert 'self.forward_low_ratio_sources(' in write and 'run_indexer=False' in write
    assert 'self.tail_source_kv_layer_id = layer.layer_id' in write
    assert 'request_window.activate(' in functions['enter_late_layer_tail']['source']
    assert 'self.tail_source_kv_layer_id = None' in functions['exit_late_layer_tail']['source']
    readers = contract['sources'][MODEL_SOURCE]['functions']['_check_late_layer_tail_readers']['source']
    assert 'CaptureHiddenMode.FULL' in readers and 'extend_logprob_start_lens_cpu' in readers

    return {
        'epoch': 1317, 'image': prefill['image'], 'roles': role_records,
        'runtime_model': prefill['runtime_selector']['model_config'],
        'runtime_model_sha256': model_sha,
        'actual_prefill_adapter_sources': adapter_sources,
        'runtime_replay_gate': {
            'source': GATE_SOURCE, 'default_platform_checks_lines': [501, 515],
            'observed_platform': 'Ascend 910B NPU',
            'automatic_decoder_SWA_bounded_replay_for_Ascend_PD': False,
        },
        'actual_prefill_source_contract': {
            'model_tail_reader_check_lines': contract['sources'][MODEL_SOURCE]['functions'][
                '_check_late_layer_tail_readers']['lines'],
            'generic_attention_backend_sha256': base_backend['sha256'],
            'backend_function_lines': {name: fn['lines'] for name, fn in functions.items()},
            'source_layer_if_enabled': 20,
            'source_layer_full_row_compressed_KV_write_precedes_tail_entry': True,
            'tail_uses_request_window_layout_when_request_window_allocated': True,
            'tail_respects_full_hidden_and_prompt_logprob_readers': True,
        },
        'native_vLLM_G804_reference_commit': REFERENCE,
        'SGLang_actual_tail_path_NPU_correctness_or_performance_verified': False,
        'scope': 'Existing startup records and immutable files read from original running P/D containers. '
                 'The observer reads startup configuration and source without executing the feature.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('prefill_receipt', type=Path)
    parser.add_argument('decode_receipt', type=Path)
    parser.add_argument('tail_contract_receipt', type=Path)
    args = parser.parse_args()
    prefill, p_proof = load_receipt(args.prefill_receipt, GATE_MARKER)
    decode, d_proof = load_receipt(args.decode_receipt, GATE_MARKER)
    contract, contract_proof = load_receipt(args.tail_contract_receipt, CONTRACT_MARKER)
    result = summarize(args.repository, {'prefill': prefill, 'decode': decode}, contract)
    result['evidence_inputs'] = {'prefill': p_proof, 'decode': d_proof, 'prefill_contract': contract_proof}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
