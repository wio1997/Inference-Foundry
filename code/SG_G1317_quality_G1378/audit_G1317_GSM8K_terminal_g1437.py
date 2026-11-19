"""Independently regrade actual G1317 GSM8K after its formal test and worker exit."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json

from audit_paired_gsm8k_g1406 import audit_run, paired, read
from run_full_gsm8k_SG_PD_g1402 import MODEL, REQUESTS, assets, digest


RUN_ID = 'sg_pd_g1317_aisbench_32k_g1324'
WORKER_SHA = '7757463512739cab3d87df65976d5a852a7ed4098ee850840f55aada8bc1ef58'
NATIVE_RESULT_SHA = 'b7465931aa775353606b92ba8bb03b2ff4d0b1a2a4aa458e8c28bc2029d56a76'
G1317_SCOPE = 'G1317 paired Ascend 910B1 GSM8K quality validation'
IMAGE = 'sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'


def terminal_binding(formal, formal_sha, control, candidate):
    assert control.name == 'SG_PD_g1317_full_GSM8K_control_g1404'
    assert candidate.name == 'SG_PD_g1317_full_GSM8K_g1404' and candidate.parent == control.parent
    assert formal.name == RUN_ID
    formal_state = read(formal / 'state.json')
    assert formal_state['phase'] == 'all_phases_terminal' and formal_state['run_id'] == RUN_ID
    formal_manifest = read(formal / 'run_manifest.json')
    audit_path = formal / 'audit_results.json'
    assert digest(audit_path) == formal_sha
    assert read(audit_path)['all_required_raw_metrics_and_usage_validated'] is True
    assert formal_manifest['engine'] == 'sglang' and formal_manifest['lengths'] == [32768]
    assert formal_manifest['gate']['profile_epoch'] == 1317

    gate = read(control / 'gate.json')
    handle = read(control / 'actual_launch_handle.json')
    worker = read(control / 'worker_handle.json')
    terminal = read(control / 'terminal.json')
    assert gate['phase'] == 'gate_passed' and gate['epoch'] == 1317 and gate['worker_SHA256'] == WORKER_SHA
    assert gate['hardware'] == 'Ascend 910B1' and gate['image'] == IMAGE
    assert gate['CPU_lease']['lease'] == 'renewed_same_container'
    assert gate['CPU_lease']['container_ID'] == formal_manifest['client_container_id']
    assert handle['controller_PID'] == worker['controller_PID'] and worker['docker_exec_PID'] > 1
    assert terminal['exit_code'] == 0
    candidate_state = read(candidate / 'state.json')
    assert candidate_state['phase'] == 'full_test_complete_requires_parent_review'
    assert candidate_state['completed'] == REQUESTS
    assert read(candidate / 'manifest.json')['scope'] == G1317_SCOPE
    assert read(candidate / 'actual_models.json')['data'][0]['id'] == MODEL

    finished = datetime.fromisoformat(candidate_state['at'])
    elapsed = candidate_state['elapsed_s']
    assert type(elapsed) in (int, float) and 0 < elapsed <= 14400
    started = finished - timedelta(seconds=elapsed)
    assert datetime.fromisoformat(formal_state['at']) <= datetime.fromisoformat(handle['at'])
    worker_started = datetime.fromisoformat(worker['at'])
    # The worker handle is written just after creating docker exec.
    assert worker_started - timedelta(seconds=5) <= started <= finished
    assert finished <= datetime.fromisoformat(terminal['at'])
    resources = gate['deadlines_UTC']
    assert set(resources) == {'P', 'D', 'guard', 'cpu'}
    assert finished < min(datetime.fromisoformat(value) for value in resources.values())
    return {'epoch': 1317, 'formal_audit_SHA256': formal_sha, 'quality_worker_elapsed_s': elapsed,
            'quality_worker_source_SHA256': WORKER_SHA,
            'worker_started_UTC': worker['at'], 'worker_finished_UTC': candidate_state['at'],
            'terminal_control_SHA256': {
                name: digest(control / name) for name in (
                    'gate.json', 'actual_launch_handle.json', 'worker_handle.json', 'terminal.json')
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for directory in ('formal', 'control', 'candidate', 'assets', 'native'):
        parser.add_argument(f'--{directory}', type=Path, required=True)
    parser.add_argument('--formal-audit-sha256', required=True)
    args = parser.parse_args()
    proof = terminal_binding(args.formal, args.formal_audit_sha256, args.control, args.candidate)
    questions, _, extract, official = assets(args.assets.resolve(strict=True))
    baseline, vllm = audit_run(args.native, NATIVE_RESULT_SHA, questions, extract, official)
    assert vllm['correct'] == 1270 and not vllm['transport_failures'] and not vllm['length_limited_responses']
    result_sha = read(args.candidate / 'result.json')['results_jsonl_sha256']
    candidate, sglang = audit_run(args.candidate, result_sha, questions, extract, official)
    print(json.dumps({
        'actual_G1317_terminal_binding': proof, 'vLLM': vllm, 'SGLang': sglang,
        'paired_official_grades': paired(baseline, candidate),
        'both_runs_all_request_bytes_verified': REQUESTS,
        'SGLang_all_transport_and_no_length_limit': not sglang['transport_failures'] and not sglang['length_limited_responses'],
    }, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
