"""Regrade the independent G1317 quality run from its terminal raw evidence."""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

from audit_G1317_GSM8K_terminal_g1437 import G1317_SCOPE, IMAGE, NATIVE_RESULT_SHA, WORKER_SHA
from audit_paired_gsm8k_g1406 import audit_run, paired, read
from check_independent_quality_boundary_g1496 import PINS, RUN_ID, checked_sources
from run_full_gsm8k_SG_PD_g1402 import MODEL, REQUESTS, assets, digest


ACTUAL_GATE_SHA = 'c088613b08f32a204767b8673043904b851caa877118b47735baa43a85bedd63'
ACTUAL_HANDLE_SHA = '95c038d67296974eb85568373360d9de1f5328458c15d57f9b91a3c0dd5c0710'
ACTUAL_MANIFEST_SHA = '21de935f3e572c0c096aca15ffc93378a475a7b35e3239f3f27a77d028558dcb'


def terminal_binding(formal, control, candidate):
    assert formal.name == RUN_ID
    assert control.name == 'SG_PD_g1317_full_GSM8K_control_g1496'
    assert candidate.name == 'SG_PD_g1317_full_GSM8K_g1496' and candidate.parent == control.parent
    original = checked_sources(formal)['state.json']
    manifest = read(formal / 'run_manifest.json')
    assert original['phase'] == 'failed_requires_review' and original['performance_validated'] is False
    assert manifest['gate']['profile_epoch'] == 1317

    assert digest(control / 'gate.json') == ACTUAL_GATE_SHA
    assert digest(control / 'worker_handle.json') == ACTUAL_HANDLE_SHA
    assert digest(candidate / 'manifest.json') == ACTUAL_MANIFEST_SHA
    gate = read(control / 'gate.json')
    boundary = gate['independent_quality_boundary']
    assert boundary['original_terminal_artifact_sha256'] == PINS and boundary['original_run_id'] == RUN_ID
    assert boundary['original_scored_performance_accepted'] is False and boundary['same_input_replay_started'] is False
    assert gate['phase'] == 'gate_passed' and gate['epoch'] == 1317 and gate['worker_SHA256'] == WORKER_SHA
    assert gate['hardware'] == 'Ascend 910B1' and gate['image'] == IMAGE
    assert gate['CPU_lease']['container_ID'] == manifest['client_container_id']
    handle = read(control / 'actual_launch_handle.json')
    worker = read(control / 'worker_handle.json')
    terminal = read(control / 'terminal.json')
    assert handle['controller_PID'] == worker['controller_PID'] == 3118006
    assert worker['docker_exec_PID'] == 3118007 and terminal['exit_code'] == 0
    state = read(candidate / 'state.json')
    assert state['phase'] == 'full_test_complete_requires_parent_review' and state['completed'] == REQUESTS
    assert read(candidate / 'manifest.json')['scope'] == G1317_SCOPE
    assert read(candidate / 'actual_models.json')['data'][0]['id'] == MODEL

    finished = datetime.fromisoformat(state['at'])
    elapsed = state['elapsed_s']
    assert type(elapsed) in (int, float) and 0 < elapsed <= 14400
    began = finished - timedelta(seconds=elapsed)
    exec_stamp = datetime.fromisoformat(worker['at'])
    assert datetime.fromisoformat(original['at']) < datetime.fromisoformat(handle['at'])
    assert exec_stamp - timedelta(seconds=5) <= began <= finished
    assert finished <= datetime.fromisoformat(terminal['at'])
    deadlines = gate['deadlines_UTC']
    assert set(deadlines) == {'P', 'D', 'guard', 'cpu'}
    assert finished < min(datetime.fromisoformat(v) for v in deadlines.values())
    return {'epoch': 1317, 'original_scored_attempt_accepted': False, 'actual_worker_elapsed_s': elapsed,
            'original_terminal_SHA256': PINS['state.json'], 'worker_source_SHA256': WORKER_SHA,
            'worker_started_UTC': worker['at'], 'worker_finished_UTC': state['at'],
            'quality_terminal_controls_sha256': {name: digest(control / name) for name in (
                'gate.json', 'actual_launch_handle.json', 'worker_handle.json', 'terminal.json')}}


def compare(formal, control, candidate, asset_dir, native):
    proof = terminal_binding(formal, control, candidate)
    questions, _, extract, official = assets(asset_dir.resolve(strict=True))
    baseline, vllm = audit_run(native, NATIVE_RESULT_SHA, questions, extract, official)
    assert vllm['correct'] == 1270 and not vllm['transport_failures'] and not vllm['length_limited_responses']
    result_sha = read(candidate / 'result.json')['results_jsonl_sha256']
    candidate_grades, sglang = audit_run(candidate, result_sha, questions, extract, official)
    return {'actual_G1317_independent_terminal_binding': proof, 'SGLang': sglang, 'vLLM': vllm,
            'paired_official_grades': paired(baseline, candidate_grades),
            'both_runs_all_request_bytes_verified': REQUESTS,
            'SGLang_all_transport_and_no_length_limit':
                not sglang['transport_failures'] and not sglang['length_limited_responses']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ('formal', 'control', 'candidate', 'assets', 'native'):
        parser.add_argument(f'--{label}', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.formal, args.control, args.candidate, args.assets, args.native),
                     sort_keys=True, indent=2))
