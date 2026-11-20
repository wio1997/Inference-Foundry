"""Bind the unscored matched trace to the actual independent G1317 quality terminal."""

from hashlib import sha256
import json
from pathlib import Path


FAILED_RUN = 'sg_pd_g1317_aisbench_32k_g1324'
ORIGINAL_FAILED_STATE_SHA = '8a5c903ee01da25f5349cd9ebf86273603b584708590c662838cffc23ec2a557'
QUALITY_GATE_SHA = 'c088613b08f32a204767b8673043904b851caa877118b47735baa43a85bedd63'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def document(path):
    return json.loads(path.read_bytes())


def verify_profile_dependency(client, quality_proof_path, proof_sha):
    quality_proof_path = quality_proof_path.resolve(strict=True)
    if not quality_proof_path.is_relative_to(client.resolve(strict=True)) or digest(quality_proof_path) != proof_sha:
        raise ValueError('Quality terminal proof path or digest mismatch')
    proof = document(quality_proof_path)
    binding = proof['actual_G1317_independent_terminal_binding']
    if binding['epoch'] != 1317 or binding['original_scored_attempt_accepted'] is not False:
        raise ValueError('Quality proof does not bind this independent G1317 run')
    if proof['both_runs_all_request_bytes_verified'] != 1319 or not proof['SGLang_all_transport_and_no_length_limit']:
        raise ValueError('Expected all paired raw GSM8K records with verified response completion')
    if proof['SGLang']['completed'] != 1319 or proof['vLLM']['completed'] != 1319:
        raise ValueError('Missing regraded candidate or reference requests')
    terminal = client / 'runs/sglang' / FAILED_RUN / 'state.json'
    if digest(terminal) != ORIGINAL_FAILED_STATE_SHA or binding['original_terminal_SHA256'] != ORIGINAL_FAILED_STATE_SHA:
        raise ValueError('Original rejected scored run has changed')

    control = client / 'SG_PD_g1317_full_GSM8K_control_g1496'
    names = ('gate.json', 'actual_launch_handle.json', 'worker_handle.json', 'terminal.json')
    if {name: digest(control / name) for name in names} != binding['quality_terminal_controls_sha256']:
        raise ValueError('Quality controller terminal evidence does not match the regrade')
    if digest(control / 'gate.json') != QUALITY_GATE_SHA:
        raise ValueError('Unexpected quality launch gate')
    launch = document(control / 'actual_launch_handle.json')
    worker = document(control / 'worker_handle.json')
    outcome = document(control / 'terminal.json')
    if launch['controller_PID'] != worker['controller_PID'] or worker['controller_PID'] != 3118006:
        raise ValueError('Unexpected GSM8K controller identity')
    if worker['docker_exec_PID'] != 3118007 or outcome['exit_code'] != 0:
        raise ValueError('GSM8K worker did not reach clean terminal')
    try:
        command = (Path('/proc') / str(worker['controller_PID']) / 'cmdline').read_bytes()
    except FileNotFoundError:
        command = b''
    if b'launch_SG_G1317_independent_quality_g1496.py' in command:
        raise ValueError('The original quality controller is still live')
    candidate = client / 'SG_PD_g1317_full_GSM8K_g1496'
    if digest(candidate / 'results.jsonl') != proof['SGLang']['results_jsonl_sha256']:
        raise ValueError('Quality raw results changed after independent regrade')
    return {
        'matched_capture_scope': 'unscored_profile_after_independently_regraded_GSM8K',
        'quality_proof_SHA256': proof_sha,
        'quality_terminal_controls_sha256': binding['quality_terminal_controls_sha256'],
        'original_scored_attempt_accepted': False,
        'verified_GSM8K_requests': proof['both_runs_all_request_bytes_verified'],
    }
