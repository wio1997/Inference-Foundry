"""Verify the original G1317 benchmark stopped before a separate quality run."""

import json
from hashlib import sha256
from pathlib import Path


RUN_ID = 'sg_pd_g1317_aisbench_32k_g1324'
ORIGINAL_PID = 3090471
PINS = {
    'state.json': '8a5c903ee01da25f5349cd9ebf86273603b584708590c662838cffc23ec2a557',
    'cold_32768_audit.json': 'b6b512396f66899b3e8695364453b2d52778df2bc99e43d8d4367fd9239fc608',
    'cold_32768_compile_audit.json': '039ed38b38799d32b8403aa910fa5f8379c706b155e67ad363fecd08529a836f',
}


def checked_sources(run):
    documents = {}
    for name, checksum in PINS.items():
        source = (run / name).read_bytes()
        if sha256(source).hexdigest() != checksum:
            raise ValueError(f'Terminal source changed: {name}')
        documents[name] = json.loads(source)
    return documents


def verify_boundary(run, previous_control, previous_result):
    sources = checked_sources(run)
    state = sources['state.json']
    if (state['phase'], state['run_id'], state['performance_validated']) != ('failed_requires_review', RUN_ID, False):
        raise ValueError('Expected the preserved original failed benchmark')
    raw = sources['cold_32768_audit.json']
    compiler = sources['cold_32768_compile_audit.json']
    if raw['fully_validated'] is not True or raw['requests'] != raw['successful_requests'] or raw['requests'] != 32:
        raise ValueError('Expected the completed raw client audit')
    if compiler['passed'] is not False or len(compiler['changes']) != 5:
        raise ValueError('Expected the original Decode compilation failure')
    if (Path('/proc') / str(ORIGINAL_PID)).exists():
        raise ValueError('The original outer PID is present')
    paths = (run / 'hot_32768_command.json', run / 'audit_results.json', previous_control, previous_result)
    if any(path.exists() for path in paths):
        raise ValueError('Unexpected replay, combined audit or earlier GSM8K output')
    return {
        'mode': 'independent_GSM8K_after_failed_scored_attempt',
        'original_run_id': RUN_ID,
        'original_terminal_at_UTC': state['at'],
        'original_terminal_artifact_sha256': PINS,
        'original_scored_performance_accepted': False,
        'same_input_replay_started': False,
        'raw_original_cold_requests': 32,
    }
