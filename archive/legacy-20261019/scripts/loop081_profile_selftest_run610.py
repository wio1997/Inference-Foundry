#!/usr/bin/env python3
"""CPU-only fail-closed admission fixtures for Run610 raw profiler provenance."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


RUN_TS = 'LOOP081-RUN610-B'
VALIDATOR = Path(__file__).with_name('loop081_profile_validate_run610.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    arm = root / 'evidence' / 'date' / 'run' / 'live' / 'b'
    arm.mkdir(parents=True)
    basis_hashes, product_hashes = {}, {}
    profile = arm / 'profile' / 'cohort5'
    for rank in range(8):
        basis = arm / 'basis' / f'rank{rank}_cohort5.json'
        runtime = arm / 'runtime' / f'rank{rank}_cohort5.json'
        worker = arm / 'product' / f'rank{rank}_cohort5.json'
        put(basis, {'rank': rank, 'cohort': 5, 'run_ts': RUN_TS,
                    'cycles': 70, 'req_ids': [f'r{i}' for i in range(12)]})
        put(runtime, {'rank': rank, 'cohort': 5, 'run_ts': RUN_TS})
        put(worker, {'rank': rank, 'cohort': 5, 'run_ts': RUN_TS,
                     'marks': [{'kind': 'runtime_built_host', 't_ns': 100},
                               {'kind': 'serve_start_host', 't_ns': 110},
                               {'kind': 'serve_synced_host', 't_ns': 140}]})
        for path in (basis, runtime):
            basis_hashes[str(path.relative_to(root))] = sha(path)
        product_hashes[str(worker)] = sha(worker)
        put(profile / f'rank{rank}_window.json',
            {'rank': rank, 'first_cycle': 64, 'cycle_count': 3,
             'started_ns': 10, 'stopped_ns': 20})
        session = profile / f'rank{rank}_1_ascend_pt'
        put(session / f'profiler_info_{rank}.json',
            {'rank_id': rank,
             'config': {'common_config': {'activities':
                 ['ProfilerActivity.CPU', 'ProfilerActivity.NPU']},
                 'experimental_config': {'_profiler_level': 'Level0',
                                         '_aic_metrics': 'ACL_AICORE_NONE'}},
             'start_info': {'start_monotonic': 120},
             'end_info': {'MonotonicTimeEnd': 130}})
        prof = session / 'PROF_1'
        device = prof / 'device_0' / 'data'
        host = prof / 'host' / 'data'
        device.mkdir(parents=True)
        host.mkdir(parents=True)
        (device / 'all_file.complete').write_bytes(b'')
        (device / 'stars_soc.data.0.slice_0').write_bytes(b'raw-device')
        for number in range(20):
            (host / f'trace_{number}').write_bytes(f'raw-host-{number}'.encode())
    put(arm / 'basis_admission.json',
        {'status': 'diagnostic_compact_basis_all8_admitted',
         'run_ts': RUN_TS, 'source_sha256': basis_hashes})
    put(arm / 'product_admission.json',
        {'status': 'product_identity_pass', 'run_ts': RUN_TS,
         'raw_sha256': product_hashes})
    put(arm / 'dispatch_admission.json',
        {'status': 'scoped_dispatch_identity_pass', 'run_ts': RUN_TS,
         'product_admission_sha256': sha(arm / 'product_admission.json')})
    return arm


def run(arm):
    return subprocess.run([sys.executable, str(VALIDATOR),
                           '--arm-dir', str(arm), '--run-ts', RUN_TS,
                           '--output', str(arm / 'profile_admission.json')],
                          text=True, capture_output=True)


def main():
    results = {}
    cases = ('positive', 'missing_rank', 'missing_session', 'basis_hash',
             'host_clock', 'wrong_level', 'missing_device_raw')
    for case in cases:
        with tempfile.TemporaryDirectory(prefix=f'run610_{case}_') as temp:
            arm = fixture(Path(temp) / 'repo')
            profile = arm / 'profile' / 'cohort5'
            if case == 'missing_rank':
                (profile / 'rank7_window.json').unlink()
            elif case == 'missing_session':
                import shutil
                shutil.rmtree(profile / 'rank7_1_ascend_pt')
            elif case == 'basis_hash':
                (arm / 'basis' / 'rank7_cohort5.json').write_text('{}')
            elif case == 'host_clock':
                p = profile / 'rank7_1_ascend_pt' / 'profiler_info_7.json'
                info = json.loads(p.read_text())
                info['start_info']['start_monotonic'] = 200
                put(p, info)
            elif case == 'wrong_level':
                p = profile / 'rank7_1_ascend_pt' / 'profiler_info_7.json'
                info = json.loads(p.read_text())
                info['config']['experimental_config']['_profiler_level'] = 'Level1'
                put(p, info)
            elif case == 'missing_device_raw':
                p = profile / 'rank7_1_ascend_pt' / 'PROF_1' / 'device_0' / 'data'
                (p / 'stars_soc.data.0.slice_0').unlink()
            result = run(arm)
            good = result.returncode == 0
            if good != (case == 'positive'):
                raise AssertionError((case, result.returncode, result.stdout, result.stderr))
            results[case] = 'pass' if good else 'rejected'
    print(json.dumps({'status': 'pass', 'results': results,
                      'validator_sha256': sha(VALIDATOR)}))


if __name__ == '__main__':
    main()
