#!/usr/bin/env python3
"""Admit eight raw cohort5 CANN profiler windows before offline parsing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--arm-dir', type=Path, required=True)
    cli.add_argument('--run-ts', required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    arm = args.arm_dir.resolve()
    need(args.run_ts == 'LOOP081-RUN610-B', 'run identity')
    upstream = {}
    for name, status in [('basis', 'diagnostic_compact_basis_all8_admitted'),
                         ('product', 'product_identity_pass'),
                         ('dispatch', 'scoped_dispatch_identity_pass')]:
        path = arm / f'{name}_admission.json'
        obj = load(path)
        need(obj['status'] == status and obj['run_ts'] == args.run_ts,
             f'{name} admission')
        upstream[name] = sha(path)
    need(load(arm / 'dispatch_admission.json')['product_admission_sha256'] == upstream['product'],
         'dispatch/Product admission SHA join')
    basis_admission = load(arm / 'basis_admission.json')
    product_admission = load(arm / 'product_admission.json')
    repo = arm.parents[4]
    root = arm / 'profile' / 'cohort5'
    need(root.is_dir(), 'selected cohort profile root missing')
    expected_windows = {f'rank{rank}_window.json' for rank in range(8)}
    windows = {p.name for p in root.glob('*_window.json')}
    need(windows == expected_windows, f'profile window set {windows}')
    raw_hashes = {}
    records = []
    for rank in range(8):
        basis_path = arm / 'basis' / f'rank{rank}_cohort5.json'
        runtime_path = arm / 'runtime' / f'rank{rank}_cohort5.json'
        product_path = arm / 'product' / f'rank{rank}_cohort5.json'
        for path in (basis_path, runtime_path):
            relative = str(path.relative_to(repo))
            need(basis_admission['source_sha256'].get(relative) == sha(path),
                 f'Basis/Runtime SHA join rank {rank}')
        need(product_admission['raw_sha256'].get(str(product_path)) == sha(product_path),
             f'Product SHA join rank {rank}')
        basis = load(basis_path)
        worker = load(product_path)
        need((basis['rank'], basis['cohort'], basis['run_ts']) ==
             (rank, 5, args.run_ts) and basis['cycles'] > 66 and
             (worker['rank'], worker['cohort'], worker['run_ts']) ==
             (rank, 5, args.run_ts), f'same-W0 cohort identity rank {rank}')
        marks = {row['kind']: row['t_ns'] for row in worker['marks']
                 if row['kind'] in ('runtime_built_host', 'serve_start_host', 'serve_synced_host')}
        need(set(marks) == {'runtime_built_host', 'serve_start_host', 'serve_synced_host'},
             f'Host marks rank {rank}')
        window_path = root / f'rank{rank}_window.json'
        window = load(window_path)
        need((window['rank'], window['first_cycle'], window['cycle_count']) ==
             (rank, 64, 3) and 0 < window['started_ns'] < window['stopped_ns'],
             f'profile window identity rank {rank}')
        roots = list(root.glob(f'rank{rank}_*_ascend_pt'))
        need(len(roots) == 1, f'raw session cardinality rank {rank}: {len(roots)}')
        session = roots[0]
        info_path = session / f'profiler_info_{rank}.json'
        info = load(info_path)
        need(info['rank_id'] == rank and
             info['config']['common_config']['activities'] ==
             ['ProfilerActivity.CPU', 'ProfilerActivity.NPU'],
             f'profile rank/activity rank {rank}')
        experimental = info['config'].get('experimental_config', {})
        need(experimental.get('_profiler_level') == 'Level0' and
             experimental.get('_aic_metrics') == 'ACL_AICORE_NONE',
             f'exact Level0/no-counter profiler rank {rank}')
        start_mono = info['start_info']['start_monotonic']
        stop_mono = info['end_info']['MonotonicTimeEnd']
        need(marks['runtime_built_host'] <= marks['serve_start_host'] <=
             start_mono < stop_mono <= marks['serve_synced_host'],
             f'profile/cohort Host monotonic join rank {rank}')
        prof_dirs = list(session.glob('PROF_*'))
        need(len(prof_dirs) == 1, f'raw PROF dir rank {rank}')
        prof = prof_dirs[0]
        device_dirs = list(prof.glob('device_*'))
        need(len(device_dirs) == 1, f'raw device dir rank {rank}')
        device_data = [p for p in (device_dirs[0] / 'data').glob('*')
                       if p.is_file() and not p.name.endswith(('.done', '.complete'))
                       and p.stat().st_size > 0]
        host_data = [p for p in (prof / 'host' / 'data').glob('*')
                     if p.is_file() and not p.name.endswith(('.done', '.complete'))
                     and p.stat().st_size > 0]
        need(device_data and host_data and (device_dirs[0] / 'data' / 'all_file.complete').exists(),
             f'raw device/host completeness rank {rank}')
        files = [window_path] + sorted(p for p in session.rglob('*') if p.is_file())
        need(info_path in files and len(files) >= 20, f'raw file cardinality rank {rank}')
        for path in files:
            key = str(path.relative_to(arm))
            raw_hashes[key] = sha(path)
        records.append({'rank': rank, 'window': str(window_path.relative_to(arm)),
                        'raw_session': str(session.relative_to(arm)),
                        'basis_cycles': basis['cycles'],
                        'request_ids': basis['req_ids'],
                        'profiler_start_monotonic_ns': start_mono,
                        'profiler_stop_monotonic_ns': stop_mono,
                        'raw_file_count': len(files),
                        'device_nonempty_count': len(device_data),
                        'host_nonempty_count': len(host_data),
                        'profiler_level': experimental.get('_profiler_level'),
                        'sha256_window': raw_hashes[str(window_path.relative_to(arm))]})
    need(len({r['raw_session'] for r in records}) == 8, 'duplicate raw sessions')
    result = {
        'status': 'all8_cohort5_raw_profile_admitted',
        'scope': 'instrumented Current; profiler.stop has a cycle66-end device fence; prior cycle63 device tail may enter the raw window and cycle66 drain is forced. Native tasks need actual correlation/Graph generation before cycle attribution; no unperturbed duration or finite Bound',
        'run_ts': args.run_ts,
        'upstream_admission_sha256': upstream,
        'records': records,
        'raw_sha256': raw_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'ranks': len(records),
                      'raw_files': len(raw_hashes)}))


if __name__ == '__main__':
    main()
