#!/usr/bin/env python3
"""Salvage Run610 raw profiles after its incorrect CANN start_info clock gate.

The original Run610 controller remains INVALID. This separate offline gate
does not turn the profiler trace into unperturbed timing or a typed DAG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def completed_payload(path: Path) -> bool:
    marker = path.with_name(path.name + '.done')
    if not marker.is_file():
        return False
    match = re.search(r'(?m)^filesize:(\d+)\s*$', marker.read_text())
    return bool(match) and int(match.group(1)) == path.stat().st_size


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--arm-dir', type=Path, required=True)
    cli.add_argument('--run-ts', required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    arm = args.arm_dir.resolve()
    need(args.run_ts == 'LOOP081-RUN610-B', 'run identity')
    live = arm.parent
    repo = arm.parents[4]
    pins = {
        repo / 'scripts/loop081_profile_validate_run610.py': '4fb534060d93e83a7c35cb141028d199e3f644d03fb891f748360f882e2833bc',
        arm / 'profile_validate.log': '493b464fba57c6131cede66587b677275f478d04ed238d61b95d78150bee3990',
        live / 'cleanup_status.txt': 'c116121d1db8ed3ad730ad0222105e8071ae0a203f08b97fcc5b81f9a7ecd006',
        arm / 'client_admission.json': '97a1eda04b3c38cfb5a946e9d6334fdeb94bcd682579fda8e9afe6dec4050ce1',
    }
    for path, expected in pins.items():
        need(sha(path) == expected, f'Run610 original-evidence SHA drift: {path}')
    need('ValueError: profile/cohort Host monotonic join rank 0' in
         (arm / 'profile_validate.log').read_text(), 'original failure reason changed')
    cleanup = dict(line.split('=', 1) for line in
                   (live / 'cleanup_status.txt').read_text().splitlines())
    need(set(cleanup) == {'run_exit', 'stop_exit', 'stop_verify_exit',
                          'restore_exit', 'source_sha_exit', 'source_compare_exit',
                          'script_sha_exit', 'script_compare_exit', 'final_exit'} and
         cleanup['run_exit'] == cleanup['final_exit'] == '1' and
         all(v == '0' for k, v in cleanup.items() if k not in ('run_exit', 'final_exit')),
         'Run610 invalid-run clean-stop/restore class changed')
    need((live / 'source_before.sha256').read_bytes() ==
         (live / 'source_after.sha256').read_bytes() and
         (live / 'scripts_before.sha256').read_bytes() ==
         (live / 'scripts_after.sha256').read_bytes(),
         'Run610 source/scripts not restored')
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
    client_admission = load(arm / 'client_admission.json')
    need(client_admission['status'] == 'client_two_phase_admitted', 'client admission status')
    clock = client_admission['measured_summary']['clock']
    need(clock['time_namespace'] == 'time:[4026531834]' and
         clock['begin']['monotonic_ns'] < clock['end']['monotonic_ns'] and
         clock['begin']['wall_time_ns'] < clock['end']['wall_time_ns'],
         'client wall/monotonic anchors')
    offsets = [clock[edge]['wall_time_ns'] - clock[edge]['monotonic_ns']
               for edge in ('begin', 'end')]
    need(max(offsets) - min(offsets) <= 1_000_000,
         'wall/monotonic offset drift above 1ms')
    low_offset, high_offset = min(offsets), max(offsets)
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
             (rank, 5, args.run_ts) and
             worker['time_namespace'] == clock['time_namespace'],
             f'same-W0 cohort/clock namespace rank {rank}')
        marks = {row['kind']: row['t_ns'] for row in worker['marks']
                 if row['kind'] in ('runtime_built_host', 'serve_start_host', 'serve_synced_host')}
        need(set(marks) == {'runtime_built_host', 'serve_start_host', 'serve_synced_host'},
             f'Host marks rank {rank}')
        window_path = root / f'rank{rank}_window.json'
        window = load(window_path)
        need((window['rank'], window['first_cycle'], window['cycle_count']) ==
             (rank, 64, 3) and 0 < window['started_ns'] < window['stopped_ns'],
             f'profile window identity rank {rank}')
        need(clock['begin']['wall_time_ns'] <= window['started_ns'] <
             window['stopped_ns'] <= clock['end']['wall_time_ns'],
             f'profile wall window outside measured client rank {rank}')
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
        # Installed torch_npu._C._profiler._get_monotonic() returns
        # CLOCK_MONOTONIC_RAW. end_info uses Python CLOCK_MONOTONIC.
        # Preserve both values but never order or subtract across domains.
        session_start_raw = info['start_info']['start_monotonic']
        session_end_mono = info['end_info']['MonotonicTimeEnd']
        end_offset = info['end_info']['collectionTimeEnd'] - session_end_mono
        need(abs(end_offset - low_offset) <= 1_000_000 and
             abs(end_offset - high_offset) <= 1_000_000,
             f'profiler/client local dual-clock consistency rank {rank}')
        # The 1ms margin is an explicit mapping assumption, not a proven
        # maximum clock error across the capture interval.
        offset_lo = min(low_offset, end_offset) - 1_000_000
        offset_hi = max(high_offset, end_offset) + 1_000_000
        start_lo, start_hi = window['started_ns'] - offset_hi, window['started_ns'] - offset_lo
        stop_lo, stop_hi = window['stopped_ns'] - offset_hi, window['stopped_ns'] - offset_lo
        need(marks['runtime_built_host'] <= marks['serve_start_host'] <=
             start_lo <= start_hi < stop_lo <= stop_hi <= marks['serve_synced_host'] and
             start_lo <= session_end_mono <= stop_hi,
             f'wall-mapped profile/cohort Host containment rank {rank}')
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
        device_end = device_dirs[0] / f'end_info.{rank}'
        host_end = prof / 'host' / 'end_info'
        framework_data = [session / 'FRAMEWORK' / name
                          for name in ('torch.op_range', 'torch.op_mark')]
        need(device_dirs[0].name == f'device_{rank}' and device_data and host_data and
             all(p.is_file() and p.stat().st_size > 0 for p in framework_data) and
             all(completed_payload(p) for p in device_data + host_data +
                 [device_end, host_end]),
             f'raw Level0 device/host done markers rank {rank}')
        files = [window_path] + sorted(p for p in session.rglob('*') if p.is_file())
        need(info_path in files and len(files) >= 20, f'raw file cardinality rank {rank}')
        for path in files:
            key = str(path.relative_to(arm))
            raw_hashes[key] = sha(path)
        records.append({'rank': rank, 'window': str(window_path.relative_to(arm)),
                        'raw_session': str(session.relative_to(arm)),
                        'basis_cycles': basis['cycles'],
                        'request_ids': basis['req_ids'],
                        'wall_mapped_start_monotonic_ns_range': [start_lo, start_hi],
                        'wall_mapped_stop_monotonic_ns_range': [stop_lo, stop_hi],
                        'cann_start_info_clock_monotonic_raw_ns': session_start_raw,
                        'cann_end_info_python_monotonic_ns': session_end_mono,
                        'profiler_end_info_wall_minus_monotonic_ns': end_offset,
                        'raw_file_count': len(files),
                        'device_nonempty_count': len(device_data),
                        'host_nonempty_count': len(host_data),
                        'profiler_level': experimental.get('_profiler_level'),
                        'sha256_window': raw_hashes[str(window_path.relative_to(arm))]})
    need(len({r['raw_session'] for r in records}) == 8, 'duplicate raw sessions')
    result = {
        'status': 'all8_cohort5_raw_profile_salvaged_after_invalid_validator_clock_gate',
        'scope': 'Run610 controller stays INVALID. CANN start_info uses CLOCK_MONOTONIC_RAW and end_info Python CLOCK_MONOTONIC; no cross-domain ordering. Same-namespace wall-to-monotonic mapping uses client and rank-local profiler dual-clock anchors with explicit assumed 1ms margin, not a proved global error bound or intra-window no-jump certificate. Producer .done file sizes match payloads. Profile is instrumented Current: cycle63 tail may enter, profiler.stop fences cycle66, and native tasks need actual correlation/Graph identity. No unperturbed duration or finite Bound.',
        'run_ts': args.run_ts,
        'upstream_admission_sha256': upstream,
        'original_invalid_evidence_sha256': {str(path.relative_to(repo)): expected
                                             for path, expected in pins.items()},
        'client_clock_offset_ns_range': [low_offset, high_offset],
        'records': records,
        'raw_sha256': raw_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'ranks': len(records),
                      'raw_files': len(raw_hashes)}))


if __name__ == '__main__':
    main()
