"""Check actual device, scheduler and logical-work coverage of exported rank databases."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3

from intake_SG_matched_profile_g1431 import identity_hash, read
from inventory_SG_raw_profile_g1423 import EXPECTED_RANKS, PROFILER_INFO, inventory
from parse_SG_step_graph_g1213 import span_data, sqlite_integer


ROLE_MODES = {'prefill': ('EXTEND',), 'decode': ('DRAFT', 'VERIFY')}
RANK_ANALYSIS_DIRECTORIES = {'ASCEND_PROFILER_OUTPUT', 'logs'}
SESSION_ANALYSIS_DIRECTORIES = {'analyze', 'mindstudio_profiler_log', 'mindstudio_profiler_output'}


def snapshot(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def read_events(path, rank, role):
    before = snapshot(path)
    assert not any(path.with_name(path.name + suffix).exists() for suffix in ('-wal', '-journal'))
    mode_steps, logical_steps = Counter(), Counter()
    work = defaultdict(Counter)
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'TASK', 'COMPUTE_TASK_INFO', 'PYTORCH_API', 'CANN_API', 'STRING_IDS'} <= tables
        device_rows, identified, first_rank, last_rank, begin, end, backward = db.execute("""
            SELECT COUNT(*), COUNT(deviceId), MIN(deviceId), MAX(deviceId),
                MIN(CAST(startNs AS INTEGER)), MAX(CAST(endNs AS INTEGER)),
                SUM(CASE WHEN startNs IS NULL OR endNs IS NULL OR CAST(startNs AS INTEGER) <= 0
                    OR CAST(endNs AS INTEGER) < CAST(startNs AS INTEGER) THEN 1 ELSE 0 END)
            FROM TASK
        """).fetchone()
        assert device_rows > 0 and identified == device_rows
        assert sqlite_integer(first_rank) == sqlite_integer(last_rank) == rank
        compute = db.execute("""
            SELECT COUNT(*) FROM TASK WHERE globalTaskId IN (SELECT globalTaskId FROM COMPUTE_TASK_INFO)
        """).fetchone()[0]
        assert 0 < compute <= device_rows and not backward and 0 < begin < end
        communication = None
        if 'COMMUNICATION_TASK_INFO' in tables:
            communication = db.execute("""
                SELECT COUNT(*) FROM TASK WHERE globalTaskId IN (SELECT globalTaskId FROM COMMUNICATION_TASK_INFO)
            """).fetchone()[0]
        replay = db.execute("""
            SELECT COUNT(*) FROM CANN_API api JOIN STRING_IDS names ON api.name = names.id
                WHERE names.value = 'aclmdlRIExecuteAsync'
        """).fetchone()[0]
        unparsed = 0
        for (name,) in db.execute("""
            SELECT names.value FROM PYTORCH_API api JOIN STRING_IDS names ON api.name = names.id
                WHERE names.value LIKE 'step[%'
        """):
            step = span_data(name)
            if step is None:
                unparsed += 1
                continue
            mode = step['mode']
            mode_steps[mode] += 1
            fields = step['actual_query_KV_aggregate_fields']
            required = ('c_sq', 'c_sk') if mode == 'EXTEND' else ('g_sq', 'g_sk')
            if all(field in fields for field in required):
                logical_steps[mode] += 1
                work[mode].update(fields)
    assert snapshot(path) == before
    required = ROLE_MODES[role]
    expected_modes = all(mode_steps[mode] > 0 for mode in required)
    logical_coverage = expected_modes and not unparsed and all(logical_steps[mode] == mode_steps[mode]
                                                               for mode in required)
    return {
        'rank': rank, 'database_bytes': before[1], 'database_mtime_ns': before[2], 'actual_device_ID': rank, 'device_task_rows': device_rows,
        'device_compute_rows': compute, 'device_communication_rows': communication, 'device_window_ns': end - begin,
        'scheduler_steps_by_mode': dict(mode_steps), 'unparsed_scheduler_step_count': unparsed,
        'host_graph_replay_calls': replay, 'expected_scheduler_modes_seen': expected_modes,
        'logical_steps_by_mode': dict(logical_steps), 'logical_fields_summed_by_mode': dict(work),
        'expected_scheduler_modes_logical_coverage': logical_coverage,
    }


def raise_walk_error(error):
    raise error


def raw_fingerprints(root):
    """Hash closed CPU/device files with paths, excluding generated SDK output."""
    files = {}
    for current, directories, names in os.walk(root, onerror=raise_walk_error):
        directory = Path(current)
        if directory == root:
            directories[:] = [name for name in directories if name not in RANK_ANALYSIS_DIRECTORIES]
        elif directory.parent == root and directory.name.startswith('PROF_'):
            directories[:] = [name for name in directories if name not in SESSION_ANALYSIS_DIRECTORIES]
        if any((directory / name).is_symlink() for name in directories):
            raise ValueError('Symlink directory in the raw rank capture')
        for name in names:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Non-regular file in the raw rank capture')
            before = snapshot(path)
            digest = sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            assert snapshot(path) == before
            files[path.relative_to(root).as_posix()] = (before[1], digest.hexdigest())
    assert files
    return files


def bound_offline_database(original, staged, rank):
    staged = staged.resolve(strict=True)
    if original.resolve(strict=True) != original or staged.is_relative_to(original) or original.is_relative_to(staged):
        raise ValueError('Offline rank must be held apart from original capture')
    filename = f'ascend_pytorch_profiler_{rank}.db'
    db = staged / 'ASCEND_PROFILER_OUTPUT' / filename
    if db.resolve(strict=True) != db or not db.is_file():
        raise ValueError('Expected regular rank database in the staged copy')
    original_files, staged_files = raw_fingerprints(original), raw_fingerprints(staged)
    if original_files != staged_files:
        raise ValueError('Raw files of staged rank do not match the original')
    packed = json.dumps(original_files, sort_keys=True, separators=(',', ':')).encode()
    return db, {'matched_raw_file_count': len(original_files),
                'matched_raw_bytes': sum(size for size, _ in original_files.values()),
                'matched_raw_manifest_sha256': sha256(packed).hexdigest()}


def role_coverage(root, role, identity=None, raw_receipt=None, offline_rank_roots=None):
    root = root.resolve(strict=True)
    current = inventory(root)
    binding_sha = None
    if identity is not None:
        binding_sha = identity_hash(identity)
        assert root.parts[-2:] == (identity['capture_prefix'], role)
        assert raw_receipt['role'] == role and raw_receipt['capture_identity_sha256'] == binding_sha
        retained = {item['rank']: item['profiler_info_sha256'] for item in raw_receipt['ranks']}
        observed = {item['rank']: item['profiler_info_sha256'] for item in current['ranks']}
        assert observed == retained, 'Rank metadata differs from the post-capture intake'

    raw_ranks = set(current['raw_cpu_and_device_ranks'])
    db_ranks = set(current['parsed_db_file_ranks'])
    offline = offline_rank_roots or {}
    if offline.keys() - raw_ranks:
        raise ValueError('Offline rank requires a complete original raw capture')
    attempts = raw_ranks & (db_ranks | offline.keys())
    receipts, rejected = {}, {}
    for info in sorted(root.glob('*_ascend_pt/profiler_info_*.json')):
        match = PROFILER_INFO.fullmatch(info.name)
        if match is None or (rank := int(match[1])) not in attempts:
            continue
        try:
            if rank in offline:
                database, proof = bound_offline_database(info.parent, offline[rank], rank)
                receipts[rank] = {**read_events(database, rank, role), 'verified_offline_copy': proof}
            else:
                database = info.parent / 'ASCEND_PROFILER_OUTPUT' / f'ascend_pytorch_profiler_{rank}.db'
                receipts[rank] = read_events(database, rank, role)
        except (AssertionError, OSError, sqlite3.Error, ValueError) as error:
            rejected[rank] = type(error).__name__
    report = {
        'role': role, 'capture_identity_SHA256': binding_sha, 'closed_raw_ranks': sorted(raw_ranks),
        'database_file_ranks': sorted(db_ranks), 'device_event_ranks': sorted(receipts),
        'raw_complete_rank_needing_export': sorted(raw_ranks - db_ranks - receipts.keys()),
        'rejected_database_ranks': rejected,
        'all8_device_event_coverage': receipts.keys() == EXPECTED_RANKS,
        'all8_expected_scheduler_modes_seen': (receipts.keys() == EXPECTED_RANKS
                                               and all(row['expected_scheduler_modes_seen'] for row in receipts.values())),
        'all8_expected_scheduler_logical_fields_covered': (
            receipts.keys() == EXPECTED_RANKS and all(row['expected_scheduler_modes_logical_coverage']
                                                      for row in receipts.values())),
        'ranks': [receipts[rank] for rank in sorted(receipts)],
    }
    if offline_rank_roots is not None:
        report['verified_offline_copy_ranks'] = sorted(offline.keys() & receipts.keys())
    return report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--role-root', type=Path, required=True)
    cli.add_argument('--role', choices=ROLE_MODES, required=True)
    cli.add_argument('--identity', type=Path)
    cli.add_argument('--raw-receipt', type=Path)
    cli.add_argument('--offline-rank', action='append', nargs=2, metavar=('RANK', 'RANK_DIR'),
                     help='Closed rank copy generated by the offline exporter')
    args = cli.parse_args()
    if bool(args.identity) != bool(args.raw_receipt):
        cli.error('Both --identity and --raw-receipt are needed to bind a matched capture')
    offline = None
    if args.offline_rank:
        offline = {int(rank): Path(folder) for rank, folder in args.offline_rank}
        if len(offline) != len(args.offline_rank) or offline.keys() - EXPECTED_RANKS:
            cli.error('Offline rank IDs must be unique and in 0–7')
    print(json.dumps(role_coverage(args.role_root, args.role,
                                   read(args.identity) if args.identity else None,
                                   read(args.raw_receipt) if args.raw_receipt else None,
                                   offline), sort_keys=True))
