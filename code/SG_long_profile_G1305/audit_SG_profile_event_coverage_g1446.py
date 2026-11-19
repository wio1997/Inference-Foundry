"""Check actual device, scheduler and logical-work coverage of exported rank databases."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from intake_SG_matched_profile_g1431 import identity_hash, read
from inventory_SG_raw_profile_g1423 import EXPECTED_RANKS, PROFILER_INFO, inventory
from parse_SG_step_graph_g1213 import span_data, sqlite_integer


ROLE_MODES = {'prefill': ('EXTEND',), 'decode': ('DRAFT', 'VERIFY')}


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


def role_coverage(root, role, identity=None, raw_receipt=None):
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
    attempts = raw_ranks & db_ranks
    receipts, rejected = {}, {}
    for info in sorted(root.glob('*_ascend_pt/profiler_info_*.json')):
        match = PROFILER_INFO.fullmatch(info.name)
        if match is None or (rank := int(match[1])) not in attempts:
            continue
        database = info.parent / 'ASCEND_PROFILER_OUTPUT' / f'ascend_pytorch_profiler_{rank}.db'
        try:
            receipts[rank] = read_events(database, rank, role)
        except (AssertionError, OSError, sqlite3.Error, ValueError) as error:
            rejected[rank] = type(error).__name__
    return {
        'role': role, 'capture_identity_SHA256': binding_sha, 'closed_raw_ranks': sorted(raw_ranks),
        'database_file_ranks': sorted(db_ranks), 'device_event_ranks': sorted(receipts),
        'raw_complete_rank_needing_export': sorted(raw_ranks - db_ranks), 'rejected_database_ranks': rejected,
        'all8_device_event_coverage': receipts.keys() == EXPECTED_RANKS,
        'all8_expected_scheduler_modes_seen': (receipts.keys() == EXPECTED_RANKS
                                               and all(row['expected_scheduler_modes_seen'] for row in receipts.values())),
        'all8_expected_scheduler_logical_fields_covered': (
            receipts.keys() == EXPECTED_RANKS and all(row['expected_scheduler_modes_logical_coverage']
                                                      for row in receipts.values())),
        'ranks': [receipts[rank] for rank in sorted(receipts)],
    }


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--role-root', type=Path, required=True)
    cli.add_argument('--role', choices=ROLE_MODES, required=True)
    cli.add_argument('--identity', type=Path)
    cli.add_argument('--raw-receipt', type=Path)
    args = cli.parse_args()
    if bool(args.identity) != bool(args.raw_receipt):
        cli.error('Both --identity and --raw-receipt are needed to bind a matched capture')
    print(json.dumps(role_coverage(args.role_root, args.role,
                                   read(args.identity) if args.identity else None,
                                   read(args.raw_receipt) if args.raw_receipt else None), sort_keys=True))
