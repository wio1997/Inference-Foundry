"""Attribute frozen SGLang device tasks to CPU forward spans."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from itertools import groupby
import json
import os
from pathlib import Path
import sqlite3

from attribute_SG_prefill_steps_g1355 import (
    SPARSE_OPERATORS, build_index, cpu_calls, device_tasks, enclosing,
    file_hash, interval_summary, scope_rows,
)
from parse_SG_step_graph_g1213 import optional_integer, sqlite_integer


def framework_calls(db):
    """The SDK maps runtime connection IDs to grouped framework API IDs."""
    rows = db.execute("""
        SELECT relation.connectionId, api.startNs, api.endNs, api.globalTid
        FROM CONNECTION_IDS relation
        JOIN PYTORCH_API api ON api.connectionId = relation.id
        JOIN ENUM_API_TYPE kind ON kind.id = api.type
        WHERE relation.connectionId IS NOT NULL AND kind.name = 'op'
        ORDER BY CAST(relation.connectionId AS INTEGER)
    """)
    return groupby(rows, key=lambda row: sqlite_integer(row[0]))


def take_connection(cursor, current, connection):
    while current is not None and current[0] < connection:
        current = next(cursor, None)
    if current is not None and current[0] == connection:
        rows = list(current[1])
        return rows, next(cursor, None)
    return [], current


def api_scope(rows, index, origin):
    if not rows:
        return None, f'missing_{origin}_api'
    if len(rows) > 1:
        return None, f'ambiguous_{origin}_connection'
    begin, end, thread = map(optional_integer, rows[0][1:])
    if None in (begin, end, thread) or begin > end:
        return None, f'incomplete_{origin}_api'
    steps = enclosing(index, thread, begin, end)
    if len(steps) != 1:
        return None, f'{origin}_outside_forward' if not steps else f'{origin}_ambiguous_forward'
    return (steps[0], begin, end, thread), f'{origin}_assigned'


def matching_marker(index, scope, begin, end, thread):
    return any(scope['begin'] <= row['begin'] <= row['end'] <= scope['end']
               for row in enclosing(index, thread, begin, end))


def ownership(db, forwards, markers, einsums, communication, has_framework_bridge):
    index, marker_index, einsum_index = map(build_index, (forwards, markers, einsums))
    cann = cpu_calls(db)
    framework = framework_calls(db) if has_framework_bridge else iter(())
    cann_group = next(cann, None)
    framework_group = next(framework, None)
    counts = Counter()
    framework_counts = Counter()
    by_step = defaultdict(list)
    devices = set()
    for cid, tasks in device_tasks(db, communication):
        reason = 'missing_task_connection'
        assignment = None
        bridge_status = 'bridge_unavailable' if not has_framework_bridge else 'missing_task_connection'
        if cid is not None:
            direct_rows, cann_group = take_connection(cann, cann_group, cid)
            bridge_rows, framework_group = take_connection(framework, framework_group, cid)
            direct, reason = api_scope(direct_rows, index, 'CANN')
            verified_cann = (
                len(direct_rows) == 1
                and reason != 'incomplete_CANN_api'
            )
            bridge = None
            if has_framework_bridge:
                bridge_status = 'no_unique_CANN_api'
                if verified_cann:
                    bridge, bridge_status = api_scope(bridge_rows, index, 'framework')
            if direct and bridge and direct[0]['step'] != bridge[0]['step']:
                reason = 'conflicting_direct_and_framework_forward'
            elif bridge:
                assignment, reason = bridge, 'framework_assigned'
            elif direct:
                assignment = direct
        if assignment:
            forward, begin, end, thread = assignment
            tagged = matching_marker(marker_index, forward, begin, end, thread)
            in_einsum = matching_marker(einsum_index, forward, begin, end, thread)
        for _, start, end, device, compute, comm, op in tasks:
            framework_counts[bridge_status] += 1
            device = optional_integer(device)
            if device is None:
                counts['missing_device_id'] += 1
                continue
            devices.add(device)
            counts[reason] += 1
            if not assignment:
                continue
            start, end = map(sqlite_integer, (start, end))
            assert 0 <= start <= end
            families = set()
            if compute is not None:
                families.add('compute')
                if op in SPARSE_OPERATORS:
                    families.add('shared_kv_sparse')
                if tagged:
                    families.add('index_score_marker')
                elif in_einsum:
                    families.add('unverified_einsum')
                if len(families) == 1:
                    families.add('other_compute')
            if comm is not None:
                families.add('communication')
            assert families
            by_step[forward['step']].append((start, end, families))
    assert sum(counts.values()) == sum(framework_counts.values())
    return by_step, counts, framework_counts, devices


def analyze(path, expected_hash):
    path = path.resolve(strict=True)
    wal = path.with_name(path.name + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0, 'Use the frozen complete database'
    before = path.stat()
    digest = file_hash(path)
    assert digest == expected_hash
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA temp_store=FILE')
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'PYTORCH_API', 'CANN_API', 'TASK', 'STRING_IDS', 'COMPUTE_TASK_INFO'} <= tables
        conflict = db.execute("""
            SELECT globalTaskId
            FROM COMPUTE_TASK_INFO GROUP BY globalTaskId
            HAVING COUNT(DISTINCT opType) > 1 LIMIT 1
        """).fetchone()
        assert conflict is None, 'Conflicting compute operator metadata'
        comm = 'COMMUNICATION_TASK_INFO' in tables
        bridge = {'CONNECTION_IDS', 'ENUM_API_TYPE'} <= tables
        forwards, markers, einsums, unparsed = scope_rows(db)
        assert forwards
        by_step, assignment, framework_evidence, devices = ownership(
            db, forwards, markers, einsums, comm, bridge
        )

    after = path.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino, before.st_size, before.st_mtime_ns
    )
    assert len(devices) == 1, 'Analyze one physical rank database at a time'
    steps = []
    for forward in forwards:
        span = forward['span']
        tasks = by_step[forward['step']]
        steps.append({
            'step': forward['step'],
            'mode': span['mode'],
            'batch_size': span['batch_size'],
            'logical_fields': span['actual_query_KV_aggregate_fields'],
            'CPU_forward_duration_ns': forward['end'] - forward['begin'],
            'attributed_device_task_count': len(tasks),
            'device_attribution': interval_summary(tasks),
        })
    attributed = sum(step['attributed_device_task_count'] for step in steps)
    assert attributed == assignment['framework_assigned'] + assignment['CANN_assigned']
    return {
        'database_sha256': digest,
        'device_id': next(iter(devices)),
        'steps': steps,
        'task_assignment_counts': dict(assignment),
        'framework_connection_evidence': dict(framework_evidence),
        'communication_metadata_present': comm,
        'explicit_index_score_CPU_markers': len(markers),
        'generic_einsum_CPU_spans': len(einsums),
        'unparsed_step_labels': unparsed,
        'scope': ('Unique CANN device connection plus exact framework API binding or direct '
                  'CANN CPU span containment. Assignment and unmatched counts cover compute '
                  'and communication task rows. Device intervals can overlap between steps.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['SQLITE_TMPDIR'] = str(args.output.parent.resolve(strict=True))
    report = analyze(args.database, args.sha256)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
