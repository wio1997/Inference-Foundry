"""Attribute named HCCL transfers and device compute to verified forward steps."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from heapq import merge
from itertools import groupby
import json
import os
from pathlib import Path
import sqlite3
import time

from attribute_SG_prefill_steps_g1355 import build_index, cpu_calls, file_hash, scope_rows
from attribute_SG_prefill_steps_g1420 import api_scope, framework_calls, take_connection
from parse_SG_step_graph_g1213 import optional_integer, sqlite_integer


def task_groups(db):
    rows = db.execute("""
        WITH kernels AS (
            SELECT c.globalTaskId,
                   MAX(COALESCE(s.value IN ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV'),0)) AS compute,
                   MAX(COALESCE(s.value = 'AI_CPU',0)) AS ai_cpu,
                   MAX(s.value IS NULL OR s.value NOT IN
                       ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','AI_CPU')) AS other_compute
            FROM COMPUTE_TASK_INFO c LEFT JOIN STRING_IDS s ON c.taskType = s.id
            GROUP BY c.globalTaskId
        ), communication_labels AS (
            SELECT c.globalTaskId,
                   CASE WHEN s.value IN ('Memcpy','Reduce_Inline') AND typeof(c.size) = 'integer' AND c.size > 0
                            THEN 'move'
                        WHEN s.value = 'Notify_Wait' AND typeof(c.size) = 'integer' AND c.size = 0 THEN 'wait'
                        WHEN s.value = 'Notify_Record' AND typeof(c.size) = 'integer' AND c.size = 0 THEN 'record'
                        ELSE 'other' END AS label
            FROM COMMUNICATION_TASK_INFO c LEFT JOIN STRING_IDS s ON c.taskType = s.id
        ), communication AS (
            SELECT globalTaskId,MAX(label='move') AS move,MAX(label='wait') AS wait,
                   MAX(label='record') AS record,MAX(label='other') AS other
            FROM communication_labels GROUP BY globalTaskId
        )
        SELECT t.connectionId,t.deviceId,t.startNs,t.endNs,
               k.compute,k.ai_cpu,k.other_compute,m.move,m.wait,m.record,m.other
        FROM TASK t LEFT JOIN kernels k ON k.globalTaskId=t.globalTaskId
        LEFT JOIN communication m ON m.globalTaskId=t.globalTaskId
        WHERE k.globalTaskId IS NOT NULL OR m.globalTaskId IS NOT NULL
        ORDER BY CAST(t.connectionId AS INTEGER)
    """)
    return groupby(rows, key=lambda row: optional_integer(row[0]))


def resolve(direct_rows, bridge_rows, index):
    direct, reason = api_scope(direct_rows, index, 'CANN')
    if len(direct_rows) != 1 or reason == 'incomplete_CANN_api':
        return None, reason
    bridge, bridge_reason = api_scope(bridge_rows, index, 'framework')
    if direct and bridge and direct[0]['step'] != bridge[0]['step']:
        return None, 'conflicting_direct_and_framework_forward'
    if bridge:
        return bridge[0]['step'], bridge_reason
    if direct:
        return direct[0]['step'], reason
    return None, reason


FAMILIES = ('compute', 'ai_cpu', 'unknown_compute_metadata', 'communication',
            'notify_wait', 'notify_record', 'other_communication_metadata')
MEASURED_INTERVALS = ('compute', 'communication', 'notify_wait')
LOW_BITS = (1 << 64) - 1


def union_duration(sorted_values):
    last = None
    total = 0
    for value in sorted_values:
        start, stop = value >> 64, value & LOW_BITS
        if last is None or start > last:
            total += stop - start
            last = stop
        elif stop > last:
            total += stop - last
            last = stop
    return total


def intervals_summary(groups):
    for values in groups.values():
        values.sort()
    compute, communication, wait = (groups[key] for key in MEASURED_INTERVALS)
    a, b, c = map(union_duration, (compute, communication, wait))
    overlap = a + b - union_duration(merge(compute, communication))
    assert 0 <= overlap <= min(a, b)
    return {'core_compute_union_ns': a, 'named_hccl_data_union_ns': b,
            'notify_wait_union_ns': c, 'compute_and_hccl_data_overlap_ns': overlap,
            'named_hccl_data_outside_compute_ns': b - overlap}


def profile_tasks(db):
    forwards, _, _, ignored = scope_rows(db)
    assert forwards
    index = build_index(forwards)
    cann, bridge = cpu_calls(db), framework_calls(db)
    dc, fc = next(cann, None), next(bridge, None)
    all_counts, counts_by_reason = Counter(), defaultdict(Counter)
    all_spans = defaultdict(list)
    by_step = defaultdict(lambda: defaultdict(list))
    devices = set()
    for cid, group in task_groups(db):
        step, reason = None, 'missing_task_connection'
        if cid is not None:
            direct, dc = take_connection(cann, dc, cid)
            framework, fc = take_connection(bridge, fc, cid)
            step, reason = resolve(direct, framework, index)
        for _, device, begin, end, *flags in group:
            families = {name for name, present in zip(FAMILIES, flags) if present}
            assert families
            assert len(families & {'communication', 'notify_wait', 'notify_record'}) <= 1
            assert len(families & {'compute', 'ai_cpu', 'unknown_compute_metadata'}) <= 1
            if {'compute', 'communication'} <= families:
                families.difference_update(('compute', 'communication'))
                families.add('ambiguous_compute_and_hccl_metadata')
            rank = optional_integer(device)
            all_counts.update(families)
            if rank is None:
                counts_by_reason['missing_device_id'].update(families)
                continue
            devices.add(rank)
            counts_by_reason[reason].update(families)
            begin, end = map(sqlite_integer, (begin, end))
            assert 0 <= begin <= end <= LOW_BITS
            value = (begin << 64) | end
            for family in families.intersection(MEASURED_INTERVALS):
                all_spans[family].append(value)
                if step is not None:
                    by_step[step][family].append(value)
    assert len(devices) == 1
    for family, count in all_counts.items():
        assert count == sum(reason[family] for reason in counts_by_reason.values())
    return {'physical_rank': next(iter(devices)), 'device_metadata_task_counts': dict(all_counts),
            'task_family_connection_outcomes': {reason: dict(counts) for reason, counts in counts_by_reason.items()},
            'all_device_intervals': intervals_summary(all_spans), 'unparsed_forward_markers': ignored,
            'steps': [{'step': f['step'], 'mode': f['span']['mode'],
                       'interval_counts_by_family': {k: len(v) for k, v in by_step[f['step']].items()},
                       'intervals': intervals_summary(by_step[f['step']])}
                      for f in forwards]}


def summarize(path, expected_digest):
    path = path.resolve(strict=True)
    before = path.stat()
    assert not any(path.with_name(path.name + ext).exists() for ext in ('-wal', '-shm', '-journal'))
    assert file_hash(path) == expected_digest
    end = time.monotonic() + 180
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA temp_store=FILE')
        db.set_progress_handler(lambda: int(time.monotonic() > end), 10000)
        result = profile_tasks(db)
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns
    )
    result['database_sha256'] = expected_digest
    result['scope'] = ('Communication intervals include only named positive-size HCCL Memcpy and '
                       'Reduce_Inline. Notify waits have their own union. Other metadata is counted by '
                       'connection outcome. Step assignment requires unique CANN and CPU containment. '
                       'Device overlap does not measure a critical path or cross-node KV transfer.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('database_sha256')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    os.environ['SQLITE_TMPDIR'] = str(args.output.parent.resolve(strict=True))
    report = summarize(args.database, args.database_sha256)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
