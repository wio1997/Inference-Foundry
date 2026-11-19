"""Attribute frozen Prefill profile device tasks through exact CANN connections."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import groupby
import json
import os
from pathlib import Path
import re
import sqlite3

from parse_SG_step_graph_g1213 import optional_integer, span_data, sqlite_integer

SCORE_MARKER = re.compile(r'ds41\.profile\.index_score(?: q=\d+ h=\d+ d=\d+ k=\d+)?')
SPARSE_OPERATORS = {'_sparse_shared_kv', '_sparse_shared_kv_two_heads'}
FAMILIES = ('compute', 'communication', 'shared_kv_sparse', 'index_score_marker',
            'unverified_einsum', 'other_compute')


def file_hash(path):
    checksum = sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024**2), b''):
            checksum.update(data)
    return checksum.hexdigest()


def build_index(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['thread']].append(row)
    index = {}
    for thread, group in buckets.items():
        ordered = sorted(group, key=lambda r: (r['begin'], r['end']))
        earliest = [row['begin'] for row in ordered]
        horizon = []
        for row in ordered:
            horizon.append(max(horizon[-1], row['end']) if horizon else row['end'])
        index[thread] = ordered, earliest, horizon
    return index


def enclosing(index, thread, begin, end):
    if thread not in index:
        return []
    ordered, earliest, horizon = index[thread]
    found = []
    i = bisect_right(earliest, begin) - 1
    while i >= 0 and horizon[i] >= end:
        row = ordered[i]
        if row['begin'] <= begin <= end <= row['end']:
            found.append(row)
        i -= 1
    return found


def interval_summary(tasks):
    changes = defaultdict(Counter)
    for start, end, families in tasks:
        for family in families:
            changes[start][family] += 1
            changes[end][family] -= 1
    unions = Counter()
    if not changes:
        return {}
    counts = Counter()
    keys = sorted(changes)
    previous = keys[0]
    for stamp in keys:
        elapsed = stamp - previous
        for family, active in counts.items():
            if active:
                unions[family] += elapsed
        if counts['compute'] or counts['communication']:
            unions['compute_or_communication'] += elapsed
        if counts['compute'] and counts['communication']:
            unions['compute_overlap_communication'] += elapsed
        counts.update(changes[stamp])
        assert all(n >= 0 for n in counts.values())
        previous = stamp
    assert not any(counts.values())
    assert unions['compute'] + unions['communication'] == unions['compute_or_communication'] + unions['compute_overlap_communication']
    return {'direct_task_window_ns':keys[-1]-keys[0], 'interval_union_ns':dict(unions),
            'gap_within_direct_task_window_ns':keys[-1]-keys[0]-unions['compute_or_communication']}


def scope_rows(db):
    forwards, markers, einsums = [], [], []
    unparsed = 0
    pattern = """
        SELECT api.startNs,api.endNs,api.globalTid,names.value
        FROM PYTORCH_API api JOIN STRING_IDS names ON api.name=names.id
        WHERE names.value LIKE 'step[%' OR names.value='aten::einsum'
          OR names.value LIKE 'ds41.profile.index_score%'
    """
    for start,end,thread,name in db.execute(pattern):
        begin, end, thread = map(sqlite_integer,(start,end,thread))
        assert begin <= end
        row = {'begin':begin, 'end':end, 'thread':thread}
        if name.startswith('step['):
            data = span_data(name)
            if data is None:
                unparsed += 1
            else:
                forwards.append({**row,'span':data})
        elif name == 'aten::einsum':
            einsums.append(row)
        elif SCORE_MARKER.fullmatch(name):
            markers.append(row)
    forwards.sort(key=lambda row: (row['begin'],row['end'],row['thread']))
    for step, row in enumerate(forwards):
        row['step'] = step
    return forwards, markers, einsums, unparsed


def cpu_calls(db):
    return groupby(db.execute("""
        SELECT connectionId,startNs,endNs,globalTid FROM CANN_API
        WHERE connectionId IS NOT NULL
        ORDER BY CAST(connectionId AS INTEGER)
    """), key=lambda row:sqlite_integer(row[0]))


def device_tasks(db, available_communication):
    communication = ('SELECT DISTINCT globalTaskId FROM COMMUNICATION_TASK_INFO'
                     if available_communication else 'SELECT NULL AS globalTaskId WHERE 0')
    sql = f"""
        WITH compute AS (
            SELECT globalTaskId, MIN(opType) AS opType
            FROM COMPUTE_TASK_INFO GROUP BY globalTaskId
            HAVING COUNT(DISTINCT opType) <= 1
        ), communication AS ({communication})
        SELECT task.connectionId,task.startNs,task.endNs,task.deviceId,
               compute.globalTaskId,communication.globalTaskId,strings.value
        FROM TASK task LEFT JOIN compute ON task.globalTaskId=compute.globalTaskId
        LEFT JOIN communication ON task.globalTaskId=communication.globalTaskId
        LEFT JOIN STRING_IDS strings ON compute.opType=strings.id
        WHERE compute.globalTaskId IS NOT NULL OR communication.globalTaskId IS NOT NULL
        ORDER BY CAST(task.connectionId AS INTEGER)
    """
    return groupby(db.execute(sql),key=lambda row:optional_integer(row[0]))


def ownership(db, forwards, markers, einsums, communication):
    index, marker_index, einsum_index = map(build_index,(forwards,markers,einsums))
    counters = Counter()
    by_step = defaultdict(list)
    graph = cpu_calls(db)
    api = next(graph,None)
    devices = set()
    for cid, task_group in device_tasks(db,communication):
        if cid is not None:
            while api is not None and api[0] < cid:
                api = next(graph,None)
        matched = None
        failure = 'missing_connection' if cid is None else 'no_CANN_api_connection'
        if api is not None and cid == api[0]:
            first = next(api[1])
            if next(api[1],None) is not None:
                failure = 'ambiguous_CANN_connection'
            else:
                begin,end,thread = map(optional_integer,first[1:])
                if begin is None or end is None or thread is None or begin>end:
                    failure = 'incomplete_CANN_api'
                else:
                    step = enclosing(index,thread,begin,end)
                    if len(step) == 1:
                        scope = step[0]
                        tagged = any(scope['begin'] <= row['begin'] <= row['end'] <= scope['end']
                                     for row in enclosing(marker_index,thread,begin,end))
                        in_einsum = any(scope['begin'] <= row['begin'] <= row['end'] <= scope['end']
                                        for row in enclosing(einsum_index,thread,begin,end))
                        matched = scope['step'], tagged, in_einsum
                    else:
                        failure = 'CANN_outside_forward' if not step else 'ambiguous_CPU_forward'
        for _, start, end, device, comp, comm, op in task_group:
            device = optional_integer(device)
            if device is None:
                counters['missing_device_id'] += 1
                continue
            devices.add(device)
            if matched is None:
                counters[failure] += 1
                continue
            step_id,tagged,in_einsum = matched
            start,end = map(sqlite_integer,(start,end))
            assert 0 <= start <= end
            classes = set()
            if comp is not None:
                classes.add('compute')
                if op in SPARSE_OPERATORS:
                    classes.add('shared_kv_sparse')
                if tagged:
                    classes.add('index_score_marker')
                elif in_einsum:
                    classes.add('unverified_einsum')
                if len(classes) == 1:
                    classes.add('other_compute')
            if comm is not None:
                classes.add('communication')
            assert classes and classes <= set(FAMILIES)
            by_step[step_id].append((start,end,classes))
            counters['directly_assigned_to_forward'] += 1
    return by_step, dict(counters), sorted(devices)


def analyze(path, expected_hash):
    path = path.resolve(strict=True)
    wal = path.with_name(path.name + '-wal')
    assert not wal.exists() or wal.stat().st_size == 0, 'Use the frozen complete database'
    before = path.stat()
    digest = file_hash(path)
    assert digest == expected_hash
    db = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA temp_store=FILE')
        tables = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'PYTORCH_API','CANN_API','TASK','STRING_IDS','COMPUTE_TASK_INFO'} <= tables
        # Fail on conflicting metadata rather than silently dropping compute tasks.
        ambiguous = db.execute("""
            SELECT COUNT(*) FROM (
                SELECT globalTaskId FROM COMPUTE_TASK_INFO GROUP BY globalTaskId HAVING COUNT(DISTINCT opType)>1
            )
        """).fetchone()[0]
        assert ambiguous == 0, 'Conflicting compute operator metadata'
        comm = 'COMMUNICATION_TASK_INFO' in tables
        forwards, markers, einsums, unparsed = scope_rows(db)
        assert forwards
        by_step, assignment, devices = ownership(db,forwards,markers,einsums,comm)
    finally:
        db.close()
    after = path.stat()
    assert (after.st_ino,after.st_size,after.st_mtime_ns) == (before.st_ino,before.st_size,before.st_mtime_ns)
    assert len(devices) == 1, 'Analyze one physical rank database at a time'
    result = []
    for forward in forwards:
        span = forward['span']
        tasks = by_step.get(forward['step'],[])
        result.append({'step':forward['step'],'mode':span['mode'],'batch_size':span['batch_size'],
                       'logical_fields':span['actual_query_KV_aggregate_fields'],
                       'CPU_forward_duration_ns':forward['end']-forward['begin'],
                       'direct_device_task_count':len(tasks),'device_attribution':interval_summary(tasks)})
    return {'database_sha256':digest,'device_id':devices[0],'steps':result,'task_assignment_counts':assignment,
            'communication_metadata_present':comm,'explicit_index_score_CPU_markers':len(markers),
            'generic_einsum_CPU_spans':len(einsums),'unparsed_step_labels':unparsed,
            'scope':'Direct CANN connection and unique CPU span containment. CPU and device times use separate domains. Generic einsum is not proof of index scoring. Device windows can overlap between steps.'}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--database',type=Path,required=True)
    cli.add_argument('--sha256',required=True)
    cli.add_argument('--output',type=Path,required=True)
    args = cli.parse_args()
    os.environ['SQLITE_TMPDIR'] = str(args.output.parent.resolve(strict=True))
    value = analyze(args.database,args.sha256)
    with args.output.open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    main()
