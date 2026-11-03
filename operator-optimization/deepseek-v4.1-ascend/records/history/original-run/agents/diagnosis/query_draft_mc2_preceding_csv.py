#!/usr/bin/env python3
"""Read two fixed historical CSVs for the six selected MC2 rows, no NPU."""
import argparse, csv, hashlib, json, sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--trace-root', required=True)
args = ap.parse_args()
spec = json.loads(FROZEN_INPUT)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def ns(value):
    x = Decimal(value.strip()) * 1000
    if x != x.to_integral_value():
        raise ValueError('sub-nanosecond time')
    return int(x)

results = []
for rank, targets in spec['ranks'].items():
    path = Path(args.trace_root) / rank / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
    if sha(path) != spec['csv_sha256_by_rank'][rank]:
        raise RuntimeError('CSV identity mismatch')
    dbs = list(path.parent.glob('ascend_pytorch_profiler_*.db'))
    if len(dbs) != 1 or sha(dbs[0]) != spec['db_sha256_by_rank'][rank]:
        raise RuntimeError('DB identity or cardinality mismatch')
    found = {str(t['local_op_id']): [] for t in targets}
    nearby = {str(t['local_op_id']): [] for t in targets}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = ('Name', 'Type', 'Model ID', 'Stream ID', 'Task ID', 'Start Time(us)', 'Duration(us)', 'Wait Time(us)')
        if not set(fields).issubset(reader.fieldnames or ()):
            raise RuntimeError('missing required fields')
        for row in reader:
            for t in targets:
                target = t['fields']
                if row['Model ID'] != target['Model ID'] or row['Stream ID'] != target['Stream ID']:
                    continue
                try:
                    start = ns(row['Start Time(us)'])
                    end = start + ns(row['Duration(us)'])
                except (InvalidOperation, ValueError):
                    continue
                tstart = ns(target['Start Time(us)'])
                tend = tstart + ns(target['Duration(us)'])
                key = str(t['local_op_id'])
                v = {k: row[k] for k in fields}
                v.update(start_ns=start, end_ns=end)
                if row['Type'] == target['Type'] and row['Task ID'] == target['Task ID'] and start == tstart and end == tend:
                    found[key].append(v)
                elif tstart - 1_000_000 <= start < tstart:
                    nearby[key].append(v)
                    if len(nearby[key]) > 1000:
                        raise RuntimeError('bounded neighborhood overflow')
    con = sqlite3.connect(dbs[0].resolve().as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    devices = con.execute('SELECT DISTINCT deviceId FROM RANK_DEVICE_MAP').fetchall()
    if len(devices) != 1:
        raise RuntimeError('device cardinality mismatch')
    for t in targets:
        key = str(t['local_op_id'])
        if len(found[key]) != 1:
            raise RuntimeError('target identity not unique')
        target = found[key][0]
        for k in fields:
            if target[k] != t['fields'][k]:
                raise RuntimeError('selected target field mismatch')
        pre = [x for x in nearby[key] if x['end_ns'] <= target['start_ns']]
        overlap = [x for x in nearby[key] if x['end_ns'] > target['start_ns']]
        if not pre:
            raise RuntimeError('no preceding task within one millisecond')
        last_end = max(x['end_ns'] for x in pre)
        last_start = max(x['start_ns'] for x in pre)
        last = [x for x in pre if x['end_ns'] == last_end]
        wait_ns = ns(target['Wait Time(us)'])
        matching = [x for x in pre if target['start_ns'] - x['end_ns'] == wait_ns]
        # Include zero-duration event rows. TASK may contain nested/duplicate records.
        task_rows = [dict(x) for x in con.execute('''SELECT t.startNs,t.endNs,t.streamId,t.modelId,t.taskId,t.globalTaskId,t.connectionId,s.value AS task_type_name
            FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType
            WHERE t.deviceId=? AND t.modelId=? AND t.streamId=? AND t.startNs>=? AND t.startNs<?
            ORDER BY t.startNs,t.endNs LIMIT 1001''',
            (devices[0][0], int(target['Model ID']), int(target['Stream ID']), target['start_ns']-1000000, target['start_ns']))]
        if len(task_rows) > 1000:
            raise RuntimeError('bounded DB neighborhood overflow')
        db_anchor = [dict(x) for x in con.execute('''SELECT t.startNs,t.endNs,t.streamId,t.modelId,t.taskId,t.globalTaskId,t.connectionId,s.value AS task_type_name
            FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType
            WHERE t.deviceId=? AND t.modelId=? AND t.streamId=? AND t.taskId=? AND t.startNs<? AND t.endNs>? LIMIT 21''',
            (devices[0][0], int(target['Model ID']), int(target['Stream ID']), int(target['Task ID']), target['end_ns'], target['start_ns']))]
        if not db_anchor or len(db_anchor)>20:
            raise RuntimeError('DB target identity did not intersect uniquely bounded window')
        task_pre = [x for x in task_rows if x['endNs'] <= target['start_ns']]
        task_last_end = max((x['endNs'] for x in task_pre), default=None)
        results.append({'rank': rank, 'local_op_id': t['local_op_id'],
            'target': target, 'search_window_ns': 1000000,
            'preceding_count': len(pre), 'overlap_count': len(overlap),
            'latest_end_rows': last, 'latest_start_rows': [x for x in pre if x['start_ns'] == last_start],
            'gap_after_latest_end_ns': target['start_ns'] - last_end,
            'wait_time_ns': wait_ns, 'gap_minus_wait_ns': target['start_ns'] - last_end - wait_ns,
            'exact_wait_match_rows': matching, 'overlap_rows': overlap,
            'last_10_preceding_rows': sorted(pre, key=lambda x:(x['end_ns'],x['start_ns']))[-10:],
            'db_anchor_rows': db_anchor, 'db_rows': task_rows,
            'db_latest_end_rows': [x for x in task_pre if x['endNs']==task_last_end],
            'db_gap_after_latest_end_ns': None if task_last_end is None else target['start_ns']-task_last_end})
    con.close()
print(json.dumps({'scope':'six frozen MC2 targets; same rank, numeric model and stream, CSV rows with starts in prior 1 ms',
    'csv_sha256_by_rank': spec['csv_sha256_by_rank'], 'db_sha256_by_rank': spec['db_sha256_by_rank'], 'samples': results,
    'limitations':['CSV kernel rows are not a complete task/event timeline.',
      'Previous CSV row is not proven to be the exporter Wait Time predecessor.',
      'Rank-local exported timestamps only; no host/device or cross-rank differences.',
      'Gap does not prove device idle or internal wait; same-stream presence does not prove tensor dependency.']},indent=2))
