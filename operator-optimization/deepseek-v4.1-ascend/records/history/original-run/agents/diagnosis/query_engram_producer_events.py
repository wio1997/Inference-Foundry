#!/usr/bin/env python3
"""EP8 existing profiler DB only: three predefined graph windows and 10 ms lookback."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--db', required=True)
args = parser.parse_args()
path = Path(args.db).resolve(strict=True)
digest = hashlib.sha256()
with path.open('rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
        digest.update(block)
sha = digest.hexdigest()
if sha != '67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58':
    raise RuntimeError('Unexpected EP8 database hash; aborting')

# Device TASK timestamps, established independently in stage 114/115. 10 ms
# lookback is a search window only and does not identify a request or replay.
windows = [
    (1, 1793152958870967561, 1793152958888461091),
    (2, 1793152960186061405, 1793152960199259541),
    (3, 1793152960261682034, 1793152960275759213),
]
con = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
con.row_factory = sqlite3.Row
con.execute('PRAGMA query_only=ON')
con.execute('PRAGMA temp_store=MEMORY')
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
required = {'TASK', 'COMPUTE_TASK_INFO', 'STRING_IDS', 'CANN_API', 'PYTORCH_API', 'CONNECTION_IDS'}
if not required <= tables:
    raise RuntimeError('Missing required tables: ' + str(sorted(required - tables)))

def bounded(sql, params=(), limit=200):
    rows = [dict(r) for r in con.execute(sql, params).fetchmany(limit + 1)]
    return {'rows': rows[:limit], 'truncated': len(rows) > limit, 'limit': limit}

schemas = {}
for name in sorted(tables):
    if any(word in name.upper() for word in ('EVENT', 'SYNC', 'TASK', 'COMMUNICATION', 'CANN_API', 'PYTORCH_API', 'CONNECTION')):
        schemas[name] = [dict(r) for r in con.execute('PRAGMA table_info("' + name.replace('"', '""') + '")')]

# Inventory proves only exported names/tables, not that profiler captured every
# event or that absence in these records means the runtime did not execute it.
names = bounded('''SELECT id,value FROM STRING_IDS WHERE
 lower(value) LIKE '%engram%' OR lower(value) LIKE '%event%' OR
 lower(value) LIKE '%notify%' OR lower(value) LIKE '%wait%' OR
 lower(value) LIKE '%sync%' OR lower(value) LIKE '%hash%'
 ORDER BY id''', limit=500)

candidate_sql = '''SELECT t.startNs,t.endNs,t.deviceId,t.streamId,t.modelId,t.taskId,
 t.globalTaskId,t.connectionId,t.taskType, st.value AS task_type_name,
 sn.value AS compute_name,so.value AS compute_op_type,
 sc.value AS communication_name,ss.value AS communication_schedule_name
 FROM TASK t
 LEFT JOIN STRING_IDS st ON st.id=t.taskType
 LEFT JOIN COMPUTE_TASK_INFO ci ON ci.globalTaskId=t.globalTaskId
 LEFT JOIN STRING_IDS sn ON sn.id=ci.name
 LEFT JOIN STRING_IDS so ON so.id=ci.opType
 LEFT JOIN COMMUNICATION_TASK_INFO ct ON ct.globalTaskId=t.globalTaskId
 LEFT JOIN STRING_IDS sc ON sc.id=ct.name
 LEFT JOIN COMMUNICATION_SCHEDULE_TASK_INFO cs ON cs.globalTaskId=t.globalTaskId
 LEFT JOIN STRING_IDS ss ON ss.id=cs.name
 WHERE t.deviceId=8 AND t.startNs < ? AND t.endNs >= ? AND (
 lower(coalesce(st.value,'') || ' ' || coalesce(sn.value,'') || ' ' ||
 coalesce(so.value,'') || ' ' || coalesce(sc.value,'') || ' ' || coalesce(ss.value,'')) LIKE '%engram%'
 OR lower(coalesce(st.value,'') || ' ' || coalesce(sn.value,'') || ' ' ||
 coalesce(so.value,'') || ' ' || coalesce(sc.value,'') || ' ' || coalesce(ss.value,'')) LIKE '%hash%'
 OR lower(coalesce(st.value,'') || ' ' || coalesce(sn.value,'') || ' ' ||
 coalesce(so.value,'') || ' ' || coalesce(sc.value,'') || ' ' || coalesce(ss.value,'')) LIKE '%event%'
 OR lower(coalesce(st.value,'') || ' ' || coalesce(sn.value,'') || ' ' ||
 coalesce(so.value,'') || ' ' || coalesce(sc.value,'') || ' ' || coalesce(ss.value,'')) LIKE '%notify%'
 OR lower(coalesce(st.value,'') || ' ' || coalesce(sn.value,'') || ' ' ||
 coalesce(so.value,'') || ' ' || coalesce(sc.value,'') || ' ' || coalesce(ss.value,'')) LIKE '%wait%'
 OR lower(coalesce(st.value,'') || ' ' || coalesce(sn.value,'') || ' ' ||
 coalesce(so.value,'') || ' ' || coalesce(sc.value,'') || ' ' || coalesce(ss.value,'')) LIKE '%sync%')
 ORDER BY t.startNs,t.streamId,t.globalTaskId'''
if not {'COMMUNICATION_TASK_INFO', 'COMMUNICATION_SCHEDULE_TASK_INFO'} <= tables:
    raise RuntimeError('Communication task tables absent; adapt query rather than assume no rows')
out = []
for run, graph_start, end in windows:
    start = graph_start - 10_000_000
    cands = bounded(candidate_sql, (end, start), limit=300)
    task_types = bounded('''SELECT t.taskType,s.value AS type_name,COUNT(*) AS n FROM TASK t
       LEFT JOIN STRING_IDS s ON s.id=t.taskType WHERE t.deviceId=8 AND t.startNs<? AND t.endNs>=?
       GROUP BY t.taskType,s.value ORDER BY n DESC''', (end, start), limit=100)
    # Exact IDs only; recorded Python/API rows are candidates, not proof of producer identity.
    connections = sorted({r['connectionId'] for r in cands['rows'] if r['connectionId'] is not None})
    links = []
    for connection in connections[:80]:
        cann = bounded('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,s.value AS name
          FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?''', (connection,), 20)
        sets = bounded('SELECT id,connectionId FROM CONNECTION_IDS WHERE connectionId=?', (connection,), 20)
        links.append({'connectionId': connection, 'cann_exact': cann, 'connection_sets': sets})
    out.append({'run':run, 'lookback_start_ns':start, 'graph_start_ns':graph_start,
                'second_matmul_end_ns':end, 'task_types':task_types, 'candidate_tasks':cands,
                'candidate_connection_links':links, 'connection_ids_truncated':len(connections)>80})
print(json.dumps({'db_sha256':sha, 'scope':'EP8 device 8, 3 predefined graph windows plus 10 ms lookback',
 'relevant_table_schemas':schemas, 'matching_exported_string_inventory':names, 'windows':out,
 'limits':['Device task timestamps are only compared within this rank.',
  '10 ms is an arbitrary search bound, not proof of the same request or graph replay.',
  'Only matching exported names can be found; unnamed or uncollected events remain unknown.',
  'Same connection numeric IDs and neighboring timestamps are candidate links, not proven tensor/data dependency.',
  'No captured wait record does not prove no runtime wait; API and device clocks are not subtracted.',
  'JOIN multiplicity can produce duplicate TASK rows; do not sum durations as independent work.']},ensure_ascii=False,indent=2))
con.close()
