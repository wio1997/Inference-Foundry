#!/usr/bin/env python3
"""Bounded EP8 query: exact producer kernel names and six frozen layer gaps."""
import argparse, hashlib, json, sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);a=ap.parse_args()
T=Path(a.task_root); db=Path(a.db).resolve(strict=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
dbsha=sha(db)
if dbsha!='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58':raise RuntimeError('DB hash mismatch')
boundary=T/'agents/diagnosis/engram-boundary-controls.json';metrics=T/'captured-layer-attribution/module-call-metrics.json'
b=json.loads(boundary.read_text());m=json.loads(metrics.read_text())
samples=[s for s in b['samples'] if '_ep8_' in s['rank'] and s['after_layer'] in (0,13)]
if len(samples)!=6:raise RuntimeError('Expected six EP8 gaps')
graph_starts={1:1793152958870967561,2:1793152960186061405,3:1793152960261682034}
ends={1:1793152958888461091,2:1793152960199259541,3:1793152960275759213}
con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
con.execute('PRAGMA query_only=ON');con.execute('PRAGMA temp_store=MEMORY')
def bounded(sql,params=(),cap=100):
 rows=[dict(r) for r in con.execute(sql,params).fetchmany(cap+1)]
 return {'rows':rows[:cap], 'truncated':len(rows)>cap,'limit':cap}
cols='''t.startNs,t.endNs,t.deviceId,t.streamId,t.modelId,t.taskId,t.globalTaskId,t.connectionId,t.taskType,
 st.value AS task_type_name,sn.value AS compute_name,so.value AS compute_op_type'''
joins=''' FROM TASK t LEFT JOIN STRING_IDS st ON st.id=t.taskType
 LEFT JOIN COMPUTE_TASK_INFO i ON i.globalTaskId=t.globalTaskId
 LEFT JOIN STRING_IDS sn ON sn.id=i.name LEFT JOIN STRING_IDS so ON so.id=i.opType'''
producer_names=('_engram_req_index_kernel','_write_hash_cache_kernel','_hash_ids_kernel','_engram_host_uva_gather_dequant_kernel')
producer=[]
for run in (1,2,3):
 start=graph_starts[run]-10_000_000;end=ends[run]
 result=bounded('SELECT '+cols+joins+''' WHERE t.deviceId=8 AND t.startNs<? AND t.endNs>=?
 AND (sn.value IN (?,?,?,?) OR so.value IN (?,?,?,?)) ORDER BY t.startNs,t.streamId''',
 (end,start,*producer_names,*producer_names),100)
 producer.append({'run':run,'lookback_start_ns':start,'graph_start_ns':graph_starts[run], 'end_ns':end,'exact_named_producers':result})
gaps=[]
for s in samples:
 q=[x for x in m if x['rank']==s['rank'] and x['run_in_model']==s['run'] and x['layer']==s['after_layer']]
 if len(q)!=1:raise RuntimeError('Gap anchor not unique')
 start=q[0]['window_end_ns'];end=start+s['gap_ns']
 events=bounded('SELECT '+cols+joins+''' WHERE t.deviceId=8 AND t.startNs<? AND t.endNs>=?
 AND (st.value LIKE '%EVENT%' OR st.value LIKE '%NOTIFY%' OR st.value LIKE '%WAIT%' OR st.value LIKE '%SYNC%')
 ORDER BY t.startNs,t.streamId''',(end,start),150)
 gaps.append({'run':s['run'],'after_layer':s['after_layer'],'start_ns':start,'end_ns':end,'event_task_rows':events})
print(json.dumps({'scope':'EP8 device-8, three graph-window + 10ms lookback exact producer names; six frozen layer gap event rows',
 'db_sha256':dbsha,'boundary_sha256':sha(boundary),'metrics_sha256':sha(metrics),'producer_names':producer_names,
 'producer_windows':producer,'gap_windows':gaps,
 'limitations':['Lookback is arbitrary, temporal coincidence does not establish same request or causal edge.',
 'Absence only concerns exact names visible to this exporter and these windows.',
 'Event task records do not include event-object identity; IDs and shared connection alone cannot pair a record and wait.',
 'No cross-rank or host-device timestamp comparison; source path and runtime branch remain unproven.']},indent=2,ensure_ascii=False))
con.close()
