#!/usr/bin/env python3
"""Existing EP8 metadata producer records, not inferred consumer tensor binding."""
import pathlib,sqlite3,json,hashlib
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
assert h.hexdigest()=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
targets=[(1,1793152958896510570,1793152958896542590),(2,1793152960208211062,1793152960208709553),(3,1793152960284673195,1793152960285171365)]
op=c.execute('SELECT id FROM STRING_IDS WHERE value=?',('VllmAscendQuantLightningIndexerV2Metadata',)).fetchone();assert op is not None
q='SELECT t.startNs,t.endNs,t.streamId,t.taskId,t.globalTaskId,t.modelId,t.connectionId,i.attrInfo,i.inputShapes,i.inputDataTypes,i.outputShapes FROM TASK t JOIN COMPUTE_TASK_INFO i ON t.globalTaskId=i.globalTaskId WHERE i.opType=? AND t.deviceId=8 AND t.endNs<=? ORDER BY t.endNs DESC LIMIT 2'
results=[]
for run,lo,hi in targets:
 indexers=[dict(r) for r in c.execute('SELECT globalTaskId,connectionId,streamId,taskId,modelId FROM TASK WHERE deviceId=8 AND streamId=89 AND taskId=1275 AND modelId=49 AND startNs=? AND endNs=?',(lo,hi))];assert len(indexers)==1
 rows=[dict(r) for r in c.execute(q,(op['id'],lo))];assert len(rows)==2
 for r in rows:
  for field in ('attrInfo','inputShapes','inputDataTypes','outputShapes'):
   item=c.execute('SELECT value FROM STRING_IDS WHERE id=?',(r[field],)).fetchone();r[field]=item['value'] if item else None
  r['globalTaskId_count']=c.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?',(r['globalTaskId'],)).fetchone()[0]
  r['before_target_ns']=lo-r['endNs']
  con=r['connectionId'];r['launch']=[dict(x) for x in c.execute('SELECT a.startNs,a.endNs,a.globalTid,s.value name FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId=?',(con,))]
  ids=[x[0] for x in c.execute('SELECT id FROM CONNECTION_IDS WHERE connectionId=?',(con,))];r['host_apis']=[]
  for id in ids:
   r['host_apis'] += [dict(x) for x in c.execute('SELECT p.connectionId,p.globalTid,p.callchainId,p.inputShapes,p.inputDtypes,s.value name FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.connectionId=?',(id,))]
 results.append({'run':run,'indexer':indexers[0],'nearest_two_metadata':rows})
print(json.dumps({'db_sha256':h.hexdigest(),'metadata_task_count':c.execute('SELECT count(*) FROM TASK t JOIN COMPUTE_TASK_INFO i ON t.globalTaskId=i.globalTaskId WHERE i.opType=? AND t.deviceId=8',(op['id'],)).fetchone()[0],'samples':results,'limits':'metadata selection based only on preceding exported device timestamps; shape and temporal proximity do not bind its output tensor, request, layer, or replay to target. Length tensor elements unavailable.'},indent=2))
