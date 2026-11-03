#!/usr/bin/env python3
"""Offline EP8 selected Indexer and same-device wait-task overlap."""
import sqlite3,pathlib,json,collections,hashlib
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db')
h=hashlib.sha256()
with p.open('rb') as f:
 for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
assert h.hexdigest()=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
samples=[(1,1793152958896510570,1793152958896542590),(2,1793152960208211062,1793152960208709553),(3,1793152960284673195,1793152960285171365)];out=[]
for run,lo,hi in samples:
 target=[dict(r) for r in c.execute('SELECT * FROM TASK WHERE deviceId=8 AND streamId=89 AND taskId=1275 AND modelId=49 AND startNs=? AND endNs=?',(lo,hi))];assert len(target)==1
 rows=[dict(r) for r in c.execute('''SELECT t.startNs,t.endNs,t.deviceId,t.streamId,t.taskId,t.globalTaskId,t.modelId,t.connectionId,s.value AS taskType,cs.value AS communicationType,ct.notifyId,ct.opId FROM TASK t LEFT JOIN STRING_IDS s ON s.id=t.taskType LEFT JOIN COMMUNICATION_TASK_INFO ct ON ct.globalTaskId=t.globalTaskId LEFT JOIN STRING_IDS cs ON cs.id=ct.name WHERE t.deviceId=8 AND t.streamId<>89 AND t.startNs<? AND t.endNs>? ORDER BY t.streamId,t.startNs''',(hi,lo))]
 for r in rows:
  r['covers_target']=r['startNs']<=lo and r['endNs']>=hi
  r['global_id_task_count']=c.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?',(r['globalTaskId'],)).fetchone()[0]
  r['communication_info_count']=c.execute('SELECT count(*) FROM COMMUNICATION_TASK_INFO WHERE globalTaskId=?',(r['globalTaskId'],)).fetchone()[0]
  if r['opId'] is not None:
   op=c.execute('SELECT o.startNs,o.endNs,o.opId,o.rankSize,o.connectionId,s.value name FROM COMMUNICATION_OP o LEFT JOIN STRING_IDS s ON o.opName=s.id WHERE o.opId=?',(r['opId'],)).fetchone();r['communication_op']=dict(op) if op else None
 groups=collections.Counter(r['taskType'] for r in rows)
 out.append({'run':run,'target_duration_ns':hi-lo,'target_globalTaskId':target[0]['globalTaskId'],'target_global_id_task_count':c.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?',(target[0]['globalTaskId'],)).fetchone()[0], 'other_task_count':len(rows),'other_type_counts':dict(groups),'covering_target_count':sum(r['covers_target'] for r in rows),'rows':rows})
print(json.dumps({'db_sha256':h.hexdigest(),'samples':out,'limits':'Same device exported interval overlap only. Reused globalTaskId across graph replays cannot identify individual replay without exact start/end. Task connection/op association does not prove tensor/event identity, resource contention or dependency.'},indent=2))
