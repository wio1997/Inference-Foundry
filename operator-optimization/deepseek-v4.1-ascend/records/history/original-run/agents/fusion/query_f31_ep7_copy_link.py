#!/usr/bin/env python3
"""Read only one predetermined EP7 builder/copy sample from saved profiler DB."""
import hashlib,json,sqlite3
p='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp0_pp0_tp7_dcp0_ep7_rank7_4869_20261028020238581_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_7.db'
h=hashlib.sha256()
with open(p,'rb') as f:
 for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
if h.hexdigest()!='8b8e72a61e2521f0d91050bc53ab852ddf26dd4e40e2d5ec5d3973652fd807c6':raise RuntimeError('database content does not match archived hash')
c=sqlite3.connect('file:'+p+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
q='''select t.startNs,t.endNs,t.streamId,t.taskId,t.globalTaskId,t.taskType,t.modelId,s.value as compute_name,a.startNs as cann_start,a.endNs as cann_end,a.globalTid,sn.value as cann_name from TASK t left join COMPUTE_TASK_INFO i on t.globalTaskId=i.globalTaskId left join STRING_IDS s on i.name=s.id left join CANN_API a on t.connectionId=a.connectionId left join STRING_IDS sn on a.name=sn.id where t.streamId=47 and t.taskId between 39195 and 39201 order by t.taskId'''
rows=[dict(r) for r in c.execute(q)]
assert [r['taskId'] for r in rows]==list(range(39195,39202))
q='''select p.startNs,p.endNs,p.globalTid,p.connectionId,p.callchainId,p.type,n.value name,ds.value dtypes,sh.value shapes from PYTORCH_API p left join STRING_IDS n on p.name=n.id left join STRING_IDS ds on p.inputDtypes=ds.id left join STRING_IDS sh on p.inputShapes=sh.id where p.globalTid=? and cast(p.startNs as integer)<=? and cast(p.endNs as integer)>=? order by cast(p.startNs as integer)'''
host=[]
for row in rows[:3]:
 t=row['cann_start'];host.append({'device_task':row['taskId'],'records':[dict(r) for r in c.execute(q,(row['globalTid'],t,t))]})
print(json.dumps({'db_sha256':h.hexdigest(),'scope':'EP7 stream 47 tasks 39195..39201 and same-thread host API containing first three CANN starts','tasks':rows,'host':host,'limitations':['Same stream order and host dequeue do not prove tensor identity or same Python call.','No device copy source or destination addresses, copy sizes, Python stack or cross-stream dependence are established.']},ensure_ascii=False,indent=2))
