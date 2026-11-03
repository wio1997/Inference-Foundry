#!/usr/bin/env python3
"""Resolve fixed host flow IDs to saved device tasks only."""
import sqlite3,pathlib,json,time
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;start=time.monotonic();c.set_progress_handler(lambda:int(time.monotonic()-start>25),100000)
ids=[102373,102386,102408,102416,102482,102582,102676]
q='SELECT t.startNs,t.endNs,t.connectionId,t.globalTaskId,t.streamId,t.taskId,t.modelId, ct.name,ct.inputDataTypes,ct.inputShapes,ct.outputDataTypes,ct.outputShapes FROM TASK t LEFT JOIN COMPUTE_TASK_INFO ct ON ct.globalTaskId=t.globalTaskId WHERE t.connectionId IN ('+','.join('?' for _ in ids)+') ORDER BY t.startNs'
rows=[dict(r) for r in c.execute(q,ids)];keys=['name','inputDataTypes','inputShapes','outputDataTypes','outputShapes'];strids=sorted({r[k] for r in rows for k in keys if r[k] is not None})
strings={}
if strids:
 q='SELECT id,value FROM STRING_IDS WHERE id IN ('+','.join('?' for _ in strids)+')';strings={r[0]:r[1] for r in c.execute(q,strids)}
for r in rows:
 for k in keys:r[k]=strings.get(r[k])
print(json.dumps({'scope':'one EP3 saved DB, fixed set of seven flow IDs','tasks':rows,'limits':'one host may have multiple mapped flows; connection and shape do not prove storage identity, no rank global ordering or whole-scope timings'},indent=2))
