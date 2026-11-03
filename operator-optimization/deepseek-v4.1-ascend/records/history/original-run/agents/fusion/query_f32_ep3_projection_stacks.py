#!/usr/bin/env python3
import sqlite3,pathlib,json,time
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;start=time.monotonic();c.set_progress_handler(lambda:int(time.monotonic()-start>25),100000)
ids=[1020801,1020815,1020903]
rows=[]
for r in c.execute('SELECT c.id,c.stackDepth,s.value frame FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN (?,?,?) ORDER BY c.id,c.stackDepth',ids):
 frame=r['frame'] or ''
 if any(x in frame for x in ('dsa_v41','cv_linear','indexer.py','linear.py','model.py')):rows.append({'id':r['id'],'stackDepth':r['stackDepth'],'frame':frame.rsplit('/',1)[-1]})
print(json.dumps({'ids':ids,'frames':rows,'limits':'recorded host callchains; code snapshot/version and tensor storage not verified'},indent=2))
