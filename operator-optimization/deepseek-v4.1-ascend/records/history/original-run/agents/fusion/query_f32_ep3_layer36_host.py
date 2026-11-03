#!/usr/bin/env python3
"""Read-only bounded host thread window around an already bound EP3 L36 MC2."""
import sqlite3,pathlib,json,time
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
start=time.monotonic();c.set_progress_handler(lambda: int(time.monotonic()-start>25),100000)
tid=9139690408016; end=1793241329159326070; begin=end-100_000_000
ids=[dict(r) for r in c.execute("SELECT id,value FROM STRING_IDS WHERE value IN ('HcclAllGather','aclnnMatmul','aclnnMatMul')")]
if not ids: raise RuntimeError('expected API names absent')
params=[r['id'] for r in ids]; mark=','.join('?' for _ in params)
q='SELECT startNs,endNs,globalTid,connectionId,callchainId,name FROM PYTORCH_API WHERE globalTid=? AND startNs>=? AND startNs<=? AND name IN ('+mark+') ORDER BY startNs DESC LIMIT 25'
rows=[dict(r) for r in c.execute(q,[tid,str(begin),str(end)]+params)];name={r['id']:r['value'] for r in ids}
chains=sorted({r['callchainId'] for r in rows}); frames={str(k):[] for k in chains}
if chains:
 q='SELECT c.id,c.stackDepth,s.value FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE c.id IN ('+','.join('?' for _ in chains)+') AND (s.value LIKE ? OR s.value LIKE ?) ORDER BY c.id,c.stackDepth'
 for r in c.execute(q,chains+['%deepseek_v41/model.py%','%sequence_parallel.py%']):frames[str(r[0])].append({'depth':r[1],'frame':r[2].rsplit('/',1)[-1]})
for r in rows:r['name']=name[r['name']];r['frames']=frames[str(r['callchainId'])]
print(json.dumps({'scope':'EP3 saved diagnostic host thread, previous 100ms of bound MC2 host start, exact two API names, last 25 events','anchor_connection_id':431408,'events':rows,'limits':'Only selected host thread/time/API, no tensor identity, event scope may include different layer/request; callstack identity and device flow not yet established'},indent=2))
