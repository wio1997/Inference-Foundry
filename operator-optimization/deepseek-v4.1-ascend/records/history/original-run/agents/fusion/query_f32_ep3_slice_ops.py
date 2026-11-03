#!/usr/bin/env python3
"""Only saved host view/slice operations within the fixed L36 input interval."""
import sqlite3,pathlib,json,time
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;start=time.monotonic();c.set_progress_handler(lambda:int(time.monotonic()-start>25),100000)
ns=[r[0] for r in c.execute("SELECT id FROM STRING_IDS WHERE value IN ('aten::slice','aten::as_strided','aten::view','aten::reshape','aten::empty_strided')")];assert ns
q='SELECT startNs,endNs,connectionId,name,inputShapes,inputDtypes FROM PYTORCH_API WHERE globalTid=? AND startNs>=? AND startNs<=? AND name IN ('+','.join('?' for _ in ns)+') ORDER BY startNs LIMIT 200'
rows=[dict(r) for r in c.execute(q,[9139690408016,'1793241329101755880','1793241329115104720']+ns)];assert len(rows)<200
need=sorted({r[k] for r in rows for k in ('name','inputShapes','inputDtypes') if r[k] is not None});s={r[0]:r[1] for r in c.execute('SELECT id,value FROM STRING_IDS WHERE id IN ('+','.join('?' for _ in need)+')',need)}
for r in rows:
 for k in ('name','inputShapes','inputDtypes'):r[k]=s.get(r[k])
print(json.dumps({'scope':'EP3 L36 host thread after gather call through KV matmul call, fixed aten view names only','events':rows,'limits':'input shape only, no output shape or tensor storage identity; selected names/window not exhaustive'},indent=2))
