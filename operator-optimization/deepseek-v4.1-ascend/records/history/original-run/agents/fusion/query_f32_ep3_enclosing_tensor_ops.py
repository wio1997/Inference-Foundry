#!/usr/bin/env python3
"""Enclosing input metadata for three fixed host calls, one saved DB scan."""
import sqlite3,pathlib,json,time
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;start=time.monotonic();c.set_progress_handler(lambda:int(time.monotonic()-start>25),100000)
tid=9139690408016; targets={'gather':('1793241329101711120','1793241329101755880'),'qa':('1793241329109190480','1793241329109280570'),'kv':('1793241329114989800','1793241329115104720')}
where=' OR '.join('(startNs<=? AND endNs>=?)' for x in targets);args=[v for pair in targets.values() for v in pair]
rows=[dict(r) for r in c.execute('SELECT startNs,endNs,connectionId,name,inputShapes,inputDtypes FROM PYTORCH_API WHERE globalTid=? AND ('+where+') LIMIT 1000',[tid]+args)];assert len(rows)<1000
need=sorted({r[k] for r in rows for k in ('name','inputShapes','inputDtypes') if r[k] is not None});s={}
if need:s={r[0]:r[1] for r in c.execute('SELECT id,value FROM STRING_IDS WHERE id IN ('+','.join('?' for _ in need)+')',need)}
out={x:[] for x in targets}
for r in rows:
 name=s.get(r['name'],'')
 if not (name.startswith('aten::') or name.startswith('c10d::') or 'AllGather' in name):continue
 for k,(a,b) in targets.items():
  if r['startNs']<=a and r['endNs']>=b:
   out[k].append({'name':name,'startNs':r['startNs'],'endNs':r['endNs'],'inputShapes':s.get(r['inputShapes']),'inputDtypes':s.get(r['inputDtypes'])})
print(json.dumps({'scope':'one EP3 diagnostic host thread, only ranges enclosing 431371,431372,431373','matches':out,'limitations':'Host scope containment is not tensor identity. No pointers or storage IDs in PYTORCH_API; shapes and inputs may be unavailable.'},indent=2))
