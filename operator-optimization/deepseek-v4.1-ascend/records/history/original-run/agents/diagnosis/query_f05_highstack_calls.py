#!/usr/bin/env python3
"""Read-only: inspect existing high-stack EP8 DB for [3,1280] dynamic quant call sites."""
import json,hashlib,sqlite3
from pathlib import Path
root=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029')
s=json.loads((root/'status.json').read_text()); item=next(x for x in s['dbs'] if x['path'].endswith('/ascend_pytorch_profiler_8.db'));p=root/item['path'];h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==item['sha256']=='95b58de883ee272b7161bce7b213d15673c1925fd165bafbf4deacf7a6b74b23'
c=sqlite3.connect(p.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row;c.execute('pragma query_only=on')
rows=[];error=None
try:
 q="""select a.globalTid,a.startNs,a.endNs,a.callchainId,sh.value shape,dt.value dtype from PYTORCH_API a join STRING_IDS n on n.id=a.name left join STRING_IDS sh on sh.id=a.inputShapes left join STRING_IDS dt on dt.id=a.inputDtypes where n.value='npu::npu_dynamic_quant' and sh.value like '3,1280;% '"""
 # Avoid whitespace-sensitive pattern; shape may end in empty optional inputs.
 q=q.replace("like '3,1280;% '","like '3,1280;%'")
 for row in c.execute(q):
  d=dict(row);d['frames']=[x[0] for x in c.execute('select s.value from PYTORCH_CALLCHAINS k join STRING_IDS s on s.id=k.stack where k.id=? order by k.stackDepth',(d['callchainId'],)).fetchall() if any(z in x[0] for z in ('cv_linear.py','dsa_v41.py','quantization','linear.py','indexer'))];rows.append(d)
except sqlite3.Error as e:error=type(e).__name__+': '+str(e)
print(json.dumps({'db_sha256':h.hexdigest(),'pytorch_api_columns':[x[1] for x in c.execute('pragma table_info(PYTORCH_API)')],'matched_calls':rows,'query_error':error,'limits':['high-stack separate capture cannot bind old layer24 run2 tasks','Python frame and shape cannot prove tensor/storage identity','partial DB export; absence and totals not proof of capture completeness','no NPU work']},ensure_ascii=False,indent=2))
