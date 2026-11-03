#!/usr/bin/env python3
"""Read bounded host ops around two known EP3 L36 projection calls; no values."""
import hashlib,json,pathlib,sqlite3
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()=='084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4'
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
names=['aten::reshape','aten::view','aten::linear','aten::matmul','aten::clone','aten::copy_','aten::contiguous','aten::_reshape_alias','npu::npu_transpose_batchmatmul','aclnnTransposeBatchMatMul','aclnnMatmul']
params=','.join('?' for _ in names)
sids=[r[0] for r in c.execute('SELECT id FROM STRING_IDS WHERE value IN ('+params+')',names)]
q='SELECT startNs,endNs,globalTid,connectionId,callchainId,name,inputShapes,inputDtypes FROM PYTORCH_API WHERE globalTid=? AND startNs>=? AND startNs<=? AND name IN ('+','.join('?' for _ in sids)+') ORDER BY startNs LIMIT 201'
rows=[dict(r) for r in c.execute(q,[9139690408016,'1793241329147000000','1793241329150500000']+sids)];assert len(rows)<=200
need={v for x in rows for k,v in x.items() if k in ('name','inputShapes','inputDtypes') and v is not None}
vals={r[0]:r[1] for r in c.execute('SELECT id,value FROM STRING_IDS WHERE id IN ('+','.join('?' for _ in need)+')',list(need))} if need else {}
for x in rows:
 for k in ('name','inputShapes','inputDtypes'):x[k]=vals.get(x[k])
 frames=c.execute('SELECT p.stackDepth,s.value FROM PYTORCH_CALLCHAINS p JOIN STRING_IDS s ON s.id=p.stack WHERE p.id=? ORDER BY p.stackDepth',(x['callchainId'],)).fetchall()
 x['frames']=[{'depth':z[0],'frame':str(z[1]).rsplit('/',1)[-1]} for z in frames if any(v in str(z[1]) for v in ('dsa_v41','dsa_cp.py','model.py','linear.py'))]
print(json.dumps({'db_sha256':h.hexdigest(),'scope':'EP3 single host thread bounded 3.5ms window, selected aten/npu API names only','results':rows,'limits':['Input metadata only; no pointer, storage, output shape or guaranteed source instance.','Absence from selected API names does not prove no copy or opaque kernel behavior.']},indent=2))
c.close()
