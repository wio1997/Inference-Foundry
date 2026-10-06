import sqlite3,json,hashlib,pathlib
inputs=[(13, '/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0249/profiles_167/dp0_pp0_tp13_dcp13_ep13_rank13_300166_20261006111553530_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_13.db', 'ec9448366baebfb3e9ff47dc56f3a0299ac07e92004df739c3d7f395c25757a8'), (15, '/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0249/profiles_167/dp0_pp0_tp15_dcp15_ep15_rank15_300276_20261006111553530_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_15.db', '95b773d4b9dd10bf1d98c264c1e5e5fd7ccef4987856cca33ad4ea5a15cdebe7')]
out=[]
for rank,path,expected in inputs:
 db=pathlib.Path(path)
 actual=hashlib.file_digest(db.open('rb'),'sha256').hexdigest()
 assert actual==expected
 with sqlite3.connect('file:'+path+'?mode=ro',uri=True) as cx:
  py=cx.execute('select globalPid from TASK limit 1').fetchone()[0];tid=(py<<32)|py
  rows=cx.execute('select a.startNs,a.endNs,n.value,a.connectionId,a.globalTid,a.type from CANN_API a join STRING_IDS n on n.id=a.name where a.globalTid=? order by a.startNs',(tid,)).fetchall()
 out.append(dict(rank=rank,db=path,db_sha256=actual,main_tid=tid,columns=['start_ns','end_ns','name','connection','tid','type'],rows=rows))
print(json.dumps(out,separators=(',',':')))
