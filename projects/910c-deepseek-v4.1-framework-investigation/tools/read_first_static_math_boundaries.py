#!/usr/bin/env python3
"""Tighten old first-SMLA/Post prefixes to pre-RoPE Qb/KVnorm endpoints."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,zlib
ROOT=Path('audit/execution-search')
REMOTE=r'''
import concurrent.futures,json,pathlib,socket,sqlite3,sys,zlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
def read(spec):
 p=pathlib.Path(spec['database']);assert p.is_relative_to('/data/tiankuan/wio');before=p.stat();assert [before.st_size,before.st_mtime_ns]==spec['identity']
 db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.execute('PRAGMA query_only=ON');db.execute('PRAGMA temp_store=MEMORY');strings=dict(db.execute('SELECT id,value FROM STRING_IDS'))
 query='SELECT t.rowid,t.startNs,t.endNs,t.streamId,t.modelId,k.opType,k.inputShapes,k.outputShapes FROM TASK t JOIN COMPUTE_TASK_INFO k USING(globalTaskId) WHERE t.rowid BETWEEN ? AND ? AND t.startNs>=? AND t.endNs<=? ORDER BY t.startNs,t.rowid'
 out=[]
 for w in spec['windows']:
  r=dict(w)
  if not w['active']:out.append(r);continue
  try:
   for name in ['target','draft']:
    preid=w['target_first_HcPre'][0] if name=='target' else w['first_HcPre'];start=w['target_first_HcPre'][1] if name=='target' else w['first_HcPre_ns'];end=w[name+'_prefix_end_task']
    rows=[[*x[:5],*[strings[v] for v in x[5:]]] for x in db.execute(query,(min(preid,end[0])-128,max(preid,end[0])+128,start,end[1]))]
    norms=[x for x in rows if x[5]=='RmsNorm' and x[6].endswith(',512;512"')]
    qbs=[x for x in rows if x[5] in ('QuantBatchMatmulV3','MatMulV2') and x[7].endswith(',32768"')]
    assert len(norms)==1 and len(qbs)<=1,(name,'static_endpoint_population',len(norms),len(qbs))
    if end[5]=='SparseFlashMla':assert len(qbs)==1,(name,'query_present_without_Qb')
    stop=max(norms+qbs,key=lambda x:x[2]);assert start<stop[2]<=end[1]
    r[name+'_pre_rope_end_task']=stop;r[name+'_cache_norm_task']=norms[0];r[name+'_Qb_task']=qbs[0] if qbs else None;r[name+'_static_after_HC_ns']=stop[2]-start
   r['math_boundary_qualified']=True
  except Exception as e:r['math_boundary_qualified']=False;r['math_boundary_error']=type(e).__name__+': '+str(e)
  out.append(r)
 db.close();after=p.stat();assert [before.st_size,before.st_mtime_ns]==[after.st_size,after.st_mtime_ns]
 return dict(rank=spec['rank'],database=spec['database'],identity=spec['identity'],windows=out)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:out=list(p.map(read,json.load(sys.stdin)))
sys.stdout.buffer.write(zlib.compress(json.dumps(out,separators=(',',':')).encode()))
'''
def main():
    prior=json.loads((ROOT/'first-static-parent-receipt.json').read_text());p=Path(prior['raw_path']);assert sha256(p.read_bytes()).hexdigest()==prior['raw_sha256']
    source=json.loads(zlib.decompress(p.read_bytes()))
    r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE],input=json.dumps(source).encode(),capture_output=True,timeout=240)
    if r.returncode:raise RuntimeError(r.stderr.decode())
    d=json.loads(zlib.decompress(r.stdout));dest=ROOT/'first-static-math-raw.json.zlib';dest.write_bytes(r.stdout)
    active=[(x['rank'],w) for x in d for w in x['windows'] if w['active']];invalid=[dict(rank=rank,**w) for rank,w in active if not w['math_boundary_qualified']]
    out=dict(status='BOUNDED_ORIGINAL_C8_STATIC_PRE_ROPE_ENDPOINT_REVIEW',raw_path=str(dest),raw_sha256=sha256(r.stdout).hexdigest(),prior_receipt_sha256=sha256((ROOT/'first-static-parent-receipt.json').read_bytes()).hexdigest(),active_windows=len(active),qualified=len(active)-len(invalid),unqualified=invalid,qualification='Same immutable original C8 files and existing prefix anchors. Source first-layer ratio0 Qb/KVnorm endpoints only; emptyquery ends after KVnorm rather than dynamic first HCPost/Oproj wait. No native bit admission or new profiling.',reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],whole_chain_complete=False)
    (ROOT/'first-static-math-receipt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(active=len(active),qualified=out['qualified'],unqualified=len(invalid),examples=invalid[:2],new_NPU_work=0)))
if __name__=='__main__':main()
