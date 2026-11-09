#!/usr/bin/env python3
"""Read finished traces/logs in the owned directory; no remote writes/imports."""
import hashlib
import json
from pathlib import Path
import subprocess

from read_inactive_dp_profile_probe import ROOT

OUT = Path('audit/execution-search')
REMOTE = r'''
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json,re,sys
root=Path(sys.argv[1]);assert root.is_relative_to('/data/tiankuan/wio')
out=[]
paths=sorted((root/'traces').glob('dp*_pp0_tp*_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
assert len(paths)==16
for p in paths:
 dp,tp=map(int,re.search(r'dp(\d+)_pp0_tp(\d+)_',str(p)).groups());stat=p.stat()
 raw=p.read_bytes();digest=sha256(raw).hexdigest();d=json.loads(raw)
 es=d.get('traceEvents',[]) if isinstance(d,dict) else d
 selected=[];native=[];counts=Counter();meta=[]
 for e in es:
  name=e.get('name','');cat=e.get('cat','')
  if e.get('ph')=='M' and name in ['process_name','process_labels']:meta.append(e)
  if e.get('ph')!='X':continue
  if name in ['HcPre','HcPost','DispatchFFNCombine']:
   native.append({k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e})
  if ((cat=='cpu_op' and name in ['gloo:all_reduce','c10d::allreduce_',
      '_C_ascend::npu_hc_pre_v3','_C_ascend::npu_hc_post','_C_ascend::dispatch_ffn_combine'])
      or cat=='python_function' or name=='AscendCL@aclmdlRIExecuteAsync'):
   selected.append({k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e});counts[(cat,name)]+=1
 assert (stat.st_size,stat.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
 out.append(dict(rank=8*dp+tp,dp=dp,tp=tp,path=str(p),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
  sha256=digest,selected=selected,native=native,metadata=meta,
  counts=[dict(category=k[0],name=k[1],n=v) for k,v in counts.items()]))
log=root/'service.log';st=log.stat();raw=log.read_bytes()
lines=[s for s in raw.decode(errors='replace').splitlines() if any(x in s for x in [
 'Max profiling iterations reached','Profiler stopped successfully','GPU profiling will stop','start_profile','stop_profile'])]
assert (st.st_size,st.st_mtime_ns)==(log.stat().st_size,log.stat().st_mtime_ns)
print(json.dumps(dict(files=out,profile_log=dict(path=str(log),sha256=sha256(raw).hexdigest(),bytes=st.st_size,
 lines=lines),new_NPU_work=0,new_profile_collection=0,remote_writes=[]),separators=(',',':')))
'''


def main():
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,ROOT],
                       capture_output=True,text=True,timeout=115)
    assert r.returncode==0,r.stderr
    d=json.loads(r.stdout)
    raw=json.dumps(d,ensure_ascii=False,separators=(',',':')).encode()
    import zlib
    path=OUT/'inactive-dp-framework-edges.json.zlib';path.write_bytes(zlib.compress(raw,9))
    receipt=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_bytes=len(raw),
                 reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 files=[{k:r[k] for k in ['rank','dp','tp','path','bytes','mtime_ns','sha256','counts']} for r in d['files']],
                 profile_log=d['profile_log'],new_NPU_work=0,new_profile_collection=0,remote_writes=[])
    (OUT/'inactive-dp-framework-edges-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=len(d['files']),raw_bytes=len(raw),compressed_bytes=path.stat().st_size,
                         profile_log_lines=d['profile_log']['lines'])))


if __name__=='__main__':
    main()
