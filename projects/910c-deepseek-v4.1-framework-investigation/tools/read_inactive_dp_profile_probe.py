#!/usr/bin/env python3
"""Read immutable completed framework traces; emit bounded event subsets only."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/framework-profile-20261109T192807Z'
OUT = Path('audit/execution-search')
REMOTE = r'''
from collections import Counter
from pathlib import Path
from hashlib import sha256
import json,sys
root=Path(sys.argv[1]);assert root.is_relative_to('/data/tiankuan/wio')
out=[]
for dp in [0,1]:
 paths=list((root/'traces').glob('dp'+str(dp)+'_pp0_tp0_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
 assert len(paths)==1,(dp,paths)
 p=paths[0];stat=p.stat();raw=p.read_bytes();digest=sha256(raw).hexdigest()
 d=json.loads(raw);es=d.get('traceEvents',[]) if isinstance(d,dict) else d
 counts=Counter();selected=[];samples={};native=[];cpu=[]
 for e in es:
  if e.get('ph')!='X':continue
  name=e.get('name','');norm=name.lower().replace('_','');cat=e.get('cat','')
  if name in ['HcPre','HcPost','DispatchFFNCombine']:
   native.append({k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e})
  if any(k in norm for k in ['gloo','hcpre','hcpost','dispatchffncombine','profilerstep','prepare input','sampletoken','npugraph','aclmdl','syncthread']):
   counts[(cat,name)]+=1
   small={k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e}
   samples.setdefault(cat+'|'+name,small)
   if 'gloo' in norm or cat=='python_function' or name in ['prepare input','sample_token'] or 'graph' in norm: selected.append(small)
  if cat=='cpu_op' and any(k in norm for k in ['allreduce','allgather','hcpre','hcpost','dispatchffncombine']):
   cpu.append({k:e[k] for k in ['name','ts','dur','pid','tid','cat','args'] if k in e})
 assert (stat.st_size,stat.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
 out.append(dict(dp=dp,path=str(p),sha256=digest,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
  counts=[dict(category=k[0],name=k[1],n=v) for k,v in counts.items()],samples=samples,
  selected=selected,native=native,cpu=cpu))
print(json.dumps(dict(files=out,new_NPU_work=0,new_profile_collection=0,remote_writes=[]),separators=(',',':')))
'''


def main():
    r = subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,ROOT],
                       capture_output=True,text=True,timeout=55)
    assert r.returncode==0,r.stderr
    d = json.loads(r.stdout)
    d['reader_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'inactive-dp-new-profile-probe.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(dp=x['dp'],bytes=x['bytes'],events=len(x['native']),cpu=len(x['cpu']),
                          counts=x['counts']) for x in d['files']]))


if __name__=='__main__':
    main()
