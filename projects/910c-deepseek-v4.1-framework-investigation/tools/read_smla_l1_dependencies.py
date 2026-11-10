#!/usr/bin/env python3
"""Pin exact CANN transfer interfaces and selected SMLA consumers without backend imports."""
from pathlib import Path
from hashlib import sha256
import subprocess,json
from read_prefix_retirement_identity import OUT,REMOTE

if __name__=='__main__':
 catalog=json.loads((OUT/'smla-l1-catalog.json').read_text());spec=[]
 for row in catalog['files']:
  rp=row['path'];p=OUT/'smla-l1-runtime-source'/rp.lstrip('/');spec.append(dict(runtime_path=rp,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 previous=json.loads((OUT/'smla-direct-kv-entry.json').read_text())
 for row in previous['sources']:spec.append(dict(runtime_path=row['runtime_path'],local_path=row['path'],local_sha256=row['sha256']))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  c=row.pop('content',None)
  if c is not None:
   p=OUT/'smla-l1-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(c);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();(OUT/'smla-l1-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matched=sum(x.get('matches_local',False) for x in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')],new_NPU_work=0,remote_writes=[])))
