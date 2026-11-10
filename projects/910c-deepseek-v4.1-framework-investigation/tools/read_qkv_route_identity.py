#!/usr/bin/env python3
"""Recheck Python producer/consumer identities and follow the quant scheme adapter."""
from hashlib import sha256
from pathlib import Path
import json,subprocess
from read_prefix_retirement_identity import OUT,REMOTE

if __name__=='__main__':
 old=json.loads((OUT/'swa-query-producer-entry.json').read_text())['sources'];spec=[]
 for r in old:
  p=Path(r['local_path']);spec.append(dict(runtime_path=r['runtime_path'],local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
 spec.append(dict(runtime_path='/vllm-workspace/vllm-ascend/vllm_ascend/quantization/method_adapters.py',local_sha256=None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=100)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  c=row.pop('content',None)
  if c is not None:
   p=OUT/'qkv-route-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(c);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current running container disk source identity only; no model import/live loadedkernel proof. Previous stopped-container entry remains immutable.');(OUT/'qkv-route-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matched=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')])))
