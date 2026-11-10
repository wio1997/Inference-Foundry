#!/usr/bin/env python3
"""Pin next Engram code/scale loading, registered owner and lookup entry read-only."""
from pathlib import Path
from hashlib import sha256
import subprocess,json
from read_prefix_retirement_identity import OUT,REMOTE
if __name__=='__main__':
 old=json.loads((OUT/'engram-execution-identity.json').read_text());spec=[]
 for row in old['source_pins']:
  if any(row['path'].endswith(x) for x in ['/engram/embedding.py','/engram/npu.py','/ops/triton/engram_lookup.py']):
   p=Path(row['local_path']);spec.append(dict(runtime_path=row['path'],local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
 assert len(spec)==3
 run=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if run.returncode:raise RuntimeError(run.stderr)
 value=json.loads(run.stdout)
 for row in value['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=OUT/'engram-storage-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 value.update(status='ENTRY_IDENTITY_ONLY_NOT_COMPLETED_STORAGE_OR_NATIVE_REVIEW',whole_chain_complete=False,every_replacement_bitwise=True,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),new_NPU_work=0,remote_writes=[])
 p=OUT/'engram-storage-identity.json';p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(spec),matched=sum(r.get('matches_local',False) for r in value['sources']))))
