#!/usr/bin/env python3
"""Read actual QLI -> prepared logical IDs -> SMLA consumer source identity."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import OUT,FROZEN,REMOTE

if __name__=='__main__':
 names=['vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py','vllm-ascend/vllm_ascend/ops/triton/prepare_indexer_indices.py','vllm-ascend/vllm_ascend/attention/dsa_v41.py'];old=[]
 for p in OUT.glob('*identity.json'):
  try:old.extend(json.loads(p.read_text()).get('sources',[]))
  except (OSError,ValueError):pass
 spec=[]
 for name in names:
  rp='/vllm-workspace/'+name;found=next((r['local_path'] for r in old if r.get('runtime_path')==rp and r.get('local_path') and Path(r['local_path']).is_file()),None);p=Path(found) if found else FROZEN/name
  spec.append(dict(runtime_path=rp,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  content=row.pop('content',None)
  if content is not None:
   p=OUT/'smla-direct-producer-runtime-source'/row['runtime_path'].removeprefix('/vllm-workspace/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();(OUT/'smla-direct-producer-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matched=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')],new_NPU_work=0,remote_writes=[])))
