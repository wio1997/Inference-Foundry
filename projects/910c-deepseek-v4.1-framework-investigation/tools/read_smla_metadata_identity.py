#!/usr/bin/env python3
"""Read-only exact source identity for SMLA schedule packet producer/consumers."""
from pathlib import Path
from hashlib import sha256
import json
import subprocess
from read_prefix_retirement_identity import OUT,REMOTE

def main():
 spec=[]
 root=OUT/'smla-runtime-source'
 for rp,r in json.loads((root/'metadata-repo/manifest.json').read_text()).items():
  p=root/'metadata-repo'/Path(rp).name;spec.append(dict(runtime_path=rp,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
 manifest=json.loads((root/'manifest.json').read_text())
 ends=['sparse_flash_mla_torch_adpt.h','op_kernel/sparse_flash_mla_kernel_metadata.h','op_kernel/arch22/sparse_flash_mla_arch22_metadata.h','op_kernel/sparse_flash_mla.cpp','op_host/sparse_flash_mla_tiling.cpp']
 for suffix in ends:
  rows=[x for x in manifest['files'] if x['runtime_path'].startswith('/vllm-workspace/') and x['runtime_path'].endswith('/'+suffix)];r,=rows
  p=root/r['local_file'];spec.append(dict(runtime_path=r['runtime_path'],local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
 remote=REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'")
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  content=row.pop('content',None)
  if content is None:continue
  p=OUT/'smla-metadata-runtime-source'/row['runtime_path'].removeprefix('/vllm-workspace/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();d['imported_reader_sha256']=sha256(Path('tools/read_prefix_retirement_identity.py').read_bytes()).hexdigest()
 (OUT/'smla-metadata-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(sources=len(d['sources']),matched=sum(x.get('matches_local',False) for x in d['sources']),missing=[x['runtime_path'] for x in d['sources'] if x.get('missing')],new_NPU_work=0,remote_writes=[])))

if __name__=='__main__':main()
