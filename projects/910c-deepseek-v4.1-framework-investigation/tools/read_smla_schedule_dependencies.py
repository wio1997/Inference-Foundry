#!/usr/bin/env python3
"""Pin actual SMLA schedule producers and physical consumers read-only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import OUT,REMOTE,FROZEN

def main():
 root=OUT/'smla-runtime-source';spec=[]
 old=json.loads((root/'manifest.json').read_text())['files']
 extra=[r['path'] for r in json.loads((OUT/'smla-schedule-catalog.json').read_text())['metadata_sources'] if '/op_host/op_api/' in r['path']]
 names=['op_kernel/arch22/sparse_flash_mla_common_arch22.h','op_kernel/arch22/sparse_flash_mla_csa_kernel.h','op_kernel/arch22/sparse_flash_mla_swa_kernel.h','op_kernel/arch22/sparse_flash_mla_csa_block_vector.h','op_kernel/arch22/sparse_flash_mla_swa_block_vector.h','op_kernel/arch22/sparse_flash_mla_csa_block_cube.h','op_kernel/arch22/sparse_flash_mla_swa_block_cube.h','op_kernel/sparse_flash_mla_template_tiling_key.h']
 for name in names:
  r,=[r for r in old if r['runtime_path'].startswith('/vllm-workspace/') and r['runtime_path'].endswith('/'+name)]
  p=root/r['local_file'];spec.append(dict(runtime_path=r['runtime_path'],local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
 for rp in extra:
  p=OUT/'smla-schedule-runtime-source'/rp.removeprefix('/vllm-workspace/');spec.append(dict(runtime_path=rp,local_path=str(p),local_sha256=None))
 rp='/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py';p=FROZEN/'vllm-ascend/vllm_ascend/attention/dsa_v41.py';spec.append(dict(runtime_path=rp,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  content=row.pop('content',None)
  if content is not None:
   p=OUT/'smla-schedule-runtime-source'/row['runtime_path'].removeprefix('/vllm-workspace/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();(OUT/'smla-schedule-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matched=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')],new_NPU_work=0,remote_writes=[])))

if __name__=='__main__':main()
