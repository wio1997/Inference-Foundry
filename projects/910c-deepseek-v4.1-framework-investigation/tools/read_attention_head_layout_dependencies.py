#!/usr/bin/env python3
"""Recheck current head/query consumers and follow native metadata producers read-only."""
from hashlib import sha256
from pathlib import Path
import json,subprocess
from read_prefix_retirement_identity import OUT,REMOTE

if __name__=='__main__':
 prior=[]
 for name,key in [('attention-head-layout-entry.json','sources'),('smla-schedule-source-review.json','source_pins'),('cache-epilogue-entry.json','sources'),('smla-direct-l1-source-review.json','sources')]:prior.extend(json.loads((OUT/name).read_text())[key])
 names=[r['runtime_path'] for r in json.loads((OUT/'attention-head-layout-entry.json').read_text())['sources']]
 names += ['/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp','/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.h','/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_kernel_metadata.h','/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py','/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py']
 base='/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla/'
 names += [base+'op_kernel/'+n for n in ['sparse_flash_mla.cpp','arch22/sparse_flash_mla_swa_kernel.h','arch22/sparse_flash_mla_swa_block_cube.h','arch22/sparse_flash_mla_swa_block_vector.h']]
 spec=[]
 for name in names:
  old=next((r for r in prior if r['runtime_path']==name),None)
  p=Path(old.get('local_path',old.get('path'))) if old else OUT/'smla-runtime-source/repo'/name.removeprefix(base)
  spec.append(dict(runtime_path=name,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=100)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  c=row.pop('content',None)
  if c is not None:
   p=OUT/'attention-head-layout-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(c);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk sources only. Listed new review ranges separate; no live object/tiling or numerical proof.');(OUT/'attention-head-layout-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matched=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')])))
