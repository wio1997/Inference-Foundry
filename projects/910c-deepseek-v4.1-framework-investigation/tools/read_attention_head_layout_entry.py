#!/usr/bin/env python3
"""Pin real SMLA head/query guards and CP layout readers for next bounded scan."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import OUT,REMOTE

if __name__=='__main__':
 prior=[]
 for name,key in [('smla-schedule-source-review.json','source_pins'),('smla-direct-l1-source-review.json','sources'),('qkv-route-identity.json','sources')]:prior.extend(json.loads((OUT/name).read_text())[key])
 suffixes=['/op_host/sparse_flash_mla_tiling.cpp','/op_host/sparse_flash_mla_metadata_check.h','/arch22/sparse_flash_mla_common_arch22.h','/arch22/sparse_flash_mla_csa_kernel.h','/arch22/sparse_flash_mla_csa_block_vector.h','/arch22/sparse_flash_mla_csa_block_cube.h','/context_parallel/dsa_v41_cp.py','/context_parallel/dsa_cp.py']
 spec=[]
 for suffix in suffixes:
  r=next(r for r in prior if r['runtime_path'].endswith(suffix));p=Path(r.get('local_path',r.get('path')));spec.append(dict(runtime_path=r['runtime_path'],local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest()))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=100)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  c=row.pop('content',None)
  if c is not None:
   p=OUT/'attention-head-layout-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(c);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(status='HEAD_LAYOUT_ACTUAL_SMLA_QUERY_CONSUMER_IDENTITY_ENTRY_ONLY',whole_chain_complete=False,every_replacement_bitwise=True,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),reviewed_ranges=[],scope='Identity/retrieval only at acquisition; read guards/ranges recorded separately. Does not qualify arbitrary head/layout changes or loadedbinary equivalence.');(OUT/'attention-head-layout-entry.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')])))
