#!/usr/bin/env python3
"""Read selected current HC source bodies; never load any model/backend."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import REMOTE

ROOT=Path('audit/execution-search')
if __name__=='__main__':
 prior=[]
 for n in ['o-projection-native-dependencies.json','cache-epilogue-entry.json']:
  prior+=json.loads((ROOT/n).read_text()).get('sources',[])
 old=ROOT/'hc-delivery-dependencies.json'
 if old.exists():prior+=json.loads(old.read_text())['sources']
 base='/vllm-workspace/vllm-ascend/'
 names=[base+'vllm_ascend/'+s for s in ['models/deepseek_v41/model.py','device/device_op.py','device/hardware_profile.py']]
 names+=[base+'csrc/'+s for s in ['torch_binding.cpp','torch_binding_meta.cpp','moe/rms_norm_cast/rms_norm_cast_torch_adpt.h','moe/hc_post/op_host/hc_post_tiling.cpp','moe/hc_post/op_host/hc_post_tiling.h','moe/hc_post/op_kernel/hc_post.cpp','moe/hc_post/op_kernel/hc_post_d_split.h','moe/hc_pre/op_host/hc_pre_tiling.cpp','moe/hc_pre/op_host/hc_pre_tiling.h','moe/hc_pre/op_kernel/hc_pre.cpp','moe/hc_pre/op_kernel/hc_pre_base.h','moe/hc_pre/op_kernel/hc_pre_cube_compute.h','moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h','moe/rms_norm_cast/op_host/rms_norm_cast_tiling.cpp','moe/rms_norm_cast/op_kernel/rms_norm_cast.cpp','moe/rms_norm_cast/op_kernel/rms_norm_cast.h','moe/rms_norm_cast/op_kernel/rms_norm_cast_base.h']]
 spec=[]
 for name in names:
  row=next((r for r in reversed(prior) if r['runtime_path']==name),None)
  p=Path(row.get('local_path',row.get('path'))) if row and row.get('local_path',row.get('path')) else None
  spec.append(dict(runtime_path=name,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=110)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=ROOT/'hc-delivery-native-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk source identities only. Review ranges separate, not selected loaded binary/key or bit proof.');old.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(requested=len(spec),sources=len(d['sources']),matches=sum(r.get('matches_local',False) for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')],new_NPU_work=0)))
