#!/usr/bin/env python3
"""Read exact selected Catlass/CANN API sources without framework imports."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_prefix_retirement_identity import FROZEN,OUT,REMOTE

PREFIX='/vllm-workspace/vllm-ascend/csrc/third_party/catlass/include/catlass/'
NAMES=[PREFIX+x for x in ['gemm/tile/atlasa2/copy_l0c_to_gm.hpp','gemm/tile/tile_copy.hpp','epilogue/tile/tile_copy.hpp','gemm/block/block_mmad.hpp','gemm/block/block_swizzle.hpp','gemm/dispatch_policy.hpp','gemm/tile/tile_mmad.hpp','gemm/tile/copy_l0c_to_ub.hpp','gemm/tile/copy_l1_to_fp.hpp','gemm/helper.hpp','layout/matrix.hpp','arch/arch.hpp','arch/atlas_a2.hpp']]
CANN='/usr/local/Ascend/cann-9.2.0/aarch64-linux/asc/'
NAMES += [CANN+x for x in ['impl/basic_api/dav_c220/kernel_operator_fixpipe_v2_impl.h','include/interface/kernel_operator_fixpipe_intf.h','impl/basic_api/kernel_operator_fixpipe_intf_impl.h','include/interface/kernel_struct_fixpipe.h','impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h','impl/basic_api/dav_c220/kernel_operator_scm_data_copy_impl.h','include/interface/kernel_operator_data_copy_intf.h']]
NAMES += ['/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine.cpp','/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/CMakeLists.txt']


def main():
 old=[]
 for p in OUT.glob('*identity.json'):
  try:old.extend(json.loads(p.read_text()).get('sources',[]))
  except (OSError,ValueError):pass
 spec=[]
 for rp in NAMES:
  found=next((x['local_path'] for x in old if x.get('runtime_path')==rp and x.get('local_path') and Path(x['local_path']).is_file()),None)
  p=Path(found) if found else FROZEN/rp.removeprefix('/vllm-workspace/')
  spec.append(dict(runtime_path=rp,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 remote=REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'")
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for x in d['sources']:
  content=x.pop('content',None)
  if content is None:continue
  rel=x['runtime_path'].lstrip('/');p=OUT/'dfc-gemm-extra-runtime-source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==x['runtime_sha256'];x['local_path']=str(p)
 d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();(OUT/'dfc-gemm-extra-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(sources=len(d['sources']),matched=sum(x.get('matches_local',False) for x in d['sources']),missing=[x['runtime_path'] for x in d['sources'] if x.get('missing')],new_NPU_work=0,remote_writes=[])))

if __name__=='__main__':main()
