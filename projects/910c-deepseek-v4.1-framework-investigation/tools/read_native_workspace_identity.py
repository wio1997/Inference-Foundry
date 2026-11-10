#!/usr/bin/env python3
"""Read exact selected Catlass/CANN API sources without framework imports."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from read_prefix_retirement_identity import FROZEN,OUT,REMOTE

NAMES=[
 "/vllm-workspace/vllm-ascend/csrc/aclnn_torch_adapter/op_api_common.h",
 "/vllm-workspace/vllm-ascend/csrc/aclnn_torch_adapter/op_api_common.cpp",
 "/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h",
 "/vllm-workspace/vllm-ascend/csrc/attention/quant_lightning_indexer_v2/quant_lightning_indexer_v2_torch_adpt.h",
 "/vllm-workspace/vllm-ascend/csrc/attention/sparse_flash_mla/sparse_flash_mla_torch_adpt.h",
 "/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/include/torch_npu/csrc/framework/OpCommand.h",
 "/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/include/torch_npu/csrc/framework/utils/OpPreparation.h"
]


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
  rel=x['runtime_path'].lstrip('/');p=OUT/'native-workspace-runtime-source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==x['runtime_sha256'];x['local_path']=str(p)
 d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest();(OUT/'native-workspace-execution-identity.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(sources=len(d['sources']),matched=sum(x.get('matches_local',False) for x in d['sources']),missing=[x['runtime_path'] for x in d['sources'] if x.get('missing')],new_NPU_work=0,remote_writes=[])))

if __name__=='__main__':main()
