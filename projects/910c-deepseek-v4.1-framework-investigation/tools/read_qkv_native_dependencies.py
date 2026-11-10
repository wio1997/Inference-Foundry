#!/usr/bin/env python3
"""Export selected installed QuantBatchMatmulV3 sources without executing them."""
from hashlib import sha256
from pathlib import Path
import json,subprocess
from read_prefix_retirement_identity import REMOTE,OUT

if __name__=='__main__':
 root='/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/quant_batch_matmul_v3/'
 names=[root+'quant_batch_matmul_v3'+s for s in ['.cpp','.h','_base.h','_basic_epilogue.h','_bf16.h','_bf16_basic.h','_bf16_opt.h','_block.h','_cube_basic.h','_kernel_tiling_data.h','_pertoken.h','_pertoken_basic.h','_pertoken_opt.h','_tiling_key.h']]
 names+=['/vllm-workspace/vllm-ascend/vllm_ascend/quantization/ascend.py','/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear_op.py']
 spec=[dict(runtime_path=n,local_sha256=None) for n in names]
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=100)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  content=row.pop('content',None)
  if content is not None:
   p=OUT/'qkv-native-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Source identities only; exact read ranges separate. No backend imports or loadedkernel equivalence claim.');(OUT/'qkv-native-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(requested=len(spec),received=sum(not r.get('missing') for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')])))
