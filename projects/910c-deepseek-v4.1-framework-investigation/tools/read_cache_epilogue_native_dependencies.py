#!/usr/bin/env python3
"""Snapshot actual epilogue dispatch and native entry bodies; imports stdlib only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import OUT,REMOTE

if __name__=='__main__':
 repo='/vllm-workspace/vllm-ascend/';cann='/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/'
 names=[repo+'vllm_ascend/ops/register_custom_ops.py','/vllm-workspace/vllm/vllm/model_executor/custom_op.py',repo+'csrc/torch_binding.cpp']
 names += [repo+'csrc/attention/inplace_partial_rotary_mul/'+n for n in ['op_kernel/inplace_partial_rotary_mul.cpp','op_kernel/inplace_partial_rotary_mul.h','op_host/inplace_partial_rotary_mul_a3_tiling.cpp','op_host/inplace_partial_rotary_mul_tiling.cpp']]
 names += [repo+'csrc/moe/scatter_nd_update_sk/'+n for n in ['op_kernel/scatter_nd_update_sk.cpp','op_host/scatter_nd_update_sk_tiling.cpp']]
 names += [cann+n for n in ['rms_norm/rms_norm.cpp','rms_norm/rms_norm.h','rms_norm/rms_norm_base.h','dynamic_quant/dynamic_quant.cpp','dynamic_quant/dynamic_quant.h','dynamic_quant/dynamic_quant_base.h']]
 spec=[dict(runtime_path=n,local_sha256=None) for n in names]
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=OUT/'cache-epilogue-native-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Entry source identities only; individual reviewedranges and actual branch selection separate.');(OUT/'cache-epilogue-native-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(requested=len(spec),received=sum(not r.get('missing') for r in d['sources']),missing=[r['runtime_path'] for r in d['sources'] if r.get('missing')])))
