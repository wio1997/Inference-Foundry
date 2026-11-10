#!/usr/bin/env python3
"""Pin actual shared/routed output, linear-layout and HC consumer sources."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
R=Path('audit/execution-search')
def main():
 old={s['runtime_path']:s for n in ['row-delivery-dependencies.json','model-gate-dependencies.json','model-gate-consumer-dependencies.json','hc-delivery-source-review.json','hc-coefficient-consumer-source-review.json'] for s in json.loads((R/n).read_text())['sources']}
 base=old['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'];frozen=Path(base.get('local_path',base.get('path'))).parents[3]
 names=['vllm-ascend/vllm_ascend/models/deepseek_v41/model.py','vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py','vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py','vllm-ascend/vllm_ascend/ops/fused_moe/shared_experts.py','vllm-ascend/vllm_ascend/ops/fused_moe/dataclass/shared_experts.py','vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py','vllm-ascend/vllm_ascend/ascend_config.py','vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py','vllm/vllm/model_executor/layers/fused_moe/layer.py','vllm/vllm/model_executor/layers/linear.py','vllm-ascend/vllm_ascend/ops/linear.py','vllm-ascend/vllm_ascend/ops/linear_op.py','vllm-ascend/vllm_ascend/patch/platform/patch_fused_moe.py','vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py','vllm-ascend/csrc/moe/hc_post/op_kernel/hc_post.cpp']
 names += ['vllm-ascend/vllm_ascend/utils.py','vllm-ascend/vllm_ascend/ops/fused_moe/router/fused_topk_router.py','vllm-ascend/csrc/moe/hc_post/op_kernel/hc_post_d_split.h']
 spec=[]
 for n in names:
  runtime='/vllm-workspace/'+n;p=Path(old[runtime].get('local_path',old[runtime].get('path'))) if runtime in old else frozen/n
  if not p.is_file():
   matches=list(R.glob('*/vllm-workspace/'+n));p=matches[0] if matches else p
  spec.append(dict(runtime_path=runtime,local_path=str(p) if p.is_file() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=150)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout)
 for s in d['sources']:
  body=s.pop('content',None)
  if body is not None:
   p=R/'moe-output-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current disk identities only, not loaded methods/binary/native bits or full-file review.')
 (R/'moe-output-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
