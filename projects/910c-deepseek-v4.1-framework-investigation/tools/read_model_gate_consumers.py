#!/usr/bin/env python3
"""Follow concrete padding/rotation consumers and generic backend inheritance."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
 prior={}
 for n in ['model-gate-dependencies.json','worker-protocol-dependencies.json','worker-protocol-aux-dependencies.json','input-resource-source-review.json']:
  prior.update({s['runtime_path']:s for s in json.loads((ROOT/n).read_text())['sources']})
 frozen=Path(prior['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3]
 names=['vllm-ascend/vllm_ascend/attention/dsa_v1.py','vllm-ascend/vllm_ascend/ops/fused_moe/router/router_factory.py','vllm-ascend/vllm_ascend/ops/fused_moe/router/fused_topk_router.py','vllm-ascend/vllm_ascend/ops/fused_moe/router/grouped_topk_router.py','vllm/vllm/model_executor/layers/fused_moe/router/fused_moe_router.py','vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py','vllm/vllm/forward_context.py','vllm-ascend/vllm_ascend/models/deepseek_v41/engram/layer.py','vllm-ascend/vllm_ascend/models/deepseek_v41/engram/common.py','vllm-ascend/vllm_ascend/quantization/configs/modelslim_config.py','vllm/vllm/config/parallel.py','vllm/vllm/model_executor/layers/fused_moe/router/base_router.py','vllm-ascend/vllm_ascend/device/hardware_profile.py']
 spec=[]
 for name in names:
  runtime='/vllm-workspace/'+name;p=Path(prior[runtime]['local_path']) if runtime in prior else frozen/name;spec.append(dict(runtime_path=runtime,local_path=str(p) if p.exists() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.exists() else None))
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=150)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout)
 for s in d['sources']:
  body=s.pop('content',None)
  if body is not None:
   p=ROOT/'model-gate-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current disk identities and consumer entry only; no framework/container process, remote mutation or native-bit claim.')
 (ROOT/'model-gate-consumer-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
