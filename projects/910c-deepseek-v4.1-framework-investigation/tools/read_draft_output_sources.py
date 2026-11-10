#!/usr/bin/env python3
"""Follow concrete DSpark output imports without importing frameworks."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
R=Path('audit/execution-search')
def main():
 old={s['runtime_path']:s for n in ['row-delivery-dependencies.json','moe-output-dependencies.json','model-gate-dependencies.json','model-gate-consumer-dependencies.json'] for s in json.loads((R/n).read_text())['sources']}
 frozen=Path(old['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3]
 names=['vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py','vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py','vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py','vllm-ascend/vllm_ascend/spec_decode/utils.py','vllm/vllm/v1/spec_decode/llm_base_proposer.py','vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py','vllm/vllm/model_executor/models/qwen3_dspark.py','vllm/vllm/model_executor/layers/logits_processor.py','vllm/vllm/model_executor/layers/vocab_parallel_embedding.py','vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py','vllm-ascend/vllm_ascend/worker/model_runner_v1.py','vllm/vllm/v1/worker/gpu_model_runner.py','vllm/vllm/v1/sample/rejection_sampler.py','vllm-ascend/vllm_ascend/sample/rejection_sampler.py','vllm-ascend/vllm_ascend/sample/sampler.py','vllm-ascend/vllm_ascend/patch/worker/patch_rejection_sampler.py','vllm/vllm/v1/sample/sampler.py','vllm/vllm/config/speculative.py','vllm-ascend/vllm_ascend/ascend_config.py','vllm-ascend/vllm_ascend/utils.py']
 names += ['vllm-ascend/vllm_ascend/ops/triton/reject_sample.py']
 specs=[]
 for n in names:
  runtime='/vllm-workspace/'+n;p=Path(old[runtime]['local_path']) if runtime in old else frozen/n
  if not p.is_file():
   matches=list(R.glob('*/vllm-workspace/'+n))+list(R.glob('*/'+n));p=matches[0] if matches else p
  specs.append(dict(runtime_path=runtime,local_path=str(p) if p.is_file() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(specs)],capture_output=True,text=True,timeout=150)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout)
 for s in d['sources']:
  body=s.pop('content',None)
  if body is not None:
   p=R/'draft-output-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current disk identities only, not loaded code, complete reviews, population or native bits.')
 (R/'draft-output-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
