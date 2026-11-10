#!/usr/bin/env python3
"""Read source for the five model guards and the actual generic selector path."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
 prior={}
 for n in ['worker-protocol-dependencies.json','worker-protocol-aux-dependencies.json','target-draft-graph-dependencies.json','target-draft-metadata-dependencies.json']:
  prior.update({s['runtime_path']:s for s in json.loads((ROOT/n).read_text())['sources']})
 frozen=Path(prior['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3]
 names=['vllm-ascend/vllm_ascend/models/deepseek_v41/model.py','vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py','vllm-ascend/vllm_ascend/attention/dsa_v41.py','vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py','vllm-ascend/vllm_ascend/models/common/ops/sequence_parallel.py','vllm/vllm/envs.py','vllm/vllm/v1/attention/selector.py','vllm-ascend/vllm_ascend/platform.py','vllm-ascend/vllm_ascend/utils.py','vllm-ascend/vllm_ascend/attention/mla_v1.py','vllm-ascend/vllm_ascend/attention/attention_v1.py','vllm-ascend/vllm_ascend/compilation/acl_graph.py','vllm-ascend/vllm_ascend/patch/platform/patch_fused_moe.py','vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py']
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
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Current disk source identities only; reviewed exact ranges separate. No framework/container process, remote writes, native bit or loaded-object claim.')
 (ROOT/'model-gate-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
