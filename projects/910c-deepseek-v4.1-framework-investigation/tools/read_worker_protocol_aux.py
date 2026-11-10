#!/usr/bin/env python3
"""Follow concrete imported transfer-state, config and metadata consumer helpers."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
ROOT=Path('audit/execution-search')
def main():
 prior={}
 for n in ['worker-protocol-dependencies.json','target-draft-metadata-dependencies.json','input-resource-source-review.json']:
  prior.update({s['runtime_path']:s for s in json.loads((ROOT/n).read_text())['sources']})
 frozen=Path(prior['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3]
 names=['vllm/vllm/distributed/kv_transfer/kv_transfer_state.py','vllm/vllm/distributed/ec_transfer/ec_transfer_state.py','vllm/vllm/engine/arg_utils.py','vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py','vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py','vllm-ascend/vllm_ascend/attention/utils.py']
 spec=[]
 for name in names:
  runtime='/vllm-workspace/'+name;p=Path(prior[runtime]['local_path']) if runtime in prior else frozen/name;spec.append(dict(runtime_path=runtime,local_path=str(p) if p.exists() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.exists() else None))
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=110)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout)
 for s in d['sources']:
  body=s.pop('content',None)
  if body is not None:
   p=ROOT/'worker-protocol-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,qualification='Source identities only; exact reviewed ranges and legal branch/dataflow constraints are separately recorded. No framework/container process or remote mutation.')
 (ROOT/'worker-protocol-aux-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
