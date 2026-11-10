#!/usr/bin/env python3
"""Pin actual aux-to-context receivers and arithmetic/communication dispatch."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
R=Path('audit/execution-search')
NAMES=[
 'vllm-ascend/vllm_ascend/models/deepseek_v41/vl_model.py',
 'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',
 'vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py',
 'vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py',
 'vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py',
 'vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py',
 'vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
 'vllm-ascend/vllm_ascend/models/common/ops/sequence_parallel.py',
 'vllm-ascend/vllm_ascend/ops/dsa.py',
 'vllm-ascend/vllm_ascend/ops/linear.py',
 'vllm-ascend/vllm_ascend/ops/linear_op.py',
 'vllm/vllm/model_executor/layers/linear.py',
 'vllm/vllm/model_executor/layers/layernorm.py',
 'vllm-ascend/vllm_ascend/ops/layernorm.py',
 'vllm-ascend/vllm_ascend/ops/rope_dsv4.py',
 'vllm-ascend/vllm_ascend/device/device_op.py',
 'vllm-ascend/vllm_ascend/device/hardware_profile.py',
 'vllm-ascend/vllm_ascend/ops/__init__.py',
]
def main():
 old={}
 for name in ['whole-forward-dependencies.json','whole-forward-binding-dependencies.json','draft-output-dependencies.json','moe-output-dependencies.json','index-key-publication-dependencies.json']:
  old.update({x['runtime_path']:x for x in json.loads((R/name).read_text())['sources']})
 frozen=Path(old['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3];spec=[]
 for name in NAMES:
  runtime='/vllm-workspace/'+name;p=Path(old[runtime]['local_path']) if runtime in old else frozen/name
  if not p.is_file():
   found=list(R.glob('*/vllm-workspace/'+name))+list(R.glob('*/'+name))
   if found:p=found[0]
  spec.append(dict(runtime_path=runtime,local_path=str(p) if p.is_file() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 result=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=150)
 if result.returncode:raise RuntimeError(result.stderr)
 d=json.loads(result.stdout)
 for s in d['sources']:
  body=s.pop('content',None)
  if body is not None:
   p=R/'aux-context-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),reviewed_ranges=[],whole_chain_complete=False,qualification='Fresh disk identities only; no framework/device imports or processes in container. Concrete receiver and consumer source walks recorded separately.')
 (R/'aux-context-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(source_pins=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
