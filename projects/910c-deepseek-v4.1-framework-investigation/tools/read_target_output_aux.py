#!/usr/bin/env python3
"""Follow stateful processors and the executor's actual async-output reader."""
from pathlib import Path
from hashlib import sha256
import json, subprocess
from read_swa_query_producer_entry import REMOTE

R=Path('audit/execution-search')
NAMES=[
 'vllm/vllm/v1/sample/thinking_budget_state.py',
 'vllm/vllm/v1/sample/logits_processor/state.py',
 'vllm/vllm/v1/executor/multiproc_executor.py',
 'vllm/vllm/v1/executor/uniproc_executor.py',
 'vllm/vllm/v1/executor/abstract.py',
 'vllm/vllm/v1/worker/worker_base.py',
 'vllm-ascend/vllm_ascend/worker/worker.py',
 'vllm/vllm/v1/core/sched/utils.py',
]
def main():
 d=json.loads((R/'target-output-dependencies.json').read_text());old={x['runtime_path']:x for x in d['sources']}
 frozen=Path(old['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py']['local_path']).parents[3];spec=[]
 for n in NAMES:
  p=frozen/n
  if not p.is_file():
   found=list(R.glob('*/vllm-workspace/'+n))
   if found:p=found[0]
  spec.append(dict(runtime_path='/vllm-workspace/'+n,local_path=str(p) if p.is_file() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=150)
 if p.returncode:raise RuntimeError(p.stderr)
 d=json.loads(p.stdout)
 for s in d['sources']:
  body=s.pop('content',None)
  if body is not None:
   p=R/'target-output-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256'];s['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Concrete imported consumer identities; no framework imports or device work. Existing main manifest is not overwritten.',whole_chain_complete=False)
 (R/'target-output-aux-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s['matches_local'] for s in d['sources']),new_bodies=sum(s['local_sha256'] is None for s in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
