#!/usr/bin/env python3
"""Read concrete OOT operator registration without extending old manifests."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
R=Path('audit/execution-search')
def main():
 pins=json.loads((R/'aux-context-dependencies.json').read_text())['sources'];old=next(x for x in pins if x['runtime_path'].endswith('/worker/model_runner_v1.py'));frozen=Path(old['local_path']).parents[3]
 name='vllm-ascend/vllm_ascend/ops/register_custom_ops.py';p=frozen/name
 spec=[dict(runtime_path='/vllm-workspace/'+name,local_path=str(p) if p.is_file() else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None)]
 result=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=60)
 if result.returncode:raise RuntimeError(result.stderr)
 d=json.loads(result.stdout)
 for s in d['sources']:
  content=s.pop('content',None)
  if content is not None:
   p=R/'aux-context-source'/s['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);s['local_path']=str(p);assert sha256(p.read_bytes()).hexdigest()==s['runtime_sha256']
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Concrete import from ops/__init__; separate manifest, no old source scope overwritten or loaded-object claim.',whole_chain_complete=False)
 (R/'aux-context-registry-dependencies.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=1,matches=sum(x['matches_local'] for x in d['sources']),remote_writes=[])))
if __name__=='__main__':main()
