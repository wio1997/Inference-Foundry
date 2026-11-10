#!/usr/bin/env python3
"""Current original WKV/row-owner source identities, no framework import."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_swa_query_producer_entry import REMOTE
OUT=Path('audit/execution-search')
def main():
 prior=[]
 for n in ['engram-joint-entry.json','o-projection-native-dependencies.json','hc-coefficient-consumer-dependencies.json']:
  prior+=json.loads((OUT/n).read_text())['sources']
 dest=OUT/'engram-row-owner-dependencies.json'
 if dest.exists():prior+=json.loads(dest.read_text())['sources']
 base='/vllm-workspace/vllm-ascend/vllm_ascend/'
 names=[base+n for n in ['models/deepseek_v41/model.py','models/deepseek_v41/engram/layer.py','models/deepseek_v41/engram/embedding.py','models/deepseek_v41/engram/parallel.py','ops/linear.py','ops/linear_op.py','quantization/configs/modelslim_config.py']]
 names+=['/vllm-workspace/vllm/vllm/model_executor/layers/linear.py','/vllm-workspace/vllm/vllm/distributed/parallel_state.py']
 spec=[]
 for name in names:
  r=next((r for r in reversed(prior) if r['runtime_path']==name),None);p=Path(r.get('local_path',r.get('path'))) if r and r.get('local_path',r.get('path')) else None
  spec.append(dict(runtime_path=name,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=110)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=OUT/'engram-row-owner-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current disk source identity; exactreview ranges separate. Native activekey/order/loadedphysicalformat not proved.');dest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(d['sources']),matches=sum(s.get('matches_local',False) for s in d['sources']),missing=[s['runtime_path'] for s in d['sources'] if s.get('missing')],new_NPU_work=0)))
if __name__=='__main__':main()
