#!/usr/bin/env python3
"""Read current HC coefficient/cross-layer/FULL graph source bodies, stdlib only."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import REMOTE

OUT=Path('audit/execution-search')
def main():
 prior=[]
 for n,key in [('hc-delivery-dependencies.json','sources'),('o-projection-native-dependencies.json','sources'),('common-frame-lifetime-dependencies.json','sources'),('engram-execution-identity.json','source_pins'),('worker-capture-source-review.json','files')]:
  for row in json.loads((OUT/n).read_text()).get(key,[]):
   if 'runtime_path' not in row:row=dict(row,runtime_path=row['path'])
   prior.append(row)
 for row in json.loads(Path('/Users/wio/work/branch-fusion-audit-20261104/full-review/flow-index.json').read_text())['files']:prior.append(row)
 dest=OUT/'hc-coefficient-consumer-dependencies.json'
 if dest.exists():prior+=json.loads(dest.read_text())['sources']
 base='/vllm-workspace/vllm-ascend/'
 names=[base+'vllm_ascend/'+n for n in ['models/deepseek_v41/model.py','models/deepseek_v41/dspark.py','models/deepseek_v41/engram/layer.py','models/deepseek_v41/engram/npu.py','models/deepseek_v41/engram/embedding.py','compilation/acl_graph.py','compilation/updatable_graph.py','ascend_forward_context.py','ascend_config.py','utils.py','worker/model_runner_v1.py','ops/fused_moe/shared_experts.py','ops/fused_moe/fused_moe.py']]
 names += [base+'csrc/moe/hc_pre/op_kernel/'+n for n in ['hc_pre_m_k_split_core.h','hc_pre_base.h']]
 spec=[]
 for name in names:
  row=next((r for r in reversed(prior) if r['runtime_path']==name),None)
  p=Path(row.get('local_path',row.get('path'))) if row and row.get('local_path',row.get('path')) else None
  spec.append(dict(runtime_path=name,local_path=str(p) if p else None,local_sha256=sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=110)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout)
 for row in d['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=OUT/'hc-coefficient-consumer-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 d.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Current source identities; bounded reviewed ranges separate. No active captured binary/key or primitivebit admission.');dest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(requested=len(spec),received=sum(not s.get('missing') for s in d['sources']),previous_matches=sum(s.get('matches_local',False) for s in d['sources']),missing=[s['runtime_path'] for s in d['sources'] if s.get('missing')],new_NPU_work=0)))
if __name__=='__main__':main()
