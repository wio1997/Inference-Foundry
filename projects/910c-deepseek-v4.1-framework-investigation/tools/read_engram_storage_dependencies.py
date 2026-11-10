#!/usr/bin/env python3
"""Read current owner/load/capture dependencies without framework imports."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
from read_prefix_retirement_identity import REMOTE,OUT

if __name__=='__main__':
 old=json.loads((OUT/'engram-execution-identity.json').read_text());entry=json.loads((OUT/'engram-storage-entry.json').read_text());known={r['path']:r['local_path'] for r in old['source_pins']}
 names=['vllm-ascend/vllm_ascend/'+s for s in ['device/device_op.py','device/hardware_profile.py','models/deepseek_v41/model.py','models/deepseek_v41/engram/model_state.py','models/deepseek_v41/engram/layer.py','worker/model_runner_v1.py','worker/worker.py']]+['vllm/vllm/model_executor/model_loader/default_loader.py']
 spec=[dict(runtime_path=r['runtime_path'],local_path=r['path'],local_sha256=r['sha256']) for r in entry['sources']]
 frozen=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot')
 for name in names:
  runtime='/vllm-workspace/'+name;p=Path(known.get(runtime,str(frozen/name)));spec.append(dict(runtime_path=runtime,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 value=json.loads(r.stdout)
 for row in value['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=OUT/'engram-storage-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 value.update(reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope='Only source identity; each read range recorded separately. No Torch/NPU/model imports or writes.');(OUT/'engram-storage-dependencies.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(spec),matched=sum(r.get('matches_local',False) for r in value['sources']),missing=[r['runtime_path'] for r in value['sources'] if r.get('missing')])))
