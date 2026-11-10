#!/usr/bin/env python3
"""Pin next latent/index-key RMS/RoPE/quant/scatter producer entry, read-only."""
from pathlib import Path
from hashlib import sha256
import subprocess,json
from read_prefix_retirement_identity import OUT,REMOTE
if __name__=='__main__':
 prior=json.loads((OUT/'index-key-publication-source-review.json').read_text());own=json.loads((OUT/'engram-storage-dependencies.json').read_text());known={r['runtime_path']:r['path'] for r in prior['sources']};known.update({r['runtime_path']:r['local_path'] for r in own['sources']})
 names=['vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py','vllm-ascend/vllm_ascend/attention/dsa_v41.py','vllm-ascend/vllm_ascend/device/device_op.py','vllm/vllm/model_executor/layers/layernorm.py','vllm-ascend/vllm_ascend/ops/layernorm.py','vllm-ascend/vllm_ascend/ops/__init__.py']
 frozen=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot');spec=[]
 for name in names:
  rp='/vllm-workspace/'+name;p=Path(known.get(rp,str(frozen/name)));spec.append(dict(runtime_path=rp,local_path=str(p),local_sha256=sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE.replace("'python3','-I','-S'","'python3','-B','-I','-S'"),json.dumps(spec)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError(r.stderr)
 value=json.loads(r.stdout)
 for row in value['sources']:
  body=row.pop('content',None)
  if body is not None:
   p=OUT/'cache-epilogue-runtime-source'/row['runtime_path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body);assert sha256(p.read_bytes()).hexdigest()==row['runtime_sha256'];row['local_path']=str(p)
 value.update(status='NATIVE_CACHE_EPILOGUE_ENTRY_IDENTITY_ONLY',whole_chain_complete=False,every_replacement_bitwise=True,reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest());(OUT/'cache-epilogue-identity.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(spec),matched=sum(s.get('matches_local',False) for s in value['sources']),missing=[s['runtime_path'] for s in value['sources'] if s.get('missing')])))
