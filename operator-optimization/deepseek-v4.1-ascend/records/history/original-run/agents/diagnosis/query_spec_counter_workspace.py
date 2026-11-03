#!/usr/bin/env python3
"""Read bounded vLLM source paths for spec metric semantics, no NPU."""
import json, shlex, subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]; SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'; C='wio-vllm-v41-repro169-stage-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
code=r'''import hashlib,json,re
from pathlib import Path
root=Path('/vllm-workspace/vllm/vllm/v1')
folders=['metrics','spec_decode','core','engine','sample']
paths=[]
for folder in folders:
 p=root/folder
 if p.is_dir() and not p.is_symlink():paths.extend(q for q in p.rglob('*.py') if q.is_file() and not q.is_symlink())
if len(paths)>3000:raise SystemExit('file bound')
out={'root':str(root),'scope':folders,'paths_count':len(paths),'files':[]};total=0
for p in sorted(paths):
 if p.stat().st_size>2000000:continue
 b=p.read_bytes();total+=len(b)
 if total>40000000:raise SystemExit('byte bound')
 lines=b.decode('utf8','replace').splitlines()
 hits=[i for i,l in enumerate(lines) if re.search(r'num_accepted_tokens_per_pos|accepted_tokens_per_pos|num_draft_tokens|num_accepted_tokens|num_drafts',l)]
 if hits:
  wanted=set()
  for i in hits:
   for j in range(max(0,i-5),min(len(lines),i+9)):wanted.add(j)
  if len(wanted)>1800:raise SystemExit('output bound')
  out['files'].append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'lines':[{'line':i+1,'text':lines[i][:500]} for i in sorted(wanted)]})
out['bytes']=total
print(json.dumps(out))'''
def remote(args):
 r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError('read-only query error '+r.stderr[-500:])
 return r.stdout
if remote(['docker','inspect','--format','{{.Image}}',C]).strip()!=IMAGE:raise RuntimeError('image differs')
ms=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for m in ms:
 d=m.get('Destination','').rstrip('/') or '/';p='/vllm-workspace'
 if d=='/' or d==p or d.startswith(p+'/') or p.startswith(d+'/'):raise RuntimeError('workspace mount detected')
x=json.loads(remote(['docker','exec',C,'python3','-c',code]));x.update({'container':C,'image':IMAGE,'no_device_work':True})
p=T/'agents/diagnosis/spec-counter-workspace-semantics.json'
if p.exists():raise RuntimeError('output exists')
p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'OK','paths':x['paths_count'],'bytes':x['bytes'],'hits':[v['path'] for v in x['files']]}))
