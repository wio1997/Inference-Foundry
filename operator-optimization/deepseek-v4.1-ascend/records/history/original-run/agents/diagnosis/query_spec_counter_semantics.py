#!/usr/bin/env python3
"""Read bounded installed vLLM source for speculative counter semantics; no imports/device."""
import json, shlex, subprocess, hashlib
from pathlib import Path
T=Path(__file__).resolve().parents[2]
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
SITE='/usr/local/python3.12.13/lib/python3.12/site-packages'
code=r'''import csv,hashlib,json,re
from pathlib import Path
site=Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
out={'distributions':[],'sources':[],'scope':'static RECORD-indexed vllm python files whose path contains metrics, stats, spec_decode or rejection_sampler'}
files=set()
for dist in sorted(site.glob('vllm-*.dist-info')):
 rec=dist/'RECORD'; meta=dist/'METADATA'
 if dist.is_symlink() or not rec.is_file() or rec.is_symlink() or rec.stat().st_size>30000000: continue
 b=rec.read_bytes(); info={'record_sha256':hashlib.sha256(b).hexdigest(),'dist':dist.name}
 if meta.is_file() and not meta.is_symlink() and meta.stat().st_size<5000000:
  m=meta.read_bytes();info['metadata_sha256']=hashlib.sha256(m).hexdigest(); info['version']=[l for l in m.decode('utf8','replace').splitlines() if l.startswith(('Name:','Version:'))]
 out['distributions'].append(info)
 for row in csv.reader(b.decode('utf8','replace').splitlines()):
  if not row: continue
  p=Path(row[0]); lower=row[0].lower()
  if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='vllm' or p.suffix!='.py':continue
  if any(word in lower for word in ('metric','stats','spec_decode','rejection_sampler')): files.add(row[0])
if len(files)>3000:raise SystemExit('file bound')
total=0
for name in sorted(files):
 p=site/name
 if p.is_symlink() or not p.is_file() or p.stat().st_size>2000000:continue
 data=p.read_bytes();total+=len(data)
 if total>40000000:raise SystemExit('byte bound')
 lines=data.decode('utf8','replace').splitlines()
 hits=[i for i,l in enumerate(lines) if re.search(r'(num_accepted_tokens_per_pos|accepted_tokens_per_pos|num_draft_tokens|num_accepted_tokens|num_drafts)',l)]
 if hits:
  idx=set()
  for i in hits:
   for j in range(max(0,i-6),min(len(lines),i+9)):idx.add(j)
  if len(idx)>1200:raise SystemExit('output bound')
  out['sources'].append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'lines':[{'line':i+1,'text':lines[i][:500]} for i in sorted(idx)]})
out['selected_file_count']=len(files);out['read_bytes']=total
print(json.dumps(out,ensure_ascii=True))
'''
def remote(args):
 p=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=120)
 if p.returncode:raise RuntimeError('fixed read-only query failed: '+p.stderr[-1000:])
 return p.stdout
if remote(['docker','inspect','--format','{{.Image}}',C]).strip()!=IMAGE:raise RuntimeError('unexpected image')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for m in mounts:
 d=m.get('Destination','').rstrip('/') or '/';s=SITE.rstrip('/')
 if d=='/' or s==d or s.startswith(d+'/') or d.startswith(s+'/'):raise RuntimeError('site mount detected')
x=json.loads(remote(['docker','exec',C,'python3','-c',code]));x.update({'container':C,'image':IMAGE,'no_device_work':True,'no_vllm_import':True})
p=T/'agents/diagnosis/spec-counter-installed-semantics.json'
if p.exists():raise RuntimeError('output exists')
p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'OK','distribution':x['distributions'],'file_count':x['selected_file_count'],'bytes':x['read_bytes'],'matches':[s['path'] for s in x['sources']]}))
