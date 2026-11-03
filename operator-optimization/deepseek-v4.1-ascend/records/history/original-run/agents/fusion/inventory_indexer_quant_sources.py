#!/usr/bin/env python3
"""Inventory bounded quantization Python sources in the existing 169 stage container; no NPU use."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOT='/vllm-workspace/vllm-ascend/vllm_ascend/quantization'
code=r'''import hashlib,json
from pathlib import Path
root=Path(%r)
if not root.is_dir(): raise SystemExit('missing root')
need=('AscendW8A8DynamicLinearMethod','W8A8_DYNAMIC','npu_dynamic_quant')
files=sorted(root.rglob('*.py'))
if len(files)>250: raise SystemExit('too many files')
results=[]
for p in files:
 if p.is_symlink() or p.stat().st_size>500000: raise SystemExit('invalid file '+str(p))
 b=p.read_bytes(); s=b.decode('utf-8')
 lines=s.splitlines(); hits=[{'line':i,'text':line.strip()[:240]} for i,line in enumerate(lines,1) if any(n in line for n in need)]
 if hits: results.append({'path':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'hits':hits})
print(json.dumps({'files_scanned':len(files),'matching':results}))
'''%ROOT
def remote(args):
 r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=90)
 if r.returncode: raise RuntimeError('read-only query failed: exit '+str(r.returncode))
 return r.stdout
if remote(['docker','inspect','--format','{{.Image}}',C]).strip()!=IMAGE: raise RuntimeError('unexpected image')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for m in mounts:
 d=m.get('Destination','').rstrip('/') or '/'
 if d=='/' or ROOT==d or ROOT.startswith(d+'/') or d.startswith(ROOT+'/'): raise RuntimeError('quantization subtree overmounted')
x=json.loads(remote(['docker','exec',C,'python3','-c',code]))
x.update({'container':C,'image_id':IMAGE,'source_subtree_overmounted':False,'no_device_work':True})
out=T/'agents/fusion/indexer-quant-source-inventory.json'
if out.is_symlink(): raise RuntimeError('symlink output')
out.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(x,indent=2,ensure_ascii=False))
