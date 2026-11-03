#!/usr/bin/env python3
"""Snapshot only the QLI-v2 source subtree from the existing 169 image; no NPU use."""
import base64, hashlib, json, shlex, subprocess
from pathlib import Path
T = Path(__file__).resolve().parents[2]
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C = 'wio-vllm-v41-repro169-stage-20261028'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOT = '/vllm-workspace/vllm-ascend/csrc/attention/quant_lightning_indexer_v2'
DEST = T / 'agents/fusion/frozen-source/csrc/attention/quant_lightning_indexer_v2'
PROV = T / 'agents/fusion/qli-v2-source-provenance.json'
CODE = r'''import base64, hashlib, json, os
from pathlib import Path
root=Path('/vllm-workspace/vllm-ascend/csrc/attention/quant_lightning_indexer_v2')
if root.is_symlink() or not root.is_dir(): raise RuntimeError('bad source root')
allowed={'.h','.hpp','.cpp','.cc','.c','.py','.json','.md','.yaml','.yml','.txt'}
files=[]; dirs=0; total=0
for cur,sub,names in os.walk(root,followlinks=False):
 dirs+=1
 if dirs>128 or len(sub)+len(names)>512: raise RuntimeError('directory bound exceeded')
 sub.sort(); names.sort()
 for name in sub:
  if (Path(cur)/name).is_symlink(): raise RuntimeError('symlink directory refused')
 for name in names:
  p=Path(cur)/name
  if p.is_symlink(): raise RuntimeError('symlink file refused')
  if p.suffix not in allowed: continue
  if not p.is_file(): raise RuntimeError('non-regular source')
  size=p.stat().st_size
  if size>1048576: raise RuntimeError('file too large')
  total+=size
  if total>12582912 or len(files)>=200: raise RuntimeError('source bound exceeded')
  raw=p.read_bytes()
  if len(raw)!=size: raise RuntimeError('source changed')
  files.append({'relative':str(p.relative_to(root)),'size':size,'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode('ascii')})
print(json.dumps({'files':files,'visited_dirs':dirs,'total_bytes':total},sort_keys=True))
'''
def remote(args):
 p=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=90)
 if p.returncode: raise RuntimeError('read-only query failed, exit '+str(p.returncode))
 return p.stdout
if remote(['docker','inspect','--format','{{.Image}}',C]).strip()!=IMAGE: raise RuntimeError('unexpected image')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for mount in mounts:
 d=mount.get('Destination','').rstrip('/') or '/'
 if d=='/' or ROOT==d or ROOT.startswith(d+'/') or d.startswith(ROOT+'/'): raise RuntimeError('source subtree overmounted')
x=json.loads(remote(['docker','exec',C,'python3','-c',CODE]))
if DEST.is_symlink() or PROV.is_symlink(): raise RuntimeError('symlink destination')
records=[]
for item in x.pop('files'):
 rel=Path(item['relative'])
 if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('bad relative path')
 out=DEST / rel
 for ancestor in [out,*out.parents]:
  if ancestor == T.parent: break
  if ancestor.is_symlink(): raise RuntimeError('symlink destination component')
 data=base64.b64decode(item.pop('base64'),validate=True)
 if len(data)!=item['size'] or hashlib.sha256(data).hexdigest()!=item['sha256']: raise RuntimeError('bad content hash')
 out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists() and out.read_bytes()!=data: raise RuntimeError('different existing snapshot')
 out.write_bytes(data)
 records.append(item)
x.update({'files':records,'container':C,'image_id':IMAGE,'root':ROOT,'source_subtree_overmounted':False,'no_device_work':True})
PROV.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(x,indent=2,ensure_ascii=False))
