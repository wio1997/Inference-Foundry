#!/usr/bin/env python3
"""Read-only bounded inventory of likely QLI native source locations on 169."""
import json, shlex, subprocess
from pathlib import Path

SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER = 'wio-vllm-v41-repro169-stage-20261028'
IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOT = '/vllm-workspace/vllm-ascend/csrc'
OUT = Path(__file__).resolve().parent / 'qli-native-path-inventory.json'
CODE = r'''import json, os
from pathlib import Path
root=Path('/vllm-workspace/vllm-ascend/csrc')
if root.is_symlink() or not root.is_dir(): raise RuntimeError('bad root')
# Examine names only, at most 3 directory levels beneath the fixed csrc root.
stack=[(root,0)]; visited=0; matches=[]; top=[]
while stack:
 p,depth=stack.pop(); visited+=1
 if visited>1200: raise RuntimeError('directory limit')
 entries=sorted(os.scandir(p),key=lambda e:e.name)
 if len(entries)>1000: raise RuntimeError('entry limit')
 for e in entries:
  if e.is_symlink(): continue
  if depth==0: top.append({'name':e.name,'directory':e.is_dir(follow_symlinks=False)})
  if any(x in e.name.lower() for x in ('indexer','lightning')):
   matches.append({'path':e.path,'directory':e.is_dir(follow_symlinks=False)})
  if depth<2 and e.is_dir(follow_symlinks=False): stack.append((Path(e.path),depth+1))
print(json.dumps({'root':str(root),'visited_dirs':visited,'top_level':top,'matches':matches},sort_keys=True))
'''
def remote(args):
 p=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=90)
 if p.returncode: raise RuntimeError('read-only query failed, exit '+str(p.returncode))
 return p.stdout
if remote(['docker','inspect','--format','{{.Image}}',CONTAINER]).strip()!=IMAGE: raise RuntimeError('unexpected image')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',CONTAINER])) or []
for mount in mounts:
 d=mount.get('Destination','').rstrip('/') or '/'
 if d=='/' or ROOT==d or ROOT.startswith(d+'/') or d.startswith(ROOT+'/'): raise RuntimeError('source subtree overmounted')
result=json.loads(remote(['docker','exec',CONTAINER,'python3','-c',CODE]))
result.update({'container':CONTAINER,'image_id':IMAGE,'source_subtree_overmounted':False,'no_device_work':True})
if OUT.is_symlink(): raise RuntimeError('symlink output')
OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
