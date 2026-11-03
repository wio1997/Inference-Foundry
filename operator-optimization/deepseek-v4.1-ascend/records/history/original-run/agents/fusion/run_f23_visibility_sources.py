#!/usr/bin/env python3
"""Read three exact container source files; never touch NPU or modify remote state."""
import base64, hashlib, json, shlex, subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOT='/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/utils/'
NAMES=['hccl_shmem.hpp','block_epilogue_pertoken_row.hpp','block_epilogue_pertoken_v2.hpp']
EXPECTED={'hccl_shmem.hpp':('596bc8a0c28748aade2e5f2e978f18a6e593d146a8cf07fe4aeac8d36c19bad6',12006)}
MAX=512000
code=r'''import base64,hashlib,json
from pathlib import Path
root=Path(%r)
names=%r
out=[]
for n in names:
 p=root/n
 if p.is_symlink() or not p.is_file(): raise SystemExit('not regular: '+str(p))
 if p.stat().st_size > %d: raise SystemExit('oversize: '+str(p))
 b=p.read_bytes()
 if len(b)>%d: raise SystemExit('oversize after read')
 out.append({'name':n,'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode('ascii')})
print(json.dumps(out))
'''%(ROOT,NAMES,MAX,MAX)
def remote(args):
 r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=120)
 if r.returncode: raise RuntimeError('read-only source query failed, exit '+str(r.returncode))
 return r.stdout
if remote(['docker','inspect','--format','{{.Image}}',C]).strip()!=IMAGE: raise RuntimeError('unexpected image')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for m in mounts:
 dest=m.get('Destination','').rstrip('/') or '/'
 if dest=='/' or ROOT.startswith(dest+'/') or dest==ROOT.rstrip('/'): raise RuntimeError('source overmounted')
 for n in NAMES:
  p=ROOT+n
  if p==dest or p.startswith(dest+'/'): raise RuntimeError('file overmounted')
x=json.loads(remote(['docker','exec',C,'python3','-c',code]))
if [v['name'] for v in x]!=NAMES: raise RuntimeError('unexpected file list')
records=[]
for v in x:
 b=base64.b64decode(v['data'],validate=True); h=hashlib.sha256(b).hexdigest()
 if len(b)!=v['size'] or len(b)>MAX or h!=v['sha256']: raise RuntimeError('source integrity')
 if v['name'] in EXPECTED and (h,len(b))!=EXPECTED[v['name']]: raise RuntimeError('prior source hash or size changed')
 out=T/'agents/fusion/frozen-source/csrc/mc2/dispatch_ffn_combine/op_kernel/utils'/v['name']
 if out.is_symlink(): raise RuntimeError('output is symlink')
 if out.exists() and out.read_bytes()!=b: raise RuntimeError('existing source differs')
 if not out.exists():
  out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('xb') as f:f.write(b)
 records.append({'source':ROOT+v['name'],'snapshot':str(out.relative_to(T)),'size':len(b),'sha256':h})
print(json.dumps({'status':'SOURCE_SNAPSHOTS_VERIFIED','container':C,'image_id':IMAGE,'source_subtree_overmounted':False,'files':records,'no_device_work':True},indent=2))
