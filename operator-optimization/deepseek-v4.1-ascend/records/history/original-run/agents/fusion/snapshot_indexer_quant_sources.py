#!/usr/bin/env python3
"""Hash-bind six exact Python files from verified 169 stage inventory, without device access."""
import base64,hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOT='/vllm-workspace/vllm-ascend/vllm_ascend/quantization/'
FILES={
'configs/modelslim_config.py':('92c924abc0af1ae37d7614ee01cbae0b3be1da86ef6abc885b2208560f3dcbb1',49091),
'method_adapters.py':('341d93cae5774414a810444a50eb752cd425330156f404d1b9a10381e6bc0138',15773),
'methods/__init__.py':('cd0e11a198dc6b6eb49f261b7e25d73f42f11cb893d08e5dc126396dd5d887f2',4307),
'methods/base.py':('51a7aafcc49b64a53902eeb3619023e3d79f9c8f4aaf846205dc83dfd35b0d24',14341),
'methods/registry.py':('65b5cc268362c87d6b0e5cda5cf37ca5d54264ad54c3f2896dda35c8d9410be5',2040),
'methods/w8a8/w8a8_dynamic.py':('a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910',25282),
}
code=r'''import base64,hashlib,json
from pathlib import Path
root=Path(%r)
files=%r
out=[]
for n in files:
 p=root/n
 if p.is_symlink() or not p.is_file() or p.stat().st_size>100000: raise SystemExit('invalid file '+str(p))
 b=p.read_bytes()
 if len(b)>100000: raise SystemExit('oversize')
 out.append({'path':n,'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode('ascii')})
print(json.dumps(out))
'''%(ROOT,list(FILES))
def remote(args):
 r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=90)
 if r.returncode: raise RuntimeError('read-only query failed, exit '+str(r.returncode))
 return r.stdout
if remote(['docker','inspect','--format','{{.Image}}',C]).strip()!=IMAGE: raise RuntimeError('unexpected image')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for m in mounts:
 d=m.get('Destination','').rstrip('/') or '/'; root=ROOT.rstrip('/')
 if d=='/' or root==d or root.startswith(d+'/') or d.startswith(root+'/'): raise RuntimeError('quantization subtree overmounted')
entries=json.loads(remote(['docker','exec',C,'python3','-c',code]))
if [e['path'] for e in entries]!=list(FILES): raise RuntimeError('unexpected file list')
records=[]
for e in entries:
 b=base64.b64decode(e['data'],validate=True); h=hashlib.sha256(b).hexdigest()
 if (h,len(b))!=FILES[e['path']] or h!=e['sha256'] or len(b)!=e['size']: raise RuntimeError('inventory hash or size mismatch')
 p=T/'agents/fusion/frozen-source/vllm_ascend/quantization'/e['path']
 if p.is_symlink(): raise RuntimeError('snapshot is symlink')
 if p.exists() and p.read_bytes()!=b: raise RuntimeError('snapshot differs')
 if not p.exists():
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f: f.write(b)
 records.append({'source':ROOT+e['path'],'snapshot':str(p.relative_to(T)),'size':len(b),'sha256':h})
print(json.dumps({'status':'SOURCE_SNAPSHOTS_VERIFIED','container':C,'image_id':IMAGE,'source_subtree_overmounted':False,'files':records,'no_device_work':True},indent=2))
