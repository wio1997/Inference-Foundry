#!/usr/bin/env python3
"""Run bounded source discovery in the staging container; never touch NPU."""
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2]
S=T/'agents/fusion/query_f23_missing_interfaces.py'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
EXPECTED='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
ROOT='/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine'
def remote(args):
 r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=120)
 if r.returncode: raise RuntimeError('read-only remote command failed, exit '+str(r.returncode))
 return r.stdout
image=remote(['docker','inspect','--format','{{.Image}}',C]).strip()
if image!=EXPECTED: raise RuntimeError('unexpected container image')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for m in mounts:
 dest=m.get('Destination','').rstrip('/') or '/'
 if dest=='/' or ROOT==dest or ROOT.startswith(dest+'/') or dest.startswith(ROOT+'/'):
  raise RuntimeError('source subtree overmounted; refusing provenance claim')
x=json.loads(remote(['docker','exec',C,'python3','-c',S.read_text()]))
if x.get('root')!=ROOT: raise RuntimeError('unexpected query root')
x.update(container=C,image_id=image,source_subtree_overmounted=False,script_sha256=hashlib.sha256(S.read_bytes()).hexdigest(),no_device_work=True)
out=T/'agents/fusion/f23-missing-interfaces.json'
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
if out.is_symlink(): raise RuntimeError('output is symlink')
if out.exists():
 if out.read_bytes()!=body: raise RuntimeError('existing output differs; refusing overwrite')
else:
 with out.open('xb') as f:f.write(body)
print(json.dumps({'scan_complete':x['scan_complete'],'reason':x.get('reason'),'missing_names':x['missing_names'],'entries_seen':x['entries_seen'],'files':[ {k:v for k,v in f.items() if k!='snippet'} for f in x['files'] ],'output_sha256':hashlib.sha256(body).hexdigest()},ensure_ascii=False,indent=2))
