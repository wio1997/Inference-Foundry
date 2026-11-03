#!/usr/bin/env python3
"""Snapshot one exact source file after image, mount, size and SHA verification."""
import base64,hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
SRC='/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute/moe_token_unpermute.h'
SHA='08982836ca51157eb674d447c66730cfeffdf8080d7172a157d887c7e58c3a0d'
SIZE=16454
remote_script=r'''import base64,hashlib,json
from pathlib import Path
p=Path(%r)
if p.is_symlink() or not p.is_file(): raise SystemExit('not a regular source')
if p.stat().st_size != %d: raise SystemExit('unexpected size')
with p.open('rb') as f: b=f.read(%d)
if len(b)!=%d or hashlib.sha256(b).hexdigest()!=%r: raise SystemExit('sha or size mismatch')
print(json.dumps({'data':base64.b64encode(b).decode('ascii'),'sha256':hashlib.sha256(b).hexdigest(),'size':len(b)}))
'''%(SRC,SIZE,SIZE+1,SIZE,SHA)
def remote(args):
 r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=120)
 if r.returncode: raise RuntimeError('read-only source query failed, exit '+str(r.returncode))
 return r.stdout
if remote(['docker','inspect','--format','{{.Image}}',C]).strip()!=IMAGE: raise RuntimeError('unexpected image')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
for m in mounts:
 dest=m.get('Destination','').rstrip('/') or '/'
 if dest=='/' or SRC==dest or SRC.startswith(dest+'/'): raise RuntimeError('source overmounted')
x=json.loads(remote(['docker','exec',C,'python3','-c',remote_script]))
b=base64.b64decode(x['data'],validate=True)
if len(b)!=SIZE or hashlib.sha256(b).hexdigest()!=SHA or x['sha256']!=SHA: raise RuntimeError('source integrity check failed')
out=T/'agents/fusion/frozen-source/csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute/moe_token_unpermute.h'
if out.is_symlink(): raise RuntimeError('output is symlink')
if out.exists() and out.read_bytes()!=b: raise RuntimeError('existing source differs')
if not out.exists():
 out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('xb') as f:f.write(b)
print(json.dumps({'status':'SOURCE_SNAPSHOT_SHA_VERIFIED','source':SRC,'container':C,'image_id':IMAGE,'size':SIZE,'sha256':SHA,'snapshot':str(out.relative_to(T)),'no_device_work':True},indent=2))
