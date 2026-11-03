#!/usr/bin/env python3
"""Verify 169 dedicated container identity/mount and read exactly allowlisted checkpoint headers."""
import hashlib,json,posixpath,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];S=T/'agents/fusion/inspect_169_engram_headers_inside.py'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
CID='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
BASE='/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8'
NAMES=('quant_model_weights.safetensors.index.json','quant_model_weights-engram-extra-1.safetensors','quant_model_weights-engram-extra-14.safetensors')
def remote(args):
 r=subprocess.run([SSH,shlex.join(args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
 if r.returncode:raise RuntimeError('read-only remote operation failed, exit '+str(r.returncode))
 return r.stdout.decode('utf-8')
image=remote(['docker','inspect','--format','{{.Image}}',C]).strip()
if image!=IMAGE:raise RuntimeError('frozen container image differs')
container_id=remote(['docker','inspect','--format','{{.Id}}',C]).strip()
if container_id!=CID:raise RuntimeError('frozen container ID differs')
mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',C])) or []
selected=[]
for name in NAMES:
 target=BASE+'/'+name
 matches=[]
 for m in mounts:
  d=m.get('Destination','').rstrip('/') or '/'
  if target==d or target.startswith(d.rstrip('/')+'/'):
   if m.get('Type')!='bind':raise RuntimeError('checkpoint source is not an allowlisted bind mount')
   host=posixpath.normpath(posixpath.join(m['Source'],posixpath.relpath(target,d)))
   matches.append((len(d),d,host))
 if not matches:raise RuntimeError('checkpoint path has no verified bind mount')
 _,dest,host=max(matches)
 if host!=target:raise RuntimeError('checkpoint path maps outside exact 169 wio model path')
 selected.append({'file':name,'mount_destination':dest,'mapped_host_path':host})
payload=json.loads(remote(['docker','exec',C,'python3','-c',S.read_text()]))
if payload.get('payload_bytes_read')!=0:raise RuntimeError('unexpected payload read')
result={'container':C,'container_id':container_id,'image_id':image,'mounts_for_selected_files':selected,
 'inside_script_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'source':payload,'no_service_changes':True}
body=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode();out=T/'agents/fusion/engram-169-header-evidence.json'
if out.is_symlink():raise RuntimeError('output is symlink')
if out.exists():
 if out.read_bytes()!=body:raise RuntimeError('existing result differs; refusing overwrite')
else:
 with out.open('xb') as f:f.write(body)
print(json.dumps({'status':'HEADERS_READ_ONLY','result_sha256':hashlib.sha256(body).hexdigest(),
 'tensors':payload['tensors'],'payload_bytes_read':0},ensure_ascii=False,indent=2))
