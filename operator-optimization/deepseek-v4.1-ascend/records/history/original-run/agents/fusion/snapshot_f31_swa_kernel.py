#!/usr/bin/env python3
"""Read-only snapshot of one frozen-image DSpark SWA Triton source file via existing 169 SSH.

Writes only agents/fusion/frozen-source/ and the matching provenance JSON.
Fails closed on differing image IDs, workspace mounts, hash mismatch, missing
source, oversized source, or a preexisting destination with different bytes.
Never imports target modules or touches NPU. Output contains no credentials.
"""
import base64
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'frozen-source'
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
EXPECTED_IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
CONTAINERS = ('wio-vllm-v41-repro169-20261028', 'wio-vllm-v41-repro169-stage-20261028')
SOURCES = (
    '/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/spec_decode/dspark_swa_indices.py',
)
REMOTE = r'''import base64,hashlib,json
from pathlib import Path
paths=%s
out=[]
for raw in paths:
 p=Path(raw)
 if p.is_symlink() or not p.is_file():
  raise SystemExit('source missing, nonregular or symlink: '+raw)
 b=p.read_bytes()
 if len(b)>1048576: raise SystemExit('source exceeds 1 MiB: '+raw)
 out.append({'path':raw,'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode('ascii')})
print(json.dumps(out))
''' % repr(list(SOURCES))

def remote(args):
 result = subprocess.run([SSH, shlex.join(args)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, timeout=120, check=False)
 if result.returncode:
  # Avoid echoing remote environment, credentials or command output.
  raise RuntimeError('remote command failed (exit %s); no files written' % result.returncode)
 return result.stdout.decode('utf-8')

def existing_matches_or_absent(path, data):
 if path.is_symlink(): raise RuntimeError('refusing symlink: '+str(path))
 if path.exists() and path.read_bytes() != data:
  raise RuntimeError('snapshot already exists with different content: '+str(path))

meta=[]; data_by_container=[]
for container in CONTAINERS:
 image_id=remote(['docker','inspect','--format','{{.Image}}',container]).strip()
 if image_id != EXPECTED_IMAGE: raise RuntimeError('unexpected frozen image: '+container)
 mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',container])) or []
 for m in mounts:
  dest=m.get('Destination','').rstrip('/') or '/'
  for src in SOURCES:
   if dest=='/' or src==dest or src.startswith(dest+'/'):
    raise RuntimeError('source path is container-overmounted: '+container+' '+dest)
 source=json.loads(remote(['docker','exec',container,'python3','-c',REMOTE]))
 if [x['path'] for x in source] != list(SOURCES): raise RuntimeError('unexpected remote source list')
 for item in source:
  b=base64.b64decode(item.pop('base64'),validate=True)
  if len(b)!=item['size'] or hashlib.sha256(b).hexdigest()!=item['sha256']:
   raise RuntimeError('transfer integrity failed: '+item['path'])
  item['bytes']=b
 meta.append({'container':container,'image_id':image_id,'workspace_source_overmount':False})
 data_by_container.append(source)

for a,b in zip(*data_by_container):
 if a['sha256']!=b['sha256'] or a['bytes']!=b['bytes']:
  raise RuntimeError('baseline/stage source mismatch: '+a['path'])
records=[]; pending=[]
for item in data_by_container[0]:
 path=OUT / Path(item['path']).relative_to('/vllm-workspace/vllm-ascend')
 existing_matches_or_absent(path,item['bytes'])
 pending.append((path,item['bytes']))
 records.append({'original_path':item['path'],'snapshot_path':str(path.relative_to(TASK)),
                 'sha256':item['sha256'],'size':item['size']})
doc={'status':'READ_ONLY_MATCHING_CONTAINER_SOURCE_SNAPSHOT','containers':meta,
     'source_files':records,'no_npu_work':True,'target_code_modified':False}
receipt=Path(__file__).resolve().parent / 'f31-swa-kernel-provenance.json'
receipt_bytes=(json.dumps(doc,ensure_ascii=False,indent=2)+'\n').encode()
existing_matches_or_absent(receipt,receipt_bytes)
for path,body in pending:
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  with path.open('xb') as f:f.write(body)
if not receipt.exists():
 with receipt.open('xb') as f:f.write(receipt_bytes)
print(json.dumps(doc,ensure_ascii=False,indent=2))
