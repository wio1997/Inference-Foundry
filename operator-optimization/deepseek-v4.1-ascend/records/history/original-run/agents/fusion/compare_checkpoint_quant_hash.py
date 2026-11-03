#!/usr/bin/env python3
"""Read-only hash comparison: 169 frozen checkpoint metadata vs local snapshot.

No model weights are opened. Never imports target code or touches NPU.
Fails closed if the frozen image changes or a pre-existing receipt differs.
"""
import hashlib,json,shlex,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent/'checkpoint-metadata-comparison.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER='wio-vllm-v41-repro169-20261028'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
REMOTE_FILE='/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/quant_model_description.json'
LOCAL_FILE=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/source172/model-metadata/quant_model_description.json')
REMOTE_CODE=r'''import hashlib,json
from pathlib import Path
p=Path(%r)
if not p.is_file(): raise SystemExit('checkpoint quant metadata not found')
h=hashlib.sha256(); size=0
with p.open('rb') as f:
 while True:
  b=f.read(1048576)
  if not b: break
  size+=len(b); h.update(b)
print(json.dumps({'size':size,'sha256':h.hexdigest()}))
''' % REMOTE_FILE

def remote(argv):
 r=subprocess.run([SSH,shlex.join(argv)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
 if r.returncode: raise RuntimeError('read-only remote query failed, exit %d; no receipt written'%r.returncode)
 return r.stdout.decode('utf-8')
def file_digest(path):
 h=hashlib.sha256();size=0
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):
   size+=len(b);h.update(b)
 return {'size':size,'sha256':h.hexdigest()}
image=remote(['docker','inspect','--format','{{.Image}}',CONTAINER]).strip()
if image!=IMAGE: raise RuntimeError('frozen image ID mismatch')
remote_digest=json.loads(remote(['docker','exec',CONTAINER,'python3','-c',REMOTE_CODE]))
local_digest=file_digest(LOCAL_FILE)
matched=remote_digest==local_digest
result={'status':'MATCH' if matched else 'MISMATCH','container':CONTAINER,'image_id':image,
 'remote_metadata_path':REMOTE_FILE,'remote_metadata':remote_digest,
 'local_metadata_path':str(LOCAL_FILE),'local_metadata':local_digest,
 'interpretation':'The file hashes match' if matched else 'The file hashes differ; do not equate descriptions',
 'limits':['metadata file bytes only, not a live loaded quant_method object or proof of inference branch','checkpoint file is not a model weight tensor'],
 'no_weight_tensors_read':True,'no_npu_work':True,'no_service_changes':True}
body=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
if OUT.is_symlink(): raise RuntimeError('refusing symlink receipt')
if OUT.exists() and OUT.read_bytes()!=body: raise RuntimeError('receipt exists with different content; preserve prior evidence')
if not OUT.exists():
 with OUT.open('xb') as f:f.write(body)
print(json.dumps(result,ensure_ascii=False,indent=2))
