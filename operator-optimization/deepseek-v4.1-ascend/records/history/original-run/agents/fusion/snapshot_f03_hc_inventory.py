#!/usr/bin/env python3
"""Read bounded HC sources from only the two frozen 169 containers; never execute model/NPU code."""
import base64
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
EXPECTED_IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
CONTAINERS=('wio-vllm-v41-repro169-20261028','wio-vllm-v41-repro169-stage-20261028')
ROOTS=('/vllm-workspace/vllm-ascend/csrc/moe/hc_pre',
       '/vllm-workspace/vllm-ascend/csrc/moe/hc_post',
       '/vllm-workspace/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py')
inside=HERE/'f03_hc_inventory_inside.py'
script=inside.read_text()

def remote(args):
    r=subprocess.run([SSH,shlex.join(args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
    if r.returncode: raise RuntimeError('remote read failed, exit '+str(r.returncode))
    return r.stdout.decode('utf-8')

images=[]
results=[]
for container in CONTAINERS:
    image=remote(['docker','inspect','--format','{{.Image}}',container]).strip()
    if image!=EXPECTED_IMAGE: raise RuntimeError('unexpected image: '+container)
    mounts=json.loads(remote(['docker','inspect','--format','{{json .Mounts}}',container])) or []
    for m in mounts:
        d=m.get('Destination','').rstrip('/') or '/'
        for root in ROOTS:
            if d=='/' or root==d or root.startswith(d+'/') or d.startswith(root+'/'):
                raise RuntimeError('source path mounted or overmounted: '+container+' '+d)
    payload=json.loads(remote(['docker','exec',container,'python3','-c',script]))
    for record in payload['files']:
        b=base64.b64decode(record['base64'],validate=True)
        if len(b)!=record['size'] or hashlib.sha256(b).hexdigest()!=record['sha256']:
            raise RuntimeError('transfer verification failed: '+record['path'])
    images.append({'name':container,'image_id':image,'checked_mount_intersection':False})
    results.append(payload)
if results[0]!=results[1]: raise RuntimeError('baseline and stage source inventories differ')
body={'status':'SOURCE_INVENTORY_ONLY_NOT_RUNNING_BINARY_PROOF',
      'containers':images,'inside_script_sha256':hashlib.sha256(inside.read_bytes()).hexdigest(),
      'inventory':results[0], 'no_npu_work':True,'target_code_modified':False}
encoded=(json.dumps(body,indent=2,sort_keys=True)+'\n').encode()
out=HERE/'f03-hc-inventory.json'
if out.is_symlink(): raise RuntimeError('snapshot is symlink')
if out.exists():
    if out.read_bytes()!=encoded: raise RuntimeError('existing snapshot differs; refusing overwrite')
else:
    with out.open('xb') as f: f.write(encoded)
print(json.dumps({'status':body['status'], 'source_count':len(results[0]['files']),
                  'missing':results[0]['missing'],'snapshot_sha256':hashlib.sha256(encoded).hexdigest()},indent=2))
