#!/usr/bin/env python3
"""Read two named metadata files in the owned service; no target imports."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

CODE = r'''
import datetime, hashlib, json, pathlib, subprocess
roots={'vllm':pathlib.Path('/vllm-workspace/vllm'),'vllm-ascend':pathlib.Path('/vllm-workspace/vllm-ascend')}
expected={'vllm':'ced6857afa0ea7b2e3f0846a62e1394e90f15607','vllm-ascend':'8fde6d4e335fe300e2b949aab0e31659cce0811d'}
out={'scope':'read only two fixed metadata files; current on-disk values, not proof of historical loaded objects','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':{},'files':{},'missing':[]}
for name,root in roots.items():
 head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
 status=subprocess.check_output(['git','-C',str(root),'status','--short','--untracked-files=no'],text=True)
 assert head==expected[name] and not status,(name,head,status)
 out['roots'][name]={'head':head,'tracked_status':status}
for raw in ['/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/generation_config.json','/vllm-workspace/vllm-ascend/vllm_ascend/_build_info.py']:
 p=pathlib.Path(raw)
 if not p.is_file():out['missing'].append(raw);continue
 b=p.read_bytes();assert len(b)<65536
 out['files'][raw]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode()}
print(json.dumps(out,indent=2))
'''
root = Path(__file__).resolve().parent
payload = base64.b64encode(CODE.encode()).decode()
command = shlex.join(['docker', 'exec', 'wio-vllm-v41-repro169-20261028', 'python3', '-c', "import base64;exec(base64.b64decode('" + payload + "'))"])
with (root/'fixed-contract-metadata.json').open('x') as out, (root/'fixed-contract-metadata.stderr').open('w') as err:
    result = subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=40,check=False)
print(json.dumps({'returncode':result.returncode,'output':'whole-scope-20261030/fixed-contract-metadata.json','no_device_or_service_requests':True}))
sys.exit(result.returncode)
