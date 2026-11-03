#!/usr/bin/env python3
"""Read two named installed Transformers sources via distribution metadata only."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

CODE=r'''
import datetime,hashlib,importlib.metadata,json,pathlib
distribution=importlib.metadata.distribution('transformers')
out={'scope':'two installed configuration sources; distribution metadata only, no transformers/torch/vllm import',
     'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':distribution.version,'files':{}}
for relative in ['transformers/configuration_utils.py','transformers/generation/configuration_utils.py']:
 p=pathlib.Path(distribution.locate_file(relative))
 assert p.is_file() and p.stat().st_size<500000
 b=p.read_bytes()
 out['files'][relative]={'installed_path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'text':b.decode()}
print(json.dumps(out,indent=2))
'''
root=Path(__file__).resolve().parent
payload=base64.b64encode(CODE.encode()).decode()
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-20261028','python3','-c',"import base64;exec(base64.b64decode('"+payload+"'))"])
with (root/'installed-transformers-contract.json').open('x') as out,(root/'installed-transformers-contract.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=35,check=False)
if result.returncode:sys.exit(result.returncode)
data=json.loads((root/'installed-transformers-contract.json').read_text())
for relative,item in data['files'].items():
    p=root/'installed-transformers-source'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['text'])
print(json.dumps({'returncode':result.returncode,'transformers_version':data['version'],'file_count':len(data['files']),
    'target_modules_imported':False,'npu_or_service_requests':False}))
