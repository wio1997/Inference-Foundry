#!/usr/bin/env python3
"""Read tracked source aliases for the missing qwen3_dspark dependency only."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

CODE=r'''
import hashlib,json,pathlib,subprocess
root=pathlib.Path('/vllm-workspace/vllm')
tracked=subprocess.check_output(['git','-C',str(root),'ls-files','*qwen3*dspark*'],text=True).splitlines()
selected=set(tracked)|{'vllm/models/__init__.py','vllm/model_executor/models/__init__.py','vllm/utils/import_utils.py'}
out={'scope':'read-only tracked source layout for missing qwen3_dspark import','tracked_matches':tracked,'files':{},'missing':[]}
for relative in sorted(selected):
 p=root/relative
 if p.is_file() and p.suffix=='.py':
  b=p.read_bytes();out['files'][relative]={'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()}
 else:out['missing'].append(relative)
print(json.dumps(out,indent=2))
'''
payload=base64.b64encode(CODE.encode()).decode();code="import base64;exec(base64.b64decode('"+payload+"'))"
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code])
root=Path(__file__).resolve().parent
with (root/'model-import-layout.json').open('w') as out,(root/'model-import-layout.stderr').open('w') as err:
 p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=30,check=False)
print('returncode',p.returncode);sys.exit(p.returncode)
