#!/usr/bin/env python3
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
data=json.loads((root/'iteration-gaps.json').read_text())
specs=[{k:x[k] for k in ('rank','source_dir','gaps')} for x in data['rows'] if x['rank'] in (0,8)]
code='SPECS='+repr(specs)+'\n'+(root/'analyze_prepare_input.py').read_text()
payload=base64.b64encode(code.encode()).decode()
script="import base64;exec(compile(base64.b64decode('"+payload+"'),'<prepare_input_analysis>','exec'))"
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',script])
start=time.time()
with (root/'prepare-input-analysis.json').open('w') as out,(root/'prepare-analysis.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=180,check=False)
(root/'prepare-analysis-receipt.json').write_text(json.dumps({'returncode':result.returncode,'started_epoch':start,'ended_epoch':time.time(),'no_device_work':True},indent=2)+'\n')
print('returncode',result.returncode)
sys.exit(result.returncode)
