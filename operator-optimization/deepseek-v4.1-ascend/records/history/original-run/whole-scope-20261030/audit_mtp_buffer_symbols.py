#!/usr/bin/env python3
"""Read-only exact-symbol audit of tracked code in both pinned repositories."""
import base64
from pathlib import Path
import shlex
import subprocess
import sys

CODE=r'''
import datetime,hashlib,json,pathlib,subprocess
roots={'vllm':pathlib.Path('/vllm-workspace/vllm'),'vllm-ascend':pathlib.Path('/vllm-workspace/vllm-ascend')}
expected={'vllm':'ced6857afa0ea7b2e3f0846a62e1394e90f15607','vllm-ascend':'8fde6d4e335fe300e2b949aab0e31659cce0811d'}
out={'scope':'exact two-symbol search of tracked Python/C++ headers; dynamic aliases/reflection not closed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repositories':{}}
for name,root in roots.items():
 head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
 status=subprocess.check_output(['git','-C',str(root),'status','--short','--untracked-files=no'],text=True)
 assert head==expected[name] and not status,(name,head,status)
 command=['git','-C',str(root),'grep','-n','-e','_mtp_hidden_buffer','-e','get_mtp_target_hidden_states','--','*.py','*.cpp','*.h','*.hpp']
 result=subprocess.run(command,text=True,capture_output=True,check=False)
 assert result.returncode in (0,1) and not result.stderr and len(result.stdout)<120000
 matches=[]
 for line in result.stdout.splitlines():
  relative,number,text=line.split(':',2);p=root/relative
  matches.append({'source':relative,'line':int(number),'text':text,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 out['repositories'][name]={'commit':head,'tracked_status':status,'returncode':result.returncode,'matches':matches}
print(json.dumps(out,indent=2))
'''
root=Path(__file__).resolve().parent
payload=base64.b64encode(CODE.encode()).decode()
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',"import base64;exec(base64.b64decode('"+payload+"'))"])
with (root/'mtp-buffer-symbol-audit.json').open('x') as out,(root/'mtp-buffer-symbol-audit.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=40,check=False)
print('returncode',result.returncode);sys.exit(result.returncode)
