#!/usr/bin/env python3
"""Read exact owned container states and NPU process ownership, no mutations."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

CODE=r'''
import datetime,json,re,subprocess
base='wio-vllm-v41-repro169-20261028';stage='wio-vllm-v41-repro169-stage-20261028'
expected={base:'b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092',stage:'42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602'}
image='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
def run(args):return subprocess.run(args,text=True,capture_output=True,timeout=25,check=False)
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'read only exact owned identities, owned-prefix container list and NPU process table; no restart/stop/workload','containers':{}}
for name in [base,stage]:
 result=run(['docker','inspect',name]);assert result.returncode==0,result.stderr
 d=json.loads(result.stdout)[0];assert d['Id']==expected[name] and d['Image']==image
 out['containers'][name]={'Id':d['Id'],'Image':d['Image'],'State':d['State']}
listed=run(['docker','ps','-a','--filter','name=^/wio-vllm-v41-','--format','{{.ID}}\t{{.Names}}\t{{.Status}}'])
out['owned_prefix_list']={'returncode':listed.returncode,'rows':listed.stdout.splitlines(),'stderr':listed.stderr}
assert listed.returncode==0 and len(listed.stdout)<50000
if out['containers'][stage]['State']['Running']:
 result=run(['docker','exec',stage,'npu-smi','info']);out['npu_smi']={'returncode':result.returncode,'raw':result.stdout,'stderr':result.stderr}
 pids=set()
 for row in result.stdout.splitlines():
  cols=[x.strip() for x in row.split('|')]
  if len(cols)>4 and re.fullmatch(r'\d+\s+\d+',cols[1]) and cols[2].isdigit():pids.add(int(cols[2]))
 own=set()
 if out['containers'][base]['State']['Running']:
  result=run(['docker','top',base,'-eo','pid'])
  if result.returncode==0:own={int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()}
 out['npu_process_ids']=sorted(pids);out['processes_not_in_baseline_pid_set']=sorted(pids-own)
out['baseline_running']=out['containers'][base]['State']['Running']
out['stage_running']=out['containers'][stage]['State']['Running']
out['model_or_npu_load_started']=False
print(json.dumps(out,indent=2))
'''
W=Path(__file__).resolve().parent;payload=base64.b64encode(CODE.encode()).decode()
command=shlex.join(['python3','-c',"import base64;exec(base64.b64decode('"+payload+"'))"])
with (W/'owned-environment-readiness.json').open('x') as out,(W/'owned-environment-readiness.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=120,check=False)
if result.returncode:sys.exit(result.returncode)
d=json.loads((W/'owned-environment-readiness.json').read_text())
print(json.dumps({'returncode':result.returncode,'baseline_running':d['baseline_running'],'stage_running':d['stage_running'],
    'npu_process_ids':d.get('npu_process_ids'),'read_only':True}))
