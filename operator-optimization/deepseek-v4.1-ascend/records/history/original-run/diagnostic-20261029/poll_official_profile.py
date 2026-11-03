"""Inspect the exact submitted controller and its existing outputs; never restarts it."""
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
receipt=json.loads((HERE/'official-launch-latest.json').read_text())
run=receipt['run_id'];pid=receipt['controller_pid']
remote=r"""
import json,pathlib,subprocess
out={'run_id':RUN,'controller_pid':PID}
try:
 raw=pathlib.Path('/proc/'+str(PID)+'/stat').read_text();state=raw[raw.rfind(')')+2:].split()[0]
 env=pathlib.Path('/proc/'+str(PID)+'/environ').read_bytes().split(b'\0')
 out['controller_live']=state!='Z' and ('WIO_RUN_ID='+RUN).encode() in env
 out['process_state']=state
except FileNotFoundError:out['controller_live']=False
stage='wio-vllm-v41-repro169-stage-20261028'
root='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/'+RUN
code='''import json,pathlib
r=pathlib.Path(ROOT);out={}
for n in ['controller-state.json','controller-error.json','top-level-error.json','comparison-validation.json','trace-inventory.json']:
 p=r/n
 if p.exists():
  try:out[n]=json.loads(p.read_text())
  except Exception as e:out[n]={'read_error':str(e)}
p=r/'runtime/logs/service.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));out['service_log_tail']=f.read().decode(errors='replace')[-6000:]
for phase in ['warmup','baseline','profile']:
 p=r/('runtime/client-'+phase+'/window-status.json')
 if p.exists():
  j=json.loads(p.read_text());out['client_'+phase]={k:j[k] for k in ['phase','valid','windows','decode_gap','counter_deltas'] if k in j}
print(json.dumps(out))
'''
p=subprocess.run(['docker','exec',stage,'python3','-c','ROOT='+repr(root)+'\n'+code],capture_output=True,text=True,timeout=20)
out['outputs']=json.loads(p.stdout) if p.returncode==0 else {'read_error':p.stderr[-1200:]}
for name in ['wio-vllm-v41-repro169-20261028','wio-vllm-v41-'+RUN]:
 p=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True,timeout=10)
 out[name]=json.loads(p.stdout) if p.returncode==0 else {'not_found_or_error':p.stderr[-500:]}
print(json.dumps(out))
"""
code='RUN='+repr(run)+'\nPID='+repr(pid)+'\n'+remote
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',code])],capture_output=True,text=True,timeout=50)
if p.returncode:raise RuntimeError(p.stderr[-1500:])
j=json.loads(p.stdout)
(HERE/'official-status-latest.json').write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(j,ensure_ascii=False,indent=2))
