"""Disabled minimal independent watchdog; never signals local processes."""
import os,sys,subprocess,time,json,shlex
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('optimized Python forbidden')
if __name__!='__main__' or len(sys.argv)!=5 or sys.argv[1]!='--execute':
    print('Review draft only');raise SystemExit(0)
raise SystemExit('Diagnostic complete; re-review required before execution')
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent))
import engram_liveness as live
run_id=sys.argv[2];owner=int(sys.argv[3]);birth=json.loads(sys.argv[4])
if len(run_id)!=32 or any(c not in '0123456789abcdef' for c in run_id):raise ValueError('bad run id')
status=here/('watchdog-'+run_id+'.json');done=here/('done-'+run_id)
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
work='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/engram-diagnostic-'+run_id

def save(state):
    p=status.with_suffix('.tmp');p.write_text(json.dumps({'state':state,'run_id':run_id,'pid':os.getpid(),'time':time.time()})+'\n');p.replace(status)

def lease():
    code='import sys\nif sys.flags.optimize: raise RuntimeError("optimized Python forbidden")\nfrom pathlib import Path;import json,time;p=Path('+repr(work)+');p.mkdir(parents=True,exist_ok=True);q=p/"watchdog-lease.json";t=p/"watchdog-lease.tmp";t.write_text(json.dumps({"binding_run_id":'+repr(run_id)+',"epoch":time.time()}));t.replace(q)'
    r=subprocess.run([ssh,shlex.join(['docker','exec',stage,'python3','-c',code])],capture_output=True,timeout=12)
    if r.returncode:raise RuntimeError('lease unavailable')

lease();save('READY');deadline=time.monotonic()+7500
while not done.exists() and time.monotonic()<deadline:
    try:alive=live.identity(owner)==birth
    except (OSError,RuntimeError):alive=False
    if not alive:break
    try:lease();save('READY')
    except Exception:save('LEASE_UNAVAILABLE')
    time.sleep(6)
# Whether the controller completed or failed, continue monitoring the exact
# baseline for a bounded period. A late stop is handled by the same guarded,
# idempotent recovery; monitoring is not proof no later stop can ever occur.
BASE_ID='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092'
health_code="""import json,subprocess,sys
if sys.flags.optimize:raise RuntimeError('optimized Python forbidden')
b=json.loads(subprocess.check_output(['docker','inspect',BASE_ID],text=True))[0]
if b['Id']!=BASE_ID:raise RuntimeError('baseline identity mismatch')
check='import urllib.request; op=urllib.request.build_opener(urllib.request.ProxyHandler({})); print(op.open(\"http://127.0.0.1:8172/health\",timeout=5).status)'
r=subprocess.run(['docker','exec',BASE_ID,'python3','-c',check],capture_output=True,text=True,timeout=15) if b['State']['Running'] else None
print(json.dumps({'running':b['State']['Running'],'health':int(r.stdout.strip()) if r and r.returncode==0 else None}))
"""
def observe():
    pre='BASE_ID='+repr(BASE_ID)+'\n'
    r=subprocess.run([ssh,shlex.join(['python3','-c',pre+health_code])],capture_output=True,text=True,timeout=30)
    if r.returncode:raise RuntimeError('baseline observation unavailable')
    return json.loads(r.stdout)

env=dict(os.environ,WIO_BINDING_RUN_ID=run_id)
healthy_since=None
need_fence_recovery=True
while True:
    try:
        if need_fence_recovery:
            save('FENCE_RECOVERY_REQUIRED')
            r=subprocess.run([sys.executable,str(here/'engram_adapter_recover.py'),'--execute'],env=env,timeout=3000)
            if r.returncode:
                save('RECOVERY_PENDING');time.sleep(30);continue
            need_fence_recovery=False
        state=observe()
        if state['running'] and state['health']==200:
            if healthy_since is None:healthy_since=time.monotonic()
            save('BASELINE_HEALTHY_MONITORING')
            if time.monotonic()-healthy_since>=300:
                save('BASELINE_HEALTHY_OBSERVED_5MIN');raise SystemExit(0)
        else:
            healthy_since=None
            if state['running']:
                save('BASELINE_RUNNING_HEALTH_PENDING')
            else:
                save('BASELINE_STOPPED_RECOVERY_RETRY')
                need_fence_recovery=True
    except Exception:healthy_since=None;save('RECOVERY_PENDING')
    time.sleep(30)
