"""Independent watchdog: ready lease before park, recover if owner exits."""
from pathlib import Path
import json, os, shlex, subprocess, sys, time

if __name__!='__main__' or len(sys.argv)!=4 or sys.argv[1]!='--execute':
    print('Review draft: no operations. Launched by orchestrate_binding.py.');raise SystemExit(0)
run_id=sys.argv[2];owner=int(sys.argv[3])
assert len(run_id)==32 and all(c in '0123456789abcdef' for c in run_id)
here=Path(__file__).resolve().parent
status=here/('watchdog-'+run_id+'.json');done=here/('watchdog-'+run_id+'.done')
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
stage='wio-vllm-v41-repro169-stage-20261028'
work='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binding-diagnostic-selective2'

def save(state,**kw):
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'binding_run_id':run_id,'pid':os.getpid(),'owner':owner,'time':time.time(),**kw})+'\n');tmp.replace(status)

def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True

def lease():
    code='from pathlib import Path;import json,time;p=Path('+repr(work)+');p.mkdir(parents=True,exist_ok=True);q=p/"watchdog-lease.json";t=p/"watchdog-lease.tmp";t.write_text(json.dumps({"binding_run_id":'+repr(run_id)+',"epoch":time.time()}));t.replace(q)'
    remote=shlex.join(['docker','exec',stage,'python3','-c',code])
    r=subprocess.run([ssh,remote],capture_output=True,text=True,timeout=12)
    if r.returncode:raise RuntimeError(r.stderr[-1000:])

try:
    lease();save('READY')
    while live(owner) and not done.exists():
        time.sleep(6)
        if done.exists():break
        try:lease();save('READY')
        except Exception as exc:save('LEASE_ERROR',error=repr(exc))
    if done.exists():save('COMPLETED');raise SystemExit(0)
    save('OWNER_EXITED_RECOVERY_STARTING')
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        env=dict(os.environ);env['WIO_BINDING_RUN_ID']=run_id
        try:
            r=subprocess.run([sys.executable,str(here/'recover_binding.py'),'--execute'],env=env,capture_output=True,text=True,timeout=3000)
            if r.returncode==0:save('BASELINE_HEALTHY');raise SystemExit(0)
            save('RECOVERY_RETRY',returncode=r.returncode,detail=(r.stderr+r.stdout)[-1200:])
        except Exception as exc:save('RECOVERY_RETRY',error=repr(exc))
        time.sleep(30)
    save('RECOVERY_RETRY_TIMEOUT');raise SystemExit(2)
except Exception as exc:
    save('WATCHDOG_ERROR',error=repr(exc));raise
