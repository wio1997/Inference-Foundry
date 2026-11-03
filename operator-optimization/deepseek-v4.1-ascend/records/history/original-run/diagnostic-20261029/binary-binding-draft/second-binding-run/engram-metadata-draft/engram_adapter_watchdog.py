import sys
if sys.flags.optimize:
    raise RuntimeError('optimized Python is forbidden for diagnostic safety checks')
"""Disabled watchdog: progress lease, timeout, verified group termination, recovery."""
from pathlib import Path
import json,os,shlex,signal,subprocess,sys,time
import engram_liveness as live

if __name__!='__main__' or len(sys.argv)!=4 or sys.argv[1]!='--execute':
    print('Review draft: no operations.');raise SystemExit(0)
raise SystemExit('DISABLED: review the independent adapter before enabling execution')

run_id=sys.argv[2];owner=int(sys.argv[3])
if len(run_id)!=32 or any(c not in '0123456789abcdef' for c in run_id):raise ValueError('bad run identity')
here=Path(__file__).resolve().parent
status=here/('watchdog-'+run_id+'.json');done=here/('watchdog-'+run_id+'.done');heartbeat=here/('heartbeat-'+run_id+'.json')
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';stage='wio-vllm-v41-repro169-stage-20261028'
work='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/engram-diagnostic-'+run_id


def save(state,**kw):
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps({'state':state,'binding_run_id':run_id,'pid':os.getpid(),'owner':owner,'time':time.time(),**kw})+'\n');tmp.replace(status)


def read_progress():
    h={}
    try:
        h=json.loads(heartbeat.read_text())
        result=live.check_heartbeat(h,run_id,time.monotonic())
        if result!='FRESH':return result,h
        if h.get('owner_pid')!=owner:return 'OWNER_MISMATCH',h
        if live.identity(owner)!=h.get('owner_birth'):return 'OWNER_IDENTITY_MISMATCH',h
        if os.getsid(owner)!=owner or os.getpgid(owner)!=owner:return 'OWNER_SESSION_MISMATCH',h
        return 'FRESH',h
    except Exception:return 'UNKNOWN',h


def lease(h):
    code='import sys\nif sys.flags.optimize: raise RuntimeError("optimized Python forbidden")\nfrom pathlib import Path;import json,time;p=Path('+repr(work)+');p.mkdir(parents=True,exist_ok=True);q=p/"watchdog-lease.json";t=p/"watchdog-lease.tmp";t.write_text(json.dumps({"binding_run_id":'+repr(run_id)+',"epoch":time.time(),"phase":'+repr(h['phase'])+'}));t.replace(q)'
    remote=shlex.join(['docker','exec',stage,'python3','-c',code])
    r=subprocess.run([ssh,remote],capture_output=True,text=True,timeout=12)
    if r.returncode:raise RuntimeError('watchdog lease write failed')


try:
    state,h=read_progress()
    if state!='FRESH':raise RuntimeError('controller heartbeat not authenticated')
    lease(h);save('READY')
    last=h
    while not done.exists():
        time.sleep(6)
        if done.exists():break
        state,h=read_progress()
        if state!='FRESH':
            save('OWNER_STALLED_OR_LOST',reason=state)
            frozen=False
            try:
                if not live.snapshot_bound(h,last,run_id,owner,time.monotonic()):
                    save('TAKEOVER_SNAPSHOT_UNKNOWN',reason=state);continue
                if live.owner_absent(owner):
                    # Latest gate record survives owner death; children may be
                    # reparented, so authenticate birth/session, not old ppid.
                    child=live.stop_owned_group(h['child_pid'],h['child_birth'])
                    if child not in ('NO_CHILD_RECORDED','EXIT_CONFIRMED'):
                        save('OWNER_DEAD_CHILD_UNKNOWN',result=child);continue
                    save('OWNER_ABSENT_CHILD_GROUP_ABSENT_REMOTE_UNCONFIRMED');break
                if not live.validate_takeover_snapshot(h,last,run_id,owner,time.monotonic()):
                    save('TAKEOVER_IDENTITY_UNKNOWN',reason=state);continue
                if not live.signal_owned_group(owner,h['owner_birth'],signal.SIGSTOP):
                    save('TAKEOVER_OWNER_FREEZE_UNKNOWN');continue
                frozen=True
                if not live.wait_owner_stopped(owner,h['owner_birth']):
                    save('OWNER_STOP_UNCONFIRMED');continue
                frozen_state,current=read_progress()
                if not live.validate_takeover_snapshot(current,last,run_id,owner,time.monotonic()):
                    save('TAKEOVER_FROZEN_IDENTITY_UNKNOWN',reason=frozen_state);continue
                child=live.stop_owned_group(current['child_pid'],current['child_birth'],expected_parent=owner)
                if child not in ('NO_CHILD_RECORDED','EXIT_CONFIRMED'):
                    save('TAKEOVER_FROZEN_CHILD_UNKNOWN',result=child);continue
                result=live.stop_owned_group(owner,current['owner_birth'])
                if result!='EXIT_CONFIRMED':
                    save('TAKEOVER_OWNER_EXIT_UNKNOWN',result=result);continue
                frozen=False
                break
            except (OSError,RuntimeError) as exc:
                save('TAKEOVER_UNKNOWN',error=type(exc).__name__)
            finally:
                if frozen:
                    # Only unfreeze the same authenticated owner. Otherwise
                    # preserve UNKNOWN and retry under the independent watchdog.
                    try:
                        resumed=live.signal_owned_group(owner,last['owner_birth'],signal.SIGCONT)
                        save('OWNER_RESUMED_SUPERVISION_CONTINUES' if resumed else 'FROZEN_OWNER_IDENTITY_UNKNOWN')
                    except (OSError,RuntimeError):save('FROZEN_OWNER_IDENTITY_UNKNOWN')
            continue

        last=h
        try:lease(h);save('READY')
        except Exception as exc:save('LEASE_ERROR',error=type(exc).__name__)
    if done.exists():save('COMPLETED');raise SystemExit(0)
    save('LOCAL_GROUPS_ABSENT_REMOTE_UNCONFIRMED_RECOVERY_STARTING')
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        env=dict(os.environ);env['WIO_BINDING_RUN_ID']=run_id
        try:
            r=subprocess.run([sys.executable,str(here/'engram_adapter_recover.py'),'--execute'],env=env,capture_output=True,text=True,timeout=3000)
            if r.returncode==0:save('BASELINE_HEALTHY');raise SystemExit(0)
            save('RECOVERY_RETRY',returncode=r.returncode)
        except Exception as exc:save('RECOVERY_RETRY',error=type(exc).__name__)
        time.sleep(30)
    save('RECOVERY_RETRY_TIMEOUT');raise SystemExit(2)
except Exception as exc:
    save('WATCHDOG_ERROR',error=type(exc).__name__);raise
