import sys
if sys.flags.optimize:
    raise RuntimeError('optimized Python is forbidden for diagnostic safety checks')
"""Disabled independent controller with progress heartbeat and bounded phases."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,uuid
import engram_liveness as live

if __name__!='__main__' or sys.argv[1:]!=['--execute']:
    print('Review draft: no operations.');raise SystemExit(0)
raise SystemExit('DISABLED: review the independent adapter before enabling execution')

here=Path(__file__).resolve().parent
fd=os.open(here/'engram-adapter-controller.lock',os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise RuntimeError('another controller owns the local lock')
if os.environ.get('PYTHONOPTIMIZE','') not in ('','0'):raise RuntimeError('optimized Python forbidden')
if os.getsid(0)!=os.getpid():os.setsid()
if os.getsid(0)!=os.getpid() or os.getpgrp()!=os.getpid():raise RuntimeError('controller session not isolated')
run_id=uuid.uuid4().hex;owner=os.getpid();owner_birth=live.identity(owner)
status=here/('watchdog-'+run_id+'.json');done=here/('watchdog-'+run_id+'.done');heartbeat=here/('heartbeat-'+run_id+'.json')
env=dict(os.environ);env['WIO_BINDING_RUN_ID']=run_id
current={'run_id':run_id,'owner_pid':owner,'owner_birth':owner_birth,'child_pid':None,'child_birth':None}


def phase(name):
    if current.get('child_pid') is not None or current.get('child_birth') is not None:
        raise RuntimeError('previous child group exit UNKNOWN; refuse phase transition')
    start=time.monotonic();current.update(phase=name,phase_start=start,deadline=start+live.PHASE_LIMITS[name]);beat()


def beat():
    current['updated']=time.monotonic()
    tmp=heartbeat.with_suffix('.tmp')
    tmp.write_text(json.dumps(current)+'\n');os.chmod(tmp,0o600);os.replace(tmp,heartbeat)


# Child must not act until parent has persisted its exact process identity.
CHILD_GATE="import os,sys\nif sys.flags.optimize: raise RuntimeError('optimized Python forbidden')\nif os.read(0,1)!=b'1': raise SystemExit(125)\nos.execv(sys.executable,[sys.executable,sys.argv[1],'--execute'])\n"


def run_phase(name,filename):
    phase(name)
    p=subprocess.Popen([sys.executable,'-c',CHILD_GATE,str(here/filename)],env=env,start_new_session=True,stdin=subprocess.PIPE)
    # Publish pending identity before further work; the child gate remains closed.
    current.update(child_pid=p.pid,child_birth=None);beat()
    try:
        birth=live.identity(p.pid)
        if os.getpgid(p.pid)!=p.pid or os.getsid(p.pid)!=p.pid:raise RuntimeError('child session identity mismatch')
        current.update(child_pid=p.pid,child_birth=birth);beat()
        p.stdin.write(b'1');p.stdin.flush();p.stdin.close()
        while p.poll() is None:
            if name!='recovering' and watchdog.poll() is not None:
                live.stop_owned_group(p.pid,birth,expected_parent=owner)
                raise RuntimeError('watchdog exited during active phase')
            if time.monotonic()>current['deadline']:
                live.stop_owned_group(p.pid,birth,expected_parent=owner)
                raise TimeoutError(name+' absolute deadline')
            beat();time.sleep(5)
        return p.returncode
    finally:
        if p.poll() is None:
            try:live.stop_owned_group(p.pid,live.identity(p.pid),expected_parent=owner)
            except Exception:pass
        # Do not discard identity while descendants may remain in the group.
        if live.group_absent(p.pid):
            current.update(child_pid=None,child_birth=None);beat()
        else:
            beat()
            raise RuntimeError('child group exit UNKNOWN; phase transition refused')


phase('watchdog_ready')
watchdog=subprocess.Popen([sys.executable,str(here/'engram_adapter_watchdog.py'),'--execute',run_id,str(owner)],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
ready=False
for _ in range(30):
    if watchdog.poll() is not None:break
    beat()
    if status.exists():
        x=json.loads(status.read_text())
        if x.get('state')=='READY' and x.get('binding_run_id')==run_id and x.get('pid')==watchdog.pid and time.time()-x.get('time',0)<15:
            ready=True;break
    time.sleep(1)
if not ready:raise RuntimeError('watchdog did not establish recovery lease; baseline was not parked')
try:
    if run_phase('starting','engram_adapter_start.py')!=0:raise RuntimeError('start failed')
    if run_phase('supervising','engram_adapter_supervise.py')!=0:raise RuntimeError('diagnostic failed')
finally:
    if current.get('child_pid') is not None:
        raise RuntimeError('child group exit UNKNOWN; watchdog retains identity and fence recovery responsibility')
    if run_phase('recovering','engram_adapter_recover.py')==0:
        done.write_text('owned baseline health verified\n')
    else:
        raise RuntimeError('baseline recovery not verified; independent watchdog must continue')
