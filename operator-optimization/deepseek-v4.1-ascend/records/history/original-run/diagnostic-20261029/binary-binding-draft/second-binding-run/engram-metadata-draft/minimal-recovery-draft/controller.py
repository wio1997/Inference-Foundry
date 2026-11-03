"""Disabled minimal controller; reviewer must validate before enabling."""
import os,sys,subprocess,time,uuid,json,fcntl
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('optimized Python forbidden')
if __name__!='__main__' or sys.argv[1:]!=['--execute']:
    print('Review draft only');raise SystemExit(0)
raise SystemExit('Diagnostic complete; re-review required before execution')
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent))
import engram_liveness as live
fd=os.open(here/'controller.lock',os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
run_id=uuid.uuid4().hex
owner=os.getpid();birth=live.identity(owner)
env=dict(os.environ,WIO_BINDING_RUN_ID=run_id)
status=here/('watchdog-'+run_id+'.json');done=here/('done-'+run_id)
watchdog=subprocess.Popen([sys.executable,str(here/'watchdog.py'),'--execute',run_id,str(owner),json.dumps(birth)],env=env,start_new_session=True)
ready=False
for _ in range(30):
    if watchdog.poll() is not None:break
    try:
        s=json.loads(status.read_text())
        if s.get('state')=='READY' and s.get('run_id')==run_id and s.get('pid')==watchdog.pid and time.time()-s.get('time',0)<15:ready=True;break
    except (OSError,ValueError):pass
    time.sleep(1)
if not ready:raise RuntimeError('recovery watchdog unavailable; no baseline stop')
try:
    subprocess.run([sys.executable,str(here/'engram_adapter_start.py'),'--execute'],env=env,check=True)
    subprocess.run([sys.executable,str(here/'engram_adapter_supervise.py'),'--execute'],env=env,check=True)
finally:
    r=subprocess.run([sys.executable,str(here/'engram_adapter_recover.py'),'--execute'],env=env)
    if r.returncode==0:done.write_text('baseline health verified\n')
    else:raise RuntimeError('baseline health not confirmed; independent watchdog will retry')
