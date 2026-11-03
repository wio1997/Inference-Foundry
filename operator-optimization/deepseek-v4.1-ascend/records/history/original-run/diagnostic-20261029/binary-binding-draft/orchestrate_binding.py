"""Start the recovery watchdog before the owned diagnostic; explicit opt-in only."""
from pathlib import Path
import json,os,subprocess,sys,time,uuid

if __name__!='__main__' or sys.argv[1:]!=['--execute']:
    print('Review draft: no operations. --execute performs the isolated diagnostic.');raise SystemExit(0)
here=Path(__file__).resolve().parent
run_id=uuid.uuid4().hex
status=here/('watchdog-'+run_id+'.json');done=here/('watchdog-'+run_id+'.done')
env=dict(os.environ);env['WIO_BINDING_RUN_ID']=run_id
watchdog=subprocess.Popen([sys.executable,str(here/'watchdog_binding.py'),'--execute',run_id,str(os.getpid())],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
ready=False
for _ in range(30):
    if watchdog.poll() is not None:break
    if status.exists():
        x=json.loads(status.read_text())
        if x.get('state')=='READY' and x.get('binding_run_id')==run_id and x.get('pid')==watchdog.pid and time.time()-x.get('time',0)<15:
            ready=True;break
    time.sleep(1)
if not ready:raise RuntimeError('Watchdog did not establish recovery lease; baseline was not parked')
print('Recovery watchdog ready; running owned-container preflight.',flush=True)
try:
    r=subprocess.run([sys.executable,str(here/'start_binding_recapture.py'),'--execute'],env=env)
    if r.returncode:raise RuntimeError('Start failed; recovery will be attempted')
    r=subprocess.run([sys.executable,str(here/'supervise_binding_recapture.py'),'--execute'],env=env)
    if r.returncode:raise RuntimeError('Diagnostic or supervisor failed; recovery will be attempted')
finally:
    r=subprocess.run([sys.executable,str(here/'recover_binding.py'),'--execute'],env=env)
    if r.returncode==0:
        done.write_text('owned baseline health verified\n')
        print('Baseline health verified; watchdog can exit.',flush=True)
    else:
        print('Recovery did not complete; independent watchdog will retry while safe.',flush=True)
        raise RuntimeError('Baseline health was not verified; diagnostic is not successful')
