"""One-shot detached controller. After a crash, inspect before manual recovery."""
from pathlib import Path
import json, os, subprocess, sys, time, uuid

if __name__ != '__main__' or sys.argv[1:] != ['--execute']:
    print('Review only: --execute starts the second isolated diagnostic controller.')
    raise SystemExit(0)
here=Path(__file__).resolve().parent
handle=uuid.uuid4().hex
reservation=here/'controller-reservation.json'
stdout=here/('controller-'+handle+'.stdout')
stderr=here/('controller-'+handle+'.stderr')
state={'handle':handle,'created_unix':time.time(),'state':'STARTING_VERIFY_BEFORE_MANUAL_RETRY','stdout':str(stdout),'stderr':str(stderr),'pid':None}
# A crash before or immediately after Popen leaves the exclusive reservation;
# never infer that a missing PID means that no controller survived.
try:
    with reservation.open('x') as f:
        json.dump(state,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
except FileExistsError:
    raise SystemExit('Controller reservation already exists. Inspect its PID, logs and recovery state; no automatic restart.')
with stdout.open('x') as out, stderr.open('x') as err:
    child=subprocess.Popen([sys.executable,str(here/'orchestrate_binding.py'),'--execute'],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,close_fds=True)
state.update(pid=child.pid,state='CONTROLLER_STARTED_NOT_DIAGNOSTIC_SUCCESS')
tmp=here/('controller-reservation-'+handle+'.tmp')
with tmp.open('x') as f:
    json.dump(state,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,reservation)
print(json.dumps(state,indent=2))
