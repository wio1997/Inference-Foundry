"""Real local synthetic processes only; no SSH, container, NPU or real recovery."""
import ast,hashlib,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
import engram_liveness as live
if sys.flags.optimize:raise RuntimeError('assertions must be enabled')
HERE=Path(__file__).resolve().parent
RUN='c'*32

def wait(pred, timeout=15):
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  value=pred()
  if value:return value
  time.sleep(.05)
 raise RuntimeError('timed out')

def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return None

with tempfile.TemporaryDirectory(prefix='synthetic-liveness-',dir=HERE) as td:
 d=Path(td);source=(HERE/'watchdog_heartbeat.template').read_text();tree=ast.parse(source)
 # Copy the real read_progress and main loop, replacing lease and recovery only.
 funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('save','read_progress')]
 main=next(x for x in tree.body if isinstance(x,ast.Try))
 module=ast.Module(body=funcs+[main],type_ignores=[]);ast.fix_missing_locations(module)
 body=ast.unparse(module).replace('time.sleep(6)','time.sleep(0.1)').replace('time.sleep(30)','time.sleep(0.1)')
 stub="""import json,os,subprocess,sys,time
from pathlib import Path
import engram_liveness as live
live.MAX_STALENESS=0.6
run_id=sys.argv[1];owner=int(sys.argv[2]);here=Path(__file__).resolve().parent
status=here/('watchdog-'+run_id+'.json');done=here/('watchdog-'+run_id+'.done');heartbeat=here/('heartbeat-'+run_id+'.json')
def lease(h): pass
"""
 (d/'engram_adapter_watchdog.py').write_text(stub+body+'\n')
 (d/'engram_adapter_recover.py').write_text("from pathlib import Path\nPath(__file__).with_name('fake-recovery-ran').write_text('FAKE ONLY')\n")
 ownercode="""import json,os,time,sys
from pathlib import Path
import engram_liveness as live
p=Path(sys.argv[1]);run=sys.argv[2];pid=os.getpid();start=time.monotonic();birth=live.identity(pid)
while True:
 h={'run_id':run,'owner_pid':pid,'owner_birth':birth,'child_pid':None,'child_birth':None,'phase':'supervising','phase_start':start,'updated':time.monotonic(),'deadline':start+live.PHASE_LIMITS['supervising']}
 t=p.with_suffix('.tmp');t.write_text(json.dumps(h));os.replace(t,p);time.sleep(.1)
"""
 env=dict(os.environ);env.pop('PYTHONOPTIMIZE',None);env['PYTHONPATH']=str(HERE)
 owner=watchdog=unrelated=None
 try:
  unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True,env=env)
  birth=wait(lambda:live.identity(unrelated.pid));assert os.getpgid(unrelated.pid)==unrelated.pid and os.getsid(unrelated.pid)==unrelated.pid
  wrong=birth.copy();wrong[1]+=1
  assert not live.signal_owned_group(unrelated.pid,wrong,signal.SIGTERM)
  assert unrelated.poll() is None
  assert not live.signal_owned_group(unrelated.pid,birth,signal.SIGTERM,expected_parent=os.getpid()+1000000)
  assert unrelated.poll() is None
  unrelated.terminate();unrelated.wait(timeout=3)
  try:ok=live.signal_owned_group(unrelated.pid,birth,signal.SIGTERM)
  except (ProcessLookupError,RuntimeError):ok=False
  assert not ok
  hb=d/('heartbeat-'+RUN+'.json')
  owner=subprocess.Popen([sys.executable,'-c',ownercode,str(hb),RUN],start_new_session=True,env=env)
  h=wait(lambda:read(hb));assert h['owner_birth']==live.identity(owner.pid) and live.check_heartbeat(h,RUN,time.monotonic())=='FRESH'
  watchdog=subprocess.Popen([sys.executable,str(d/'engram_adapter_watchdog.py'),RUN,str(owner.pid)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  status=d/('watchdog-'+RUN+'.json');wait(lambda:(lambda x:x and x.get('state')=='READY')(read(status)))
  os.kill(owner.pid,signal.SIGSTOP)
  wait(lambda:(d/'fake-recovery-ran').exists(),timeout=12)
  assert watchdog.wait(timeout=3)==0 and owner.wait(timeout=3)==-signal.SIGTERM
  result={'result':'PASS','template_sha256':hashlib.sha256(source.encode()).hexdigest(),'real_libproc_identity':'PASS','wrong_birth_and_parent_do_not_signal':'PASS','exited_process_refused':'PASS','sigstop_stale_causes_only_authenticated_owner_termination':'PASS','recovery':'fake-only','lease':'fake-only','changes_to_template':'copied AST save/read_progress/main loop; timing shortened; recovery and lease replaced; disable gate intentionally removed in private temporary copy','no_ssh_container_npu':True}
  (HERE/'real-liveness-cpu-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:
  for p in (watchdog,owner,unrelated):
   if p is not None and p.poll() is None:
    # All processes here were created by this test and their Popen identities are held.
    p.terminate()
    try:p.wait(timeout=3)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=3)
