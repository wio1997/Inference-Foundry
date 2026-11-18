"""Bounded native-router-only readiness endpoint repair; existing P/D untouched."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,signal,socket,subprocess,time,urllib.request
ROOT=Path('<TASK_ROOT>');R=Path(__file__).resolve().parent;SERVICE=ROOT/'sg_pd_g316';CID='0e8232575c7a4d9ad4065eda5d9682b9096876f030ddb1602c8c42dfdd30f5dc'
assert socket.gethostname()=='<ROLE_NODE>' and R==ROOT/'router_health_ready_g565'
os.chdir(R);START=time.monotonic();CHILD=None;NS_PID=None;PORTLOCK=None;FAILURE=None

def save(name,d):
 p=R/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def state(phase,**kw):
 d={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-START,**kw};save('state.json',d);print(json.dumps(d),flush=True)
def http(url):
 with urllib.request.urlopen(url,timeout=8) as f:return {'status':f.status,'body':f.read().decode()}
def command(args,timeout=30):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,cwd=R);assert p.returncode==0,(args[:3],p.stderr);return p.stdout

def namespace_router(action='inspect',expected=None):
 code="""from pathlib import Path
import json,os,signal
rows=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:a=p.read_bytes().split(b'\\0');a=[s.decode() for s in a if s]
 except (OSError,UnicodeDecodeError):continue
 if a==['sglang::router']:
  inodes=set()
  for table in ['tcp','tcp6']:
   for line in Path('/proc/net',table).read_text().splitlines()[1:]:
    fields=line.split()
    if int(fields[1].rsplit(':',1)[1],16)==18080 and fields[3]=='0A':inodes.add(fields[9])
  owned=[]
  for fd in p.parent.joinpath('fd').iterdir():
   try:target=os.readlink(fd)
   except OSError:continue
   if target.startswith('socket:[') and target[8:-1] in inodes:owned.append(target[8:-1])
  if owned:rows.append({'pid':int(p.parent.name),'argv':a,'listen_port':18080,'socket_inodes':owned})
print(json.dumps(rows),flush=True)
"""
 if action=='stop':
  code+='\nassert len(rows)==1 and rows[0]["pid"]=='+repr(expected)+'\nos.kill(rows[0]["pid"],signal.SIGTERM)\n'
 return json.loads(command(['docker','exec',CID,'python3','-c',code]))

try:
 state('preflight')
 a=json.loads(command(['docker','inspect',CID]))[0];assert a['State']['Running'] and a['Config']['Labels']['codex.run']=='sg_pd_g316' and a['Image']=='sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'
 service=json.loads((SERVICE/'state.json').read_text());assert service['pid']==936231 and service['container_id']==CID and str(SERVICE/'node_controller.py').encode() in Path('/proc/936231/cmdline').read_bytes()
 proof=json.loads((R/'first_cold_failure_parent.json').read_text());assert proof['first256cold_total']==32 and proof['failed']==14 and proof['hot_submitted'] is False and proof['first_cold_replay'] is False
 old=json.loads((SERVICE/'router_handle.json').read_text());assert old['pid']==958636 and old['container_id']==CID
 prior=ROOT/'router_health_ready_g560'
 previous=json.loads((prior/'state.json').read_text());assert previous['phase']=='failed_requires_review' and previous['pid']==1027953
 assert "oldproc=Path('/proc/958636/cmdline')" in (prior/'controller.log').read_text()
 oldrows=json.loads((prior/'old_router_namespace.json').read_text());assert len(oldrows)==1 and oldrows[0]['pid']==20823 and oldrows[0]['listen_port']==18080
 proc=Path('/proc/1027953/cmdline');assert not proc.exists() or not proc.read_bytes() or str(prior/'controller.py').encode() not in proc.read_bytes()
 oldproc=Path('/proc/958636/cmdline');deadline=time.monotonic()+15
 while oldproc.exists() and oldproc.read_bytes():
  assert time.monotonic()<deadline,'Old owned dockerexec still active; do not launch';time.sleep(.5)
 assert not namespace_router(),'An actual listen-socket owner remains; preserve it'
 with socket.socket() as sock:sock.bind(('<HOST>',18080))
 for base in ['http://<HOST>:19083','http://<HOST>:19084']:
  assert http(base+'/ready')['status']==200
 save('recovery_previous_state.json',previous);save('old_router_namespace.json',oldrows);save('old_router_handle.json',old)
 oldargv=old['argv'];assert oldargv[:3]==['flock','--nonblock','<COORD_ROOT>/<ROLE_NODE>-port18080.lock']
 state('verified_exact_old_router_stopped_and_port_free_before_new_start')
 PORTLOCK=Path('<COORD_ROOT>/<ROLE_NODE>-port18080.lock').open('a');fcntl.flock(PORTLOCK,fcntl.LOCK_EX|fcntl.LOCK_NB)
 argv=oldargv[3:];argv[argv.index('python3'):]=['timeout','--signal=TERM','--kill-after=30','43200',*argv[argv.index('python3'):]];argv+=['--health-check-endpoint','/ready']
 save('new_router_command.json',{'argv':argv,'only_router_policy_delta':['--health-check-endpoint','/ready'],'P_D_service_ids_unchanged':True,'requests_unchanged':False,'original_failed_first_untouched':True,'scope':'Availability policy repair only; no repaired long-load acceptance or performance gain'})
 with (R/'router.log').open('x') as log:CHILD=subprocess.Popen(argv,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 deadline=time.monotonic()+120
 while True:
  assert CHILD.poll() is None,'New router exited'
  try:
   rows=namespace_router();w=json.loads(http('http://<HOST>:18080/workers')['body'])
   if len(rows)==1 and len(w['workers'])==2 and all(x['is_healthy'] and x['load']==0 for x in w['workers']):break
  except Exception:
   if time.monotonic()>deadline:raise
  assert time.monotonic()<deadline,'New router registration bound';time.sleep(2)
 NS_PID=rows[0]['pid'];assert rows[0]['argv']==['sglang::router'] and rows[0]['listen_port']==18080;save('new_router_namespace.json',rows);save('after_workers.json',w)
 state('router_ready_long_load_UNVERIFIED',namespace_pid=NS_PID,docker_exec_pid=CHILD.pid,endpoint='/ready',bound_s=43200,P_D_service_ids_unchanged=True)
 while CHILD.poll() is None:
  if (R/'stop.request').exists():break
  assert time.monotonic()-START<43360,'Router controller bound';time.sleep(5)
 if CHILD.poll() not in [None,0]:raise RuntimeError('Router exited '+str(CHILD.returncode))
except BaseException as exc:
 FAILURE=repr(exc);state('failed_requires_review',error=FAILURE);raise
finally:
 if CHILD is not None:
  rows=namespace_router()
  if rows:
   assert NS_PID is not None and len(rows)==1 and rows[0]['pid']==NS_PID
   namespace_router('stop',NS_PID)
  CHILD.wait(timeout=45)
  assert not namespace_router();state('router_stopped_port_released',error=FAILURE)
 if PORTLOCK is not None:PORTLOCK.close()
