"""Bounded native-router-only readiness endpoint repair; existing P/D untouched."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,signal,socket,subprocess,time,urllib.request
ROOT=Path('<TASK_ROOT>');R=Path(__file__).resolve().parent;SERVICE=ROOT/'sg_pd_g316';CID='0e8232575c7a4d9ad4065eda5d9682b9096876f030ddb1602c8c42dfdd30f5dc'
assert socket.gethostname()=='<ROLE_NODE>' and R==ROOT/'router_health_ready_g553'
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
 if 'sglang_router.launch_router' in a and '--port' in a and a[a.index('--port')+1]=='18080':rows.append({'pid':int(p.parent.name),'argv':a})
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
 oldcmd=Path('/proc/958636/cmdline').read_bytes();assert b'sglang_router.launch_router' in oldcmd and b'18080' in oldcmd
 oldargv=old['argv'];assert oldargv[:3]==['flock','--nonblock','<COORD_ROOT>/<ROLE_NODE>-port18080.lock'];assert '--health-check-endpoint' not in oldargv
 for base in ['http://<HOST>:19083','http://<HOST>:19084']:
  assert http(base+'/ready')['status']==200
 workers=json.loads(http('http://<HOST>:18080/workers')['body']);assert len(workers['workers'])==2 and all(w['load']==0 and w['is_healthy'] for w in workers['workers'])
 time.sleep(2);workers2=json.loads(http('http://<HOST>:18080/workers')['body']);assert all(w['load']==0 and w['is_healthy'] for w in workers2['workers'])
 save('before_workers.json',workers2);save('old_router_handle.json',old)
 rows=namespace_router();assert len(rows)==1;save('old_router_namespace.json',rows)
 assert rows[0]['argv'][-1]=='--disable-retries';assert '--health-check-endpoint' not in rows[0]['argv']
 state('stopping_exact_owned_old_router',namespace_pid=rows[0]['pid'])
 namespace_router('stop',rows[0]['pid']);deadline=time.monotonic()+40
 while namespace_router():
  assert time.monotonic()<deadline,'Oldrouter did not exit, no replacement';time.sleep(1)
 oldproc=Path('/proc/958636/cmdline');assert not oldproc.exists() or not oldproc.read_bytes()
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
 NS_PID=rows[0]['pid'];assert rows[0]['argv'][-2:]==['--health-check-endpoint','/ready'];save('new_router_namespace.json',rows);save('after_workers.json',w)
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
