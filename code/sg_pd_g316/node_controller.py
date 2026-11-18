"""Canonical eight-NPU flock/owner lifecycle, one bounded standalone diagnostic.

Derived from task node_controller.py; never starts on the local analysis client.
Run only by root on <ROLE_NODE> after deploying all three entry files to RUN.
"""
import datetime,fcntl,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path

TASK_ID='glm53_pd_align_910b_20261116';HOST=socket.gethostname();ROLE={'<ROLE_NODE>':'p','<ROLE_NODE>':'d'}[HOST]
ROOT=Path('/data' if ROLE=='p' else '/data1')/'tiankuan/yxy'/TASK_ID;RUN=ROOT/'sg_pd_g316'
COORD=Path('<COORD_ROOT>');NAME='yxy-glm53-sg-'+ROLE+'-g316'
IMAGE='sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'
MODEL=Path('/data/models/GLM-5.3-Flash-w8a8');PORT=19083 if ROLE=='p' else 19084;LIFETIME=72000
LOCKS=[];OWNERS=[];CLAIMED=[];CHILD=None;CREATE_ATTEMPTED=False;FAILURE=None


def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)

def cmd(args,timeout=30):return subprocess.run(args,capture_output=True,text=True,timeout=timeout,cwd=ROOT)

def state(phase,**extra):
 d=dict(task_id=TASK_ID,host=HOST,role=ROLE,generation=316,pid=os.getpid(),phase=phase,
        container=NAME,image_id=IMAGE,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
 atomic(RUN/'state.json',d);print(json.dumps(d),flush=True)

def inspect():
 r=cmd(['docker','inspect',NAME])
 if r.returncode:
  if 'No such' in r.stderr:return None
  raise RuntimeError('Docker inspect failed: '+r.stderr)
 return json.loads(r.stdout)[0]

def ours(info):
 labels=info.get('Config',{}).get('Labels') or {}
 return (labels.get('codex.task')==TASK_ID and labels.get('codex.controller_pid')==str(os.getpid())
         and labels.get('codex.run')==RUN.name and info.get('Image')==IMAGE)

def interrupted(sig,frame):raise RuntimeError('Controller received signal '+str(sig))

def main():
 global CHILD,CREATE_ATTEMPTED,FAILURE
 # Reconcile an existing run externally; never overwrite its authoritative state.
 if HOST not in ['<ROLE_NODE>','<ROLE_NODE>'] or os.geteuid()!=0:raise RuntimeError('Requires allocated GLM host root')
 if Path(__file__).resolve().parent!=RUN:raise RuntimeError('Deploy controller and entry files to '+str(RUN))
 if (RUN/'state.json').exists():raise RuntimeError('Existing run requires explicit recovery; state preserved')
 for name in ['run_pd.sh','launch_pd.py','native_readiness.py','candidate.json','glm53_native_gate/__init__.py','glm53_native_gate/entry.py']:
  if not (RUN/name).is_file():raise RuntimeError('Missing entry '+name)
 if not MODEL.is_dir() or not (ROOT/'runtime_sglang_g32/integrated_g18_manifest.json').is_file():raise RuntimeError('Missing read-only model/integrated G18 runtime')
 RUN.mkdir(parents=True,exist_ok=True);COORD.mkdir(parents=True,exist_ok=True)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  assert inspect() is None,'Container name already exists'
  for device in range(8):
   key=HOST+'-npu'+str(device);f=(COORD/(key+'.lock')).open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except Exception:f.close();raise
   LOCKS.append(f);p=COORD/(key+'.owner.json')
   if p.exists():assert json.loads(p.read_text()).get('state')=='released','Existing ownership requires reconciliation: '+str(p)
   OWNERS.append(p)
  assert len(LOCKS)==len(OWNERS)==8
  npu=cmd(['npu-smi','info']);(RUN/'npu_before.txt').write_text(npu.stdout+npu.stderr)
  assert len(re.findall(r'\|\s*\d+\s+910B1\s*\|\s*OK\s*\|',npu.stdout))==8,'All8 health must be OK'
  assert npu.returncode==0 and all('No running processes found in NPU '+str(i) in npu.stdout for i in range(8)),'NPU occupancy changed; reconcile first'
  for port in [PORT]+([8998,8678] if ROLE=='p' else []):
   f=(COORD/(HOST+'-port'+str(port)+'.lock')).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);LOCKS.append(f)
   with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('<HOST>',port))
  ip=cmd(['ip','-j','-4','addr','show','dev','enp67s0f0']);assert ip.returncode==0
  expected_ip='<HOST>' if ROLE=='p' else '<HOST>';assert expected_ip in ip.stdout
  atomic(RUN/'nic_before.json',json.loads(ip.stdout))
  deadline=time.monotonic()+LIFETIME
  # All eight flock locks are held before the first ownership claim.
  for p in OWNERS:
   prior=json.loads(p.read_text()) if p.exists() else {}
   d=dict(task_id=TASK_ID,host=HOST,generation=int(prior.get('generation',0))+1,state='claimed',pid=os.getpid(),run_directory=str(RUN))
   atomic(p,d);CLAIMED.append((p,d['generation']))
  state('claimed',physical_npus=list(range(8)),api_port=PORT,lifetime_seconds=LIFETIME)
  if (RUN/'stop.request').exists():raise RuntimeError('Stopped before container start')
  for d in ['tmp','home','shm','cache','profiles','logs','weight_ledgers']:(RUN/d).mkdir(exist_ok=True)
  args=['docker','run','-d','--name',NAME,'--label','codex.task='+TASK_ID,'--label','codex.controller_pid='+str(os.getpid()),
        '--label','codex.run='+RUN.name,'--network','host','--log-driver','none','--cap-add','SYS_PTRACE',
        '--security-opt','seccomp=unconfined','--read-only','-v',str(ROOT)+':/task:ro','-v',str(RUN)+':/task/'+RUN.name,
        '-v',str(ROOT/'runtime_sglang_g32')+':/task/runtime_sglang_g32:ro',
        '-v',str(MODEL)+':/models/glm53:ro','-v',str(RUN/'tmp')+':/tmp',
        '-v',str(RUN/'home')+':/root','-v',str(RUN/'shm')+':/dev/shm','-w','/task/'+RUN.name]
  for p in ['/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/etc/ascend_install.info','/etc/hccn.conf','/usr/local/sbin/npu-smi']:
   if Path(p).exists():args+=['-v',p+':'+p+':ro']
  for p in [*['/dev/davinci'+str(i) for i in range(8)],'/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc','/dev/infiniband']:
   if Path(p).exists():args+=['--device',p]
  env={'ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','HOME':'/task/'+RUN.name+'/home',
       'TMPDIR':'/task/'+RUN.name+'/tmp','HF_HOME':'/task/'+RUN.name+'/cache/hf',
       'TORCH_EXTENSIONS_DIR':'/task/'+RUN.name+'/cache/torch_extensions','TRITON_CACHE_DIR':'/task/'+RUN.name+'/cache/triton',
       'ASCEND_CACHE_PATH':'/task/'+RUN.name+'/cache/ascend','PYTHONPYCACHEPREFIX':'/task/'+RUN.name+'/cache/pycache',
       'ASCEND_WORK_PATH':'/task/'+RUN.name+'/cache/ascend_work','ASCEND_PROCESS_LOG_PATH':'/task/'+RUN.name+'/logs/ascend'}
  for value in env.values():
   if value.startswith('/task/'):
    directory=ROOT/value.removeprefix('/task/')
    assert RUN in directory.parents or directory==RUN,'Runtime writes must remain under current run'
    directory.mkdir(parents=True,exist_ok=True)
  for k,v in env.items():args+=['-e',k+'='+v]
  args+=['--entrypoint','bash',IMAGE,'-c','exec sleep 72100']
  atomic(RUN/'container_argv.json',args);CREATE_ATTEMPTED=True
  r=cmd(args,90);atomic(RUN/'container_start.json',dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr))
  assert r.returncode==0,'Container start failed: '+r.stderr
  info=inspect();assert info is not None and ours(info),'Container identity changed'
  cache_probe='import json,os;from pathlib import Path;keys='+repr(list(env))+';checks={k:dict(path=os.environ[k],exists=Path(os.environ[k]).is_dir(),writable=os.access(os.environ[k],os.W_OK)) for k in keys if os.environ[k].startswith("/task/")};print(json.dumps(checks));assert all(v["exists"] and v["writable"] for v in checks.values())'
  probe=cmd(['docker','exec',NAME,'python3','-c',cache_probe],30)
  atomic(RUN/'runtime_directory_preflight.json',dict(rc=probe.returncode,stdout=probe.stdout,stderr=probe.stderr))
  assert probe.returncode==0,'Cache directory preflight failed before model loading: '+probe.stderr
  with (RUN/'server.log').open('x') as log:
   child_args=['docker','exec',NAME,'bash','/task/'+RUN.name+'/run_pd.sh']
   CHILD=subprocess.Popen(child_args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
   state('model_starting',docker_exec_pid=CHILD.pid,container_id=info['Id'],lifetime_seconds=LIFETIME,command=child_args)
   while CHILD.poll() is None:
    if (RUN/'stop.request').exists():raise RuntimeError('Stop requested')
    if time.monotonic()>deadline:raise TimeoutError('72000s PD lifecycle bound')
    time.sleep(5)
   if CHILD.returncode:raise RuntimeError('Serving process exited '+str(CHILD.returncode))
 except Exception as exc:
  FAILURE=str(exc);state('failed',error=FAILURE)
 finally:
  stopped=True;cleanup={}
  if CREATE_ATTEMPTED:
   try:
    info=inspect()
    if info is not None:
     assert ours(info),'Container labels/image mismatch; refusing cleanup'
     r=cmd(['docker','stop','--time','15',NAME],45);cleanup['stop_rc']=r.returncode
     info=inspect();stopped=info is not None and ours(info) and not info['State']['Running'] and info['State']['Pid']==0
     cleanup['container_state']=info['State'] if info else None
   except Exception as exc:stopped=False;cleanup['error']=repr(exc)
  if CHILD is not None and stopped:
   try:CHILD.wait(timeout=15)
   except subprocess.TimeoutExpired:CHILD.terminate();CHILD.wait(timeout=10)
  if stopped and CLAIMED:
   after=cmd(['npu-smi','info']);(RUN/'npu_after.txt').write_text(after.stdout+after.stderr)
   cleanup['all8_process_lists_clear']=after.returncode==0 and all('No running processes found in NPU '+str(i) in after.stdout for i in range(8))
   stopped=cleanup['all8_process_lists_clear']
  if stopped:
   for p,generation in CLAIMED:
    d=json.loads(p.read_text())
    if d.get('task_id')!=TASK_ID or d.get('pid')!=os.getpid() or d.get('generation')!=generation or d.get('run_directory')!=str(RUN):
     stopped=False;cleanup['owner_error']='Owner identity changed: '+str(p);break
   if stopped:
    for p,generation in CLAIMED:
     d=json.loads(p.read_text());d['state']='released';atomic(p,d)
  state(('failed_released' if FAILURE else 'stopped_released') if stopped else 'cleanup_unverified_ownership_retained',failure=FAILURE,cleanup=cleanup)
  for f in LOCKS:f.close()
 return 1 if FAILURE else 0

if __name__=='__main__':sys.exit(main())
