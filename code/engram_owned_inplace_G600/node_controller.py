"""Canonical eight-NPU flock/owner lifecycle, one bounded standalone diagnostic.

Derived from task node_controller.py; never starts on the local analysis client.
Run only by root on <ROLE_HOSTNAME> after deploying all three entry files to RUN.
"""
import datetime,fcntl,json,os,re,signal,socket,subprocess,sys,time
from pathlib import Path

TASK_ID='deepseek_v41_w4a8_pd_910b_20261117';HOST=socket.gethostname()
ROOT=Path('/PRIVATE_WORK_ROOT')/TASK_ID;RUN=ROOT/'sg_engram_owned_inplace_g600'
COORD=Path('/root/.codex/coord');NAME='yxy-ds41-sg-engram-owned-inplace-g600-'+HOST[-4:]
IMAGE='sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'
LIFETIME=1200
LOCKS=[];OWNERS=[];CLAIMED=[];CHILD=None;CREATE_ATTEMPTED=False;FAILURE=None


def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)

def cmd(args,timeout=30):return subprocess.run(args,capture_output=True,text=True,timeout=timeout,cwd=ROOT)

def state(phase,**extra):
 d=dict(task_id=TASK_ID,host=HOST,role='SG_native_lowratio_target_draft_pool',generation=600,pid=os.getpid(),phase=phase,
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
 if HOST!='<ROLE_HOSTNAME>' or os.geteuid()!=0:raise RuntimeError('Requires root on <ROLE_HOSTNAME>')
 if Path(__file__).resolve().parent!=RUN:raise RuntimeError('Deploy controller and entry files to '+str(RUN))
 if (RUN/'state.json').exists():raise RuntimeError('Existing run requires explicit recovery; state preserved')
 for name in ['run_probe.sh','probe.py']:
  if not (RUN/name).is_file():raise RuntimeError('Missing entry '+name)

 RUN.mkdir(parents=True,exist_ok=True);COORD.mkdir(parents=True,exist_ok=True)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  assert inspect() is None,'Container name already exists'
  port_lock=(COORD/(HOST+'-port32600.lock')).open('a');fcntl.flock(port_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);LOCKS.append(port_lock)
  import socket as _socket
  with _socket.socket() as test_socket:test_socket.bind(('127.0.0.1',32600))
  for device in range(8):
   key=HOST+'-npu'+str(device);f=(COORD/(key+'.lock')).open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except Exception:f.close();raise
   LOCKS.append(f);p=COORD/(key+'.owner.json')
   if p.exists():assert json.loads(p.read_text()).get('state')=='released','Existing ownership requires reconciliation: '+str(p)
   OWNERS.append(p)
  assert len(LOCKS)==9 and len(OWNERS)==8
  npu=cmd(['npu-smi','info']);(RUN/'npu_before.txt').write_text(npu.stdout+npu.stderr)
  assert len(re.findall(r'\|\s*\d+\s+910B1\s*\|\s*OK\s*\|',npu.stdout))==8,'NPU health must be8/8 OK after prior isolated fault'
  assert npu.returncode==0 and all('No running processes found in NPU '+str(i) in npu.stdout for i in range(8)),'NPU occupancy changed; reconcile first'

  deadline=time.monotonic()+LIFETIME
  # All requested flock locks are held before the first ownership claim.
  for p in OWNERS:
   prior=json.loads(p.read_text()) if p.exists() else {}
   d=dict(task_id=TASK_ID,host=HOST,generation=int(prior.get('generation',0))+1,state='claimed',pid=os.getpid(),run_directory=str(RUN))
   atomic(p,d);CLAIMED.append((p,d['generation']))
  state('claimed',physical_npus=list(range(8)),scope='Bounded256row Engram eager exactBF16bitpatterns and peak gate only; no model/PD/performance claim',lifetime_seconds=LIFETIME)
  if (RUN/'stop.request').exists():raise RuntimeError('Stopped before container start')
  for d in ['tmp','home','shm','cache','profiles','logs','weight_ledgers']:(RUN/d).mkdir(exist_ok=True)
  args=['docker','run','-d','--cpuset-cpus','168-175','--name',NAME,'--label','codex.task='+TASK_ID,'--label','codex.controller_pid='+str(os.getpid()),
        '--label','codex.run='+RUN.name,'--network','host','--read-only','--cpus','8','--memory','64g','--log-driver','none','-v',str(ROOT)+':/task:ro','-v',str(RUN)+':/task/'+RUN.name,
        '-v','/PRIVATE_TASK_ROOT/models/DeepSeek-V4.1-Flash-1017-w4a8:/models/ds41:ro','-v',str(RUN/'tmp')+':/tmp',
        '-v',str(RUN/'home')+':/root','-v',str(RUN/'shm')+':/dev/shm','-w','/task/'+RUN.name]
  for p in ['/usr/local/Ascend/driver','/usr/local/Ascend/firmware','/etc/ascend_install.info','/etc/hccn.conf','/usr/local/sbin/npu-smi']:
   if Path(p).exists():args+=['-v',p+':'+p+':ro']
  for p in [*['/dev/davinci'+str(i) for i in range(8)],'/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
   if Path(p).exists():args+=['--device',p]
  env={'ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','HOME':'/task/'+RUN.name+'/home',
       'TMPDIR':'/task/'+RUN.name+'/tmp','HF_HOME':'/task/'+RUN.name+'/cache/hf',
       'TORCH_EXTENSIONS_DIR':'/task/'+RUN.name+'/cache/torch_extensions','TRITON_CACHE_DIR':'/task/'+RUN.name+'/cache/triton',
       'ASCEND_CACHE_PATH':'/task/'+RUN.name+'/cache/ascend','PYTHONPYCACHEPREFIX':'/task/'+RUN.name+'/cache/pycache',
       'ASCEND_WORK_PATH':'/task/'+RUN.name+'/cache/ascend_work','ASCEND_PROCESS_LOG_PATH':'/task/'+RUN.name+'/logs/ascend'}
  env.update(OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',GLOO_SOCKET_IFNAME='enp67s0f0',HCCL_SOCKET_IFNAME='enp67s0f0',HCCL_BUFFSIZE='200',HCCL_CONNECT_TIMEOUT='180',HCCL_EXEC_TIMEOUT='180')
  for value in env.values():
   if value.startswith('/task/'):(ROOT/value.removeprefix('/task/')).mkdir(parents=True,exist_ok=True)
  for k,v in env.items():args+=['-e',k+'='+v]
  args+=['--entrypoint','bash',IMAGE,'-c','exec sleep 1400']
  atomic(RUN/'container_argv.json',args);CREATE_ATTEMPTED=True
  r=cmd(args,90);atomic(RUN/'container_start.json',dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr))
  assert r.returncode==0,'Container start failed: '+r.stderr
  info=inspect();assert info is not None and ours(info),'Container identity changed'
  with (RUN/'server.log').open('x') as log:
   child_args=['docker','exec',NAME,'bash','/task/'+RUN.name+'/run_probe.sh']
   CHILD=subprocess.Popen(child_args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
   state('component_running',docker_exec_pid=CHILD.pid,container_id=info['Id'],lifetime_seconds=LIFETIME,command=child_args)
   while CHILD.poll() is None:
    if (RUN/'stop.request').exists():raise RuntimeError('Stop requested')
    if time.monotonic()>deadline:raise TimeoutError('1200s component lifecycle bound')
    time.sleep(5)
   if CHILD.returncode:raise RuntimeError('Component process exited '+str(CHILD.returncode))
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
   after=cmd(['npu-smi','info']);(RUN/'npu_after.txt').write_text(after.stdout+after.stderr);cleanup['all8_process_lists_clear']=after.returncode==0 and all('No running processes found in NPU '+str(i) in after.stdout for i in range(8));stopped=cleanup['all8_process_lists_clear']
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
