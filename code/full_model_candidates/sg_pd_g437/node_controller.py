"""Canonical eight-NPU flock/owner lifecycle, one bounded standalone diagnostic.

Derived from task node_controller.py; never starts on the local analysis client.
Run only by root on <ROLE_HOSTNAME> after deploying all three entry files to RUN.
"""
import datetime,fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path

TASK_ID='deepseek_v41_w4a8_pd_910b_20261117';HOST=socket.gethostname();ROLE={'<ROLE_HOSTNAME>':'p','<ROLE_HOSTNAME>':'d'}[HOST]
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent
PREVIOUS_ROOT=Path('/PRIVATE_WORK_ROOT')/TASK_ID
COORD=Path('/root/.codex/coord');NAME='yxy-ds41-sg-'+ROLE+'-g437'
IMAGE='sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'
MODEL=PREVIOUS_ROOT/'models/DeepSeek-V4.1-Flash-1017-w4a8';PORT=19093 if ROLE=='p' else 19094;LIFETIME=72000
LOCKS=[];OWNERS=[];CLAIMED=[];CHILD=None;CREATE_ATTEMPTED=False;FAILURE=None


def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)

def cmd(args,timeout=30):return subprocess.run(args,capture_output=True,text=True,timeout=timeout,cwd=ROOT)

def state(phase,**extra):
 d=dict(task_id=TASK_ID,host=HOST,role=ROLE,generation=437,pid=os.getpid(),phase=phase,
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
 if HOST not in ['<ROLE_HOSTNAME>','<ROLE_HOSTNAME>'] or os.geteuid()!=0:raise RuntimeError('Requires root on <ROLE_HOSTNAME>')
 if Path(__file__).resolve().parent!=RUN:raise RuntimeError('Deploy controller and entry files to '+str(RUN))
 if (RUN/'state.json').exists():raise RuntimeError('Existing run requires explicit recovery; state preserved')
 for name in ['run_pd.sh','launch_pd.py','candidate.json','adapter.py','source_uva.py','ds41_native_gate/__init__.py','ds41_native_gate/entry.py']:
  if not (RUN/name).is_file():raise RuntimeError('Missing entry '+name)
 if not MODEL.is_dir():raise RuntimeError('Missing model root')
 RUN.mkdir(parents=True,exist_ok=True);COORD.mkdir(parents=True,exist_ok=True)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  assert inspect() is None,'Container name already exists'
  # Do not claim NPUs while waiting for the last owned checksum-pinned download.
  waiting_deadline=time.monotonic()+1800
  download_state=PREVIOUS_ROOT/('download_state_g64.json' if ROLE=='p' else 'download_state_g60.json')
  downloader='download_repair_g64.py' if ROLE=='p' else 'download_repair_g60.py'
  state('waiting_verified_weights',download_state=str(download_state))
  while True:
   d=json.loads(download_state.read_text()) if download_state.exists() else None
   if d and d['phase']=='complete':break
   assert d is not None,'Missing authoritative download state'
   proc=Path('/proc',str(d['pid']),'cmdline');assert proc.exists() and downloader.encode() in proc.read_bytes(),'Download stopped before complete; preserve state'
   assert time.monotonic()<waiting_deadline,'Verified-weight wait deadline reached'
   if (RUN/'stop.request').exists():raise RuntimeError('Stopped during weight wait')
   time.sleep(15)
  assert len(d['verified'])==d['total_files']==95 and not d['failed']
  for name,e in d['verified'].items():assert (MODEL/name).stat().st_size==e['bytes']
  atomic(RUN/'verified_weight_snapshot.json',d)
  for device in range(8):
   key=HOST+'-npu'+str(device);f=(COORD/(key+'.lock')).open('a')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except Exception:f.close();raise
   LOCKS.append(f);p=COORD/(key+'.owner.json')
   if p.exists():assert json.loads(p.read_text()).get('state')=='released','Existing ownership requires reconciliation: '+str(p)
   OWNERS.append(p)
  assert len(LOCKS)==len(OWNERS)==8
  npu=cmd(['npu-smi','info']);(RUN/'npu_before.txt').write_text(npu.stdout+npu.stderr)
  assert npu.returncode==0 and all('No running processes found in NPU '+str(i) in npu.stdout for i in range(8)),'NPU occupancy changed; reconcile first'
  for port in [PORT]+([18998,18678] if ROLE=='p' else []):
   f=(COORD/(HOST+'-port'+str(port)+'.lock')).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);LOCKS.append(f)
   with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('0.0.0.0',port))
  ip=cmd(['ip','-j','-4','addr','show','dev','enp67s0f0']);assert ip.returncode==0
  expected_ip='<ROLE_HOST>' if ROLE=='p' else '<ROLE_HOST>';assert expected_ip in ip.stdout
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
        '--label','codex.run='+RUN.name,'--network','host','--read-only','--log-driver','none','--cap-add','SYS_PTRACE',
        '--security-opt','seccomp=unconfined','--cpuset-cpus','0-159','--cpus','160','--memory','1200g',
        '-v',str(ROOT)+':/task:ro','-v',str(RUN)+':/task/sg_pd_g437',
        '-v',str(MODEL)+':/models/ds41:ro','-v',str(RUN/'tmp')+':/tmp',
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
   if value.startswith('/task/'):(ROOT/value.removeprefix('/task/')).mkdir(parents=True,exist_ok=True)
  for k,v in env.items():args+=['-e',k+'='+v]
  args+=['--entrypoint','bash',IMAGE,'-c','exec sleep 72100']
  atomic(RUN/'container_argv.json',args);CREATE_ATTEMPTED=True
  r=cmd(args,90);atomic(RUN/'container_start.json',dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr))
  assert r.returncode==0,'Container start failed: '+r.stderr
  info=inspect();assert info is not None and ours(info),'Container identity changed'
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
