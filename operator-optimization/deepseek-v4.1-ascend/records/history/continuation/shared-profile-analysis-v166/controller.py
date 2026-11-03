"""One owned service reload, full fixed case and reusable low-overhead profiler."""
import hashlib,importlib.util,json,os,re,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
C=Path('/data/tiankuan/wio/migration-171-20261102/builtin-cohort-v164-228ad25fcf9b')
sys.path.insert(0,str(C))
spec=importlib.util.spec_from_file_location('owned_tools',C/'controller.py');tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/npu166')
APP='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash'
SERVE=ROOT.parent.parent/'scripts/serve.sh'
EXPECTED='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
state={};child=None;changed_runtime=False;script_changed=False
def save():
 p=ROOT/'STATUS.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ROOT/'STATUS.json')
def restore_script():
 global script_changed
 if script_changed:
  b=(ROOT/'serve-original.sh').read_bytes();p=SERVE.with_suffix('.v166-restore');p.write_bytes(b);p.chmod(0o755);p.replace(SERVE)
  assert hashlib.sha256(SERVE.read_bytes()).hexdigest()==EXPECTED
  script_changed=False;state['original_startup_script_restored']=True;save()
def health():
 try:return tools.request(30075,'/health',timeout=3)[0]==200
 except Exception:return False
def wait_health(seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  if health():return
  m=tools.inspect(tools.MAIN)
  if not m['State']['Running']:raise RuntimeError('Owned service exited while loading: '+str(m['State']))
  state['last_load_check_unix']=time.time();save();time.sleep(3)
 raise TimeoutError('Owned model readiness deadline exceeded')
def guard():
 m=tools.inspect(tools.MAIN)
 assert m['State']['Running'] and m['Config']['Labels'].get('wio.migration')=='166-to-171-20261102'
 assert hashlib.sha256(SERVE.read_bytes()).hexdigest()==EXPECTED
 tools.idle_metrics(tools.request(30075,'/metrics')[1].decode())
 busy,text=tools.busy();tools.owns_processes(busy,tools.owned(tools.MAIN));(ROOT/'npu-before.txt').write_text(text)
 return m
def worker_inventory():
 code="import json,re;from pathlib import Path;r=[]\nfor p in Path('/proc').glob('[0-9]*'):\n try:\n  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode().strip()\n  if re.fullmatch(r'VLLM::Worker_DP\\d+_TP\\d+_EP\\d+',cmd):r.append({'pid':int(p.name),'name':cmd,'start_ticks':(p/'stat').read_text().rsplit(')',1)[1].split()[19]})\n except (FileNotFoundError,PermissionError,ProcessLookupError):pass\nprint(json.dumps(r))"
 return json.loads(tools.command(['docker','exec',tools.MAIN,'python3','-c',code],30))
def monitor():
 if time.monotonic()>deadline:raise TimeoutError('Acquisition total deadline reached')
 size=tools.disk_bytes(ROOT);state['observed_output_bytes']=size
 if size>16*1024**3:raise RuntimeError('16GiB acquisition stop threshold exceeded')
 save()
def run_client():
 global child
 env=dict(os.environ);env.update(WIO_PHASE='profile',WIO_OUTPUT=str(ROOT/'case'),
  WIO_FROZEN_ROOT=str(C/'runtime/benchmark-20261028/workload/frozen'),WIO_API_URL='http://127.0.0.1:30075')
 with (ROOT/'client.stdout').open('w') as out,(ROOT/'client.stderr').open('w') as err:
  child=subprocess.Popen(['python3','-I','-S',str(C/'kit/client.py')],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  state.update(state='FULL_FIXED_CASE_PROFILE_RUNNING',client_pid=child.pid,client_start_ticks=tools.proc_start(child.pid),profile_attempts=1);save()
  end=time.monotonic()+600
  while child.poll() is None:
   monitor()
   if time.monotonic()>end:raise TimeoutError('Full fixed-case client deadline exceeded')
   time.sleep(1)
  rc=child.returncode;child=None
 if rc:raise RuntimeError('Full fixed-case client returned '+str(rc))
 status=json.loads((ROOT/'case/window-status.json').read_text());assert status['valid']
 before=json.loads((C/'outputs/baseline/responses.json').read_text());after=json.loads((ROOT/'case/responses.json').read_text())
 matches=[x['text']==y['text'] for x,y in zip(before,after)];assert len(before)==len(after)==8 and all(matches)
 state.update(case_valid=True,exact_response_text_matches=sum(matches),windows=status['windows']);save()
def execute():
 global deadline,changed_runtime,script_changed,state
 assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171' and os.geteuid()==0
 fd=os.open(ROOT/'STATUS.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
 deadline=time.monotonic()+2700
 state={'state':'GUARD','pid':os.getpid(),'start_ticks':tools.proc_start(os.getpid()),'started_unix':time.time(),
  'main_container':tools.MAIN,'profile_attempts':0,'model_reloads':0,'overall_complete':False,
  'authorization':'User explicitly authorizes owned container inference/framework/probe processes.',
  'mode':'NPU-only,Level0,noCPU-stack,no-memory,no-PMU;raw export offline',
  'timing_comparability':'New service incarnation and existing caches;not a stable production before/after or speedup proof.'};save()
 try:
  old=guard();state['main_started_at_before']=old['State']['StartedAt'];save()
  (ROOT/'serve-original.sh').write_bytes(SERVE.read_bytes())
  log=ROOT.parent.parent/'logs/service.log'
  if log.exists():
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-131072));(ROOT/'prior-service-log-tail.txt').write_bytes(f.read())
   subprocess.run(['cp','--reflink=auto',str(log),str(ROOT/'prior-service.log')],check=True)
  p=SERVE.with_suffix('.v166-new');p.write_bytes((ROOT/'serve-enabled.sh').read_bytes());p.chmod(0o755);p.replace(SERVE);script_changed=True
  state['state']='RELOADING_OWN_SERVICE_FOR_REUSABLE_PROFILE_ENTRY';save();changed_runtime=True
  tools.command(['docker','stop','--time','60',tools.MAIN],90)
  end=time.monotonic()+90
  while True:
   busy,_=tools.busy()
   if not busy:break
   if time.monotonic()>end:raise RuntimeError('NPU process remained;no foreign task will be killed')
   time.sleep(2)
  tools.command(['docker','start',tools.MAIN],30);state['model_reloads']=1;state['state']='WAITING_PROFILE_CAPABLE_MODEL_HEALTH';save();wait_health(2100)
  # Preserve original files for all subsequent normal starts. The currently
  # running owned service retains the explicit diagnostic capability until restart.
  restore_script();workers=worker_inventory();assert len(workers)==16
  for w in workers:
   headers=list((ROOT/'hook-records').glob(str(w['pid'])+'-*-hook_installed-*.json'));assert len(headers)==1,(w,headers)
  state['actual_workers']=workers;state['profile_entry_ready_in_all16_workers']=True;save()
  tools.idle_metrics(tools.request(30075,'/metrics')[1].decode());run_client()
  state['final_stop_profile_http']=tools.request(30075,'/stop_profile',{},timeout=120)[0]
  assert health();tools.idle_metrics(tools.request(30075,'/metrics')[1].decode())
  profiles=sorted((ROOT/'raw').glob('*_ascend_pt'));assert len(profiles)==32,len(profiles)
  constructors=[json.loads(p.read_text()) for p in (ROOT/'hook-records').glob('*-profile_constructed-*.json')]
  assert len(constructors)==32,len(constructors)
  assert all(r['activities']==['NPU'] and not any(r[k] for k in ['record_shapes','with_stack','with_modules','profile_memory']) for r in constructors)
  state.update(state='FULL_FIXED_CASE_DEVICE_ONLY_PROFILE_COMPLETE',finished_unix=time.time(),profile_count=len(profiles),constructor_receipts=len(constructors),health_http=200,
   live_service_profile_capability=True,original_script_sha256=hashlib.sha256(SERVE.read_bytes()).hexdigest(),main_started_at_after=tools.inspect(tools.MAIN)['State']['StartedAt'])
  save()
 except BaseException:
  state['failure']=traceback.format_exc();state['state']='FAILED_RESTORING_ORIGINAL_SERVICE';save()
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  restore_script()
  if changed_runtime:
   tools.command(['docker','stop','--time','60',tools.MAIN],90)
   end=time.monotonic()+180
   while tools.busy()[0]:
    if time.monotonic()>end:raise RuntimeError('Restore waiting for foreign NPU processes;no kill')
    time.sleep(2)
   tools.command(['docker','start',tools.MAIN],30);wait_health(2400)
  state.update(state='FAILED_ORIGINAL_SERVICE_RESTORED',health_http=200 if health() else None,finished_unix=time.time(),live_service_profile_capability=False);save()
  raise
if __name__=='__main__':
 def interrupted(signum,frame):raise RuntimeError('Owned controller interrupted: '+str(signum))
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 execute()
