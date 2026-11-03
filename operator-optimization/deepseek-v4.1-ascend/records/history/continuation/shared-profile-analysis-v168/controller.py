"""One shared diagnostic cohort;two planned owned-service reloads,then original."""
import hashlib,importlib.util,json,os,re,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
C=Path('/data/tiankuan/wio/migration-171-20261102/builtin-cohort-v164-228ad25fcf9b')
sys.path.insert(0,str(C));spec=importlib.util.spec_from_file_location('owned_tools',C/'controller.py');tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
ROOT=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168')
SERVE=ROOT.parent.parent/'scripts/serve.sh';EXPECTED='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
state={};child=None;script_changed=False;runtime_changed=False;restoring=False;deadline=0

def save():
 p=ROOT/'STATUS.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(ROOT/'STATUS.json')
def health():
 try:return tools.request(30075,'/health',timeout=3)[0]==200
 except Exception:return False
def monitor(acquisition=True):
 if acquisition and time.monotonic()>deadline:raise TimeoutError('Acquisition45minute deadline')
 size=tools.disk_bytes(ROOT);state['observed_output_bytes']=size
 if acquisition and size>16*1024**3:raise RuntimeError('16GiB diagnostic stop threshold exceeded')
 save()
def wait_health(seconds,acquisition=True):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  if health():return
  m=tools.inspect(tools.MAIN)
  if not m['State']['Running']:raise RuntimeError('Owned service exited: '+str(m['State']))
  state['last_load_check_unix']=time.time();monitor(acquisition);time.sleep(3)
 raise TimeoutError('Owned model health deadline')
def restore_script():
 global script_changed
 if script_changed:
  p=SERVE.with_suffix('.joint168-restore');p.write_bytes((ROOT/'serve-original.sh').read_bytes());p.chmod(0o755);p.replace(SERVE)
  assert hashlib.sha256(SERVE.read_bytes()).hexdigest()==EXPECTED
  script_changed=False;state['original_script_restored']=True;save()
def stop_child():
 global child
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 child=None

def idle_owned_guard():
 m=tools.inspect(tools.MAIN)
 assert m['State']['Running'] and m['Config']['Labels'].get('wio.migration')=='166-to-171-20261102'
 assert hashlib.sha256(SERVE.read_bytes()).hexdigest()==EXPECTED
 tools.idle_metrics(tools.request(30075,'/metrics')[1].decode())
 busy,text=tools.busy();tools.owns_processes(busy,tools.owned(tools.MAIN));(ROOT/'npu-before.txt').write_text(text)
 return m

def wait_vacant(seconds):
 end=time.monotonic()+seconds
 while tools.busy()[0]:
  if time.monotonic()>end:raise RuntimeError('NPU processes remain;no unrelated process will be killed')
  time.sleep(2)

def restore_runtime():
 global restoring,runtime_changed
 if not runtime_changed:return
 restoring=True;state['state']='RESTORING_ORIGINAL_OWNED_SERVICE';save();restore_script()
 tools.command(['docker','stop','--time','60',tools.MAIN],90);wait_vacant(180)
 tools.command(['docker','start',tools.MAIN],30);state['model_reloads']+=1;save();wait_health(2400,False)
 runtime_changed=False;state['original_runtime_restored']=True;state['original_restored_started_at']=tools.inspect(tools.MAIN)['State']['StartedAt'];save()

def run_case():
 global child
 env=dict(os.environ);env.update(WIO_PHASE='profile',WIO_OUTPUT=str(ROOT/'case'),WIO_FROZEN_ROOT=str(C/'runtime/benchmark-20261028/workload/frozen'),WIO_API_URL='http://127.0.0.1:30075')
 with (ROOT/'client.stdout').open('w') as out,(ROOT/'client.stderr').open('w') as err:
  child=subprocess.Popen(['python3','-I','-S',str(ROOT/'client.py')],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  state.update(state='FULL_CASE_JOINT_COLLECTION_RUNNING',client_pid=child.pid,client_start_ticks=tools.proc_start(child.pid),case_attempts=1);save();end=time.monotonic()+600
  while child.poll() is None:
   monitor()
   if time.monotonic()>end:raise TimeoutError('FullCase diagnostic client10minute deadline')
   time.sleep(1)
  state['client_returncode']=child.returncode;child=None;save()
 # Validation is reported separately from collection and cannot trigger another acquisition/reload.
 if state['client_returncode']!=0:raise RuntimeError('Diagnostic client returned '+str(state['client_returncode']))
 state['state']='ACQUISITION_RETURNED_REVIEW_PENDING';state['acquisition_finished_unix']=time.time();save()


def review_receipts():
 notes=[];case=ROOT/'case/window-status.json';status=json.loads(case.read_text()) if case.exists() else {}
 state['case_valid']=bool(status.get('valid'));state['counter_deltas']=status.get('counter_deltas')
 a=C/'outputs/baseline/responses.json';b=ROOT/'case/responses.json'
 if a.exists() and b.exists():
  old=json.loads(a.read_text());new=json.loads(b.read_text());state['exact_response_text_matches']=sum(x['text']==y['text'] for x,y in zip(old,new));state['same8responses']=len(old)==len(new)==8 and state['exact_response_text_matches']==8
 finals=[]
 for p in sorted((ROOT/'workers').glob('*/FINAL.json')):
  d=json.loads(p.read_text());finals.append({'path':str(p),'pid':d['pid'],'process_uuid':d['process_uuid'],'epochs':d['epochs'],'banks':len(d['banks']),'graphs':len(d['graphs']),'overflow':d['overflow'],'history_bytes':(p.parent/'history.pt').stat().st_size})
 state['worker_histories']=finals;state['startup_profiles']=len(list((ROOT/'startup').glob('*_ascend_pt')));state['runtime_profiles']=len(list((ROOT/'runtime').glob('*_ascend_pt')))
 if len(finals)!=16:notes.append('Expected16actualworkerhistoryreceipts;observed'+str(len(finals)))
 if any(d['overflow'] for d in finals):notes.append('Historyepochoverflow;fullhistorycoverage unproven')
 if state['startup_profiles']!=16 or state['runtime_profiles']!=16:notes.append('Profilecount mismatch;retain actual data for independentreview')
 if not state['case_valid'] or not state.get('same8responses'):notes.append('FullCase/output equality not established')
 state['review_notes']=notes;state['review_status']='INDEPENDENT_DB_AND_HISTORY_REVIEW_PENDING';save()

def execute():
 global state,deadline,runtime_changed,script_changed
 assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171' and os.geteuid()==0
 fd=os.open(ROOT/'STATUS.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
 deadline=time.monotonic()+2700
 state={'state':'GUARD','pid':os.getpid(),'start_ticks':tools.proc_start(os.getpid()),'started_unix':time.time(),'model_reloads':0,'case_attempts':0,'overall_complete':False,
 'authorization':'User broadexplicit authorization for this owned container inference/framework/probes.','scope':'CreationCPU+NPU,fullCaseNPU-only,allMC2count/valid rows;IDs onlytargetfirst+ordinal20capturedM<=24.','diagnostic_timings_are_not_baseline_performance':True};save()
 failure=None
 try:
  m=idle_owned_guard();state['main_started_at_before']=m['State']['StartedAt'];save();(ROOT/'serve-original.sh').write_bytes(SERVE.read_bytes())
  p=SERVE.with_suffix('.joint168-enabled');p.write_bytes((ROOT/'serve-enabled.sh').read_bytes());p.chmod(0o755);p.replace(SERVE);script_changed=True
  runtime_changed=True;state['state']='RELOADING_OWNED_SERVICE_FOR_JOINT_CAPTURE';save()
  tools.command(['docker','stop','--time','60',tools.MAIN],90);wait_vacant(90)
  tools.command(['docker','start',tools.MAIN],30);state['model_reloads']+=1;state['state']='WAITING_JOINT_MODEL_HEALTH';save();wait_health(2100)
  restore_script();state['diagnostic_started_at']=tools.inspect(tools.MAIN)['State']['StartedAt'];state['state']='JOINT_MODEL_READY';save()
  tools.idle_metrics(tools.request(30075,'/metrics')[1].decode());run_case()
 except BaseException:
  failure=traceback.format_exc();state['acquisition_failure']=failure;state['state']='ACQUISITION_FAILED_RESTORE_REQUIRED';save();stop_child()
 finally:
  restore_script()
  try:restore_runtime()
  except BaseException:
   state['restoration_failure']=traceback.format_exc();state['state']='RESTORATION_REQUIRES_ATTENTION';save();raise
 # Postcollection review errors remain reviewpending;originalservicealreadyrestored.
 try:review_receipts()
 except BaseException:state['review_failure']=traceback.format_exc();save()
 state.update(state='ORIGINAL_SERVICE_RESTORED_REVIEW_PENDING' if failure is None else 'ACQUISITION_FAILED_ORIGINAL_SERVICE_RESTORED',health_http=200 if health() else None,finished_unix=time.time(),original_serve_sha256=hashlib.sha256(SERVE.read_bytes()).hexdigest());save()
 if failure:raise RuntimeError(failure)
if __name__=='__main__':
 def interrupted(signum,frame):raise RuntimeError('Owned cohort controller interrupted:'+str(signum))
 signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted);execute()
