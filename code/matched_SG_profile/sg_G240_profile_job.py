"""CPU AISBench child, native staged eight-step P/D capture, no service edits."""
from pathlib import Path
import os,json,time,datetime,subprocess,urllib.request,urllib.error,concurrent.futures,signal,hashlib,re
ROOT=Path('/task/client_vllm_g42/profile_sg_g240');os.chdir(ROOT)
def interrupted(sig,frame):raise RuntimeError('Profilejob signal '+str(sig))
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
def save(name,obj):
 p=ROOT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def state(phase,**kw):save('state.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),phase=phase,**kw))
def call(role,action,body=None):
 url=('http://<HOST>:19083' if role=='p' else 'http://<HOST>:19084')+'/'+action;r=dict(role=role,action=action,url=url,at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 try:
  req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else b'',headers={'Content-Type':'application/json'},method='POST')
  with urllib.request.urlopen(req,timeout=120) as q:r.update(status=q.status,body=q.read().decode())
 except urllib.error.HTTPError as e:r.update(status=e.code,body=e.read().decode(),error=repr(e))
 except BaseException as e:r.update(error=repr(e))
 save(role+'_'+action+'.json',r);return r
roles=['p','d'];attempted=[];proc=None;error=None;stop={}
try:
 manifest=json.loads((ROOT/'assets_manifest.json').read_text());b=(ROOT/'dataset.jsonl').read_bytes();assert hashlib.sha256(b).hexdigest()==manifest['dataset_sha256']
 state('waiting_to_launch_actual_C16_before_profiler_arm')
 def start(role):
  attempted.append(role);return call(role,'start_profile',dict(output_dir='/task/sg_pd_g181/profile_g240/traces/'+role,num_steps=8,activities=['CPU','GPU'],profile_by_stage=True,with_stack=False,record_shapes=False,profile_id='sg-g240-'+role,merge_profiles=False,profile_prefix='sg-g240-'+role,detailed_annotations=True))
 argv=['ais_bench',str(ROOT/'aisbench.py'),'--mode','perf','--num-warmups','0','--num-prompts','16','--max-num-workers','1','--work-dir',str(ROOT/'aisbench_work')];save('workload_command.json',dict(argv=argv,requests=16,concurrency=16,input_tokens=32768,output_tokens=1024))
 with (ROOT/'aisbench.log').open('x') as f:
  begun=time.monotonic();proc=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);state('AISBench_actual_child_running_before_profile',workload_pid=proc.pid)
  observations=[]
  while time.monotonic()-begun<180:
   assert proc.poll() is None,('AISBench exited before queuedrequests',proc.returncode)
   with urllib.request.urlopen('http://<HOST>:19083/metrics',timeout=10) as q:body=q.read().decode()
   counters={}
   for name in ['sglang:num_queue_reqs','sglang:num_prefill_bootstrap_queue_reqs','sglang:num_prefill_inflight_queue_reqs']:
    vals=[float(line.split()[-1]) for line in body.splitlines() if line.startswith(name+'{') and 'tp_rank="0"' in line];assert vals,name;counters[name]=sum(vals)
   observations.append(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**counters))
   if sum(counters.values())>=2:break
   time.sleep(.25)
  else:raise RuntimeError('No multiple actualP requests observed before profilearm bound')
  save('actual_queued_requests_before_arming.json',dict(observations=observations,workload_pid=proc.pid,scope='Arm aftermultiplequeued realP requests; by-stage predicate excludesidle/prebuilt. Actualhealth/shape inclusion muststill bechecked inrawtrace.'))
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:starts=dict(zip(roles,pool.map(start,roles)))
  assert all(v.get('status')==200 for v in starts.values()),starts
  state('native_armed_eight_real_stage_steps',starts=starts,workload_pid=proc.pid)
  rc=proc.wait(timeout=max(1,600-(time.monotonic()-begun)));assert rc==0,rc

except BaseException as exc:error=repr(exc)
finally:
 if proc is not None and proc.poll() is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
 state('explicit_native_stop_reconciliation',error=error)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:stop=dict(zip(attempted,pool.map(lambda r:call(r,'stop_profile'),attempted)))
 state('terminal_requires_parent_all16_native_done_and_raw_stability',error=error,workload_rc=proc.returncode if proc else None,stop_responses=stop,auto_stop_500_not_accepted_without_native8rank_done_and_stable_raw=True,device_coverage_verified=False)
if error:raise SystemExit(1)
