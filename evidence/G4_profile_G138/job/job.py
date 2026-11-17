"""One bounded fresh-input G4 profile after all four scored shapes, without service restart."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,signal,subprocess,sys,urllib.request
from collect_profile import main as collect_main,validate
from audit_aisbench import audit_phase

ROOT=Path('<TASK_ROOT>')
RUN=ROOT/'vllm_G4_profile_g138';CLIENT=ROOT/'client_vllm_g42';OUT=CLIENT/'profile_g138'
NAME='yxy-glm53-aisbench-0002-g42'
def atomic(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def state(phase,**kw):
 d={'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'diagnostic_only':True,'benchmark_score':False,**kw}
 atomic(RUN/'state.json',d)
def snapshot(roles,directory):
 directory.mkdir(exist_ok=False)
 for role,entry in roles.items():
  with urllib.request.urlopen(entry['endpoint']+'/metrics',timeout=15) as response:body=response.read().decode()
  atomic(directory/(role+'.json'),{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'body':body})
  for metric in ['vllm:num_requests_running','vllm:num_requests_waiting']:
   values=[float(v) for v in re.findall('^'+re.escape(metric)+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)(?:\s+\d+)?$',body,re.M)]
   assert values and sum(values)==0,(role,metric,values)

def main():
 assert Path(__file__).resolve().parent==RUN
 lock=Path('<COORD_ROOT>/<ROLE_NODE>-client-g42.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state('preflight')
 signal.signal(signal.SIGTERM,lambda sig,frame:(_ for _ in ()).throw(RuntimeError('Profile job received SIGTERM')))
 gate=json.loads((RUN/'gate.json').read_text());assert gate['service_generation']==4
 assert (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(gate['validated_at'])).total_seconds()<1200
 completed=CLIENT/'runs/vllm/vllm_G4_C16_256K_g56'
 assert json.loads((completed/'state.json').read_text())['phase']=='all_phases_terminal'
 assert json.loads((completed/'audit_results.json').read_text())['all_required_raw_metrics_and_usage_validated']
 top=subprocess.check_output(['docker','top',NAME,'-eo','pid,args'],text=True);assert 'ais_bench' not in top
 client=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0]
 assert client['Id']==gate['client_container_id'] and client['Image']==gate['client_image_id'] and client['State']['Running']
 assert client['HostConfig']['Devices'] in [[],None]
 assert client['HostConfig']['NanoCpus']==4000000000 and client['HostConfig']['Memory']==34359738368
 atomic(RUN/'actual_client_identity.json',client)
 OUT.mkdir(exist_ok=False);snapshot(gate['roles'],RUN/'before')
 state('CPU_assets_preparation')
 prepare=['docker','exec','-w','/task',NAME,'timeout','--signal=TERM','--kill-after=15','300','python3','/task/vllm_G4_profile_g138/prepare_assets.py']
 with (RUN/'prepare.log').open('x') as f:subprocess.run(prepare,stdout=f,stderr=subprocess.STDOUT,timeout=330,check=True,cwd=ROOT)
 assets=json.loads((OUT/'assets/assets.json').read_text());assert assets['first128_unique_vs_all_source_phases']
 for filename,key in [('profile_32768.jsonl','profile_sha256'),('profile.py','config_sha256')]:
  assert hashlib.sha256((OUT/'assets'/filename).read_bytes()).hexdigest()==assets[key]
 workload=['docker','exec','-w','/task','-e','BFCL_PROJECT_ROOT=/task/client_vllm_g42/bfcl',NAME,'timeout','--signal=TERM','--kill-after=30','570',
           'ais_bench','/task/client_vllm_g42/profile_g138/assets/profile.py','--mode','perf','--num-warmups','0','--num-prompts','16','--max-num-workers','1',
           '--work-dir','/task/client_vllm_g42/profile_g138/aisbench']
 gate.update(workload_argv=workload,workload_cwd=str(ROOT));validate(gate);atomic(RUN/'capture_gate.json',gate)
 snapshot(gate['roles'],RUN/'immediately_before_capture')
 state('paired_profile_capture')
 previous=sys.argv
 try:
  sys.argv=['collect_profile.py','--gate',str(RUN/'capture_gate.json'),'--out',str(RUN/'capture_control'),'--execute'];collect_main()
 finally:sys.argv=previous
 control=json.loads((RUN/'capture_control/state.json').read_text())
 assert control['phase']=='terminal' and control['workload_rc']==0 and not control['error'] and not control['stop_errors']
 for role in ['p','d']:assert json.loads((RUN/f'capture_control/{role}_stop.json').read_text())['status']==200
 a=audit_phase(OUT/'aisbench',32768,requests=16,usage_contract=ROOT/'client_usage_contract_g45.json')
 assert a['fully_validated'];a.update(diagnostic_only=True,benchmark_score=False);atomic(RUN/'profile_workload_audit.json',a)
 snapshot(gate['roles'],RUN/'after')
 state('capture_terminal_pending_all16_raw_device_flush',profile_success_verified=False)

if __name__=='__main__':
 try:main()
 except BaseException as e:state('failed_requires_review',error=repr(e),profile_success_verified=False);raise
