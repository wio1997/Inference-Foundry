"""Stage a concrete stopped diagnostic container; no model/device execution."""
import sys,json,hashlib,base64,uuid
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke

source=N/'shared-builtin-cohort-v164'
assert read(source/'cpu-bundle-contract-result.json')['status']=='PASS_NEW_CPU_BUNDLE_CONTRACTS_ONLY'
assert not (O/'shared-builtin-prepared-v164.json').exists(),'Prepare once;inspect a partial preparation rather than silently creating another one'
run_id='builtin171-v164-'+uuid.uuid4().hex[:12]
remote='/data/tiankuan/wio/migration-171-20261102/builtin-cohort-v164-'+run_id.rsplit('-',1)[-1]
plan={'schema':'ONE_SHARED_FIXED_CASE_BUILTIN_COHORT_V1','run_id':run_id,
 'root':remote,'target':'172.16.10.171','approval_slot_id':run_id+'-window',
 'original_service':'3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb',
 'diagnostic_name':'wio-'+run_id,'diagnostic_port':30076,
 'case':{'concurrency':8,'input_tokens_each':32768,'output_tokens_each':1024,
  'prefix_queries':262144,'prefix_hits':235904,'zero_preemptions':True,
  'temperature':0,'ignore_eos':True,'seed_requests_per_pass':2,
  'formal_passes':['baseline','profile'],'same_frozen_requests_and_client_algorithm':True},
 'observation':{'startup':'One CPU-only raw profiler around capture_model per worker',
  'formal':'Two existing-client requested windows,each at most3 worker execute_model calls per rank',
  'raw_flags':{'record_shapes':True,'profile_memory':True,'with_stack':True},
  'formal_CPU_and_NPU':True,'ignore_frontend':True,'offline_auto_analysis':False,
  'graph_and_runner_guard_scopes':True,'TorchDispatchMode':False,'native_interposition':False,
  'window_phase_labels_are_candidates_not_unconditional_device_phase_proofs':True},
 'limits':{'acquisition_seconds':2700,'diagnostic_load_seconds':1800,'client_seconds':900,
  'restore_load_seconds':2400,'directory_stop_threshold_bytes':16*1024**3,
  'directory_limit_poll_seconds':1,'single_file_hard_bytes':256*1024**2,
  'hard_aggregate_directory_cap':None,'polling_overshoot_possible':True,
  'supplement_records_per_process':65536,'supplement_bytes_per_process':64*1024**2,
  'Docker_log_rotation_max_bytes':30*1024**2},
 'restoration':{'stop_only_original_own_service_after_reference_valid':True,
  'diagnostic_container_stopped_afterward':True,'original_service_reloaded_and_health200_required':True,
  'foreign_NPU_processes_never_killed_or_overlapped':True,'occupied_pool_defers_restore':True,
  'no_automatic_workload_retry_or_resubmit_from_observation_timeout':True},
 'not_proved_by_this_collection':['historical1352 task parameters','all native/error/bypass/readers',
  'full module/branch exhaustiveness','physical_MFU/lowest/theoretical capacity','optimization speedup'],
 'authorization':'NOT_GRANTED_THIS_IS_A_PREPARED_SLOT_ONLY','overall_complete':False}
files={}
for p in sorted((source/'kit').glob('*')):
 if p.is_file():files['kit/'+p.name]=base64.b64encode(p.read_bytes()).decode()
for name in ['controller.py','cohort_policy.py']:
 files[name]=base64.b64encode((source/name).read_bytes()).decode()
payload={'plan':plan,'files':files,
 'frozen':read(O/'shared-builtin-frozen-case-v164.json')['files'],
 'serve_sha256':'c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba',
 'client_sha256':'06f8356a07e72a7e20a1ee17d737dbf1f466c5855e80e2057dcd7d2de870a578'}
worker=r'''import json,base64,copy,hashlib,http.client,os,shutil,socket,stat,subprocess,urllib.parse
from pathlib import Path
plan=PAYLOAD['plan'];root=Path(plan['root']);main=plan['original_service']
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171' and os.geteuid()==0
assert root.parent==Path('/data/tiankuan/wio/migration-171-20261102') and not root.exists()
root.mkdir(mode=0o700);(root/'control').mkdir();(root/'outputs').mkdir()
class Docker(http.client.HTTPConnection):
 def __init__(self):super().__init__('localhost',timeout=30)
 def connect(self):self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.connect('/var/run/docker.sock')
def api(method,path,body=None):
 c=Docker();c.request(method,path,body=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 r=c.getresponse();data=r.read();code=r.status;c.close();assert 200<=code<300,(code,data[:1000]);return json.loads(data) if data else None
d=api('GET','/containers/'+main+'/json')
assert d['State']['Running'] and d['Config']['Labels']['wio.migration']=='166-to-171-20261102'
assert d['HostConfig']['ReadonlyRootfs'] and d['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
assert shutil.disk_usage(root).free>64*1024**3
source=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime')
backup=Path('/data/tiankuan/wio/migration-171-20261102/bringup-171-20261102-120a3333/original-runtime')
assert hashlib.sha256((source/'scripts/serve.sh').read_bytes()).hexdigest()==PAYLOAD['serve_sha256']
assert hashlib.sha256((source/'tools/whole_scope166_reference_client.py').read_bytes()).hexdigest()==PAYLOAD['client_sha256']
def ignore(path,names):
 if Path(path)==backup:return [n for n in names if n in ['cache','tmp','logs','runs','diagnostics']]
 return [n for n in names if not stat.S_ISREG((Path(path)/n).lstat().st_mode) and not stat.S_ISDIR((Path(path)/n).lstat().st_mode)]
shutil.copytree(backup,root/'runtime',ignore=ignore,symlinks=True)
for name in ['home','tmp','cache/xdg','cache/huggingface','cache/vllm','cache/torchinductor','cache/triton','cache/ascend','cache/ascend-work','logs/ascend','runs','diagnostics/startup','diagnostics/formal','diagnostics/supplement']:
 (root/'runtime'/name).mkdir(parents=True,exist_ok=True)
(root/'runtime/tmp').chmod(0o1777);(root/'container-tmp').mkdir(mode=0o1777)
assert (source/'config/serve.env').read_bytes()==(root/'runtime/config/serve.env').read_bytes()
assert hashlib.sha256((root/'runtime/scripts/serve.sh').read_bytes()).hexdigest()==PAYLOAD['serve_sha256']
frozen=source/'benchmark-20261028/workload/frozen';clone=root/'runtime/benchmark-20261028/workload/frozen'
for row in PAYLOAD['frozen']:
 a=frozen/row['relative'];b=clone/row['relative']
 assert a.read_bytes()==b.read_bytes() and hashlib.sha256(a.read_bytes()).hexdigest()==row['sha256']
for rel,value in PAYLOAD['files'].items():
 p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value));p.chmod(0o600)
plan_bytes=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(root/'control/PLAN.json').write_bytes(plan_bytes)
plan_sha=hashlib.sha256(plan_bytes).hexdigest()
cfg=copy.deepcopy(d['Config']);host=copy.deepcopy(d['HostConfig'])
cfg['Hostname']='';cfg['Image']=d['Image'];cfg['Entrypoint']=['/usr/local/python3.12.13/bin/python3'];cfg['Cmd']=['-I','-S','/wio-diagnostic/launch_diag.py']
cfg['Labels']=dict(cfg.get('Labels') or {},**{'wio.shared-cohort':plan['run_id'],'wio.parent-main':main,'wio.plan-sha256':plan_sha})
env=dict(x.split('=',1) for x in cfg['Env']);env.update(WIO_BUILTIN_COHORT_ENABLE='1',WIO_COHORT_CONTROL='/wio-window-control',
 WIO_COHORT_OUTPUT='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/diagnostics',WIO_COHORT_APPROVAL_ID=plan['approval_slot_id'],
 WIO_TARGETED_DISPATCH_ENABLE='0',WIO_HOST_TRACE_ENABLE='0')
cfg['Env']=[k+'='+v for k,v in env.items()]
host['PortBindings']={'8172/tcp':[{'HostIp':'127.0.0.1','HostPort':str(plan['diagnostic_port'])}]}
host['LogConfig']={'Type':'json-file','Config':{'max-size':'10m','max-file':'3'}}
assert host['RestartPolicy']['Name']=='no'
for mount in host['Mounts']:
 if mount['Target']=='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash':mount['Source']=str(root/'runtime')
 elif mount['Target']=='/tmp':mount['Source']=str(root/'container-tmp')
 else:assert mount.get('ReadOnly') is True
host['Mounts'] +=[{'Type':'bind','Source':str(root/'kit'),'Target':'/wio-diagnostic','ReadOnly':True},
 {'Type':'bind','Source':str(root/'control'),'Target':'/wio-window-control','ReadOnly':True}]
spec=dict(cfg,HostConfig=host)
(root/'control/create-spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
created=api('POST','/containers/create?name='+urllib.parse.quote(plan['diagnostic_name'],safe=''),spec)
current=api('GET','/containers/'+created['Id']+'/json')
assert current['State']['Status']=='created' and current['State']['Pid']==0
assert current['Image']==d['Image'] and current['Config']['Cmd']==cfg['Cmd'] and current['HostConfig']['Mounts']==host['Mounts']
assert current['Config']['Env']==cfg['Env']
paths=[root/'controller.py',root/'cohort_policy.py',root/'control/create-spec.json',source/'scripts/serve.sh',source/'config/serve.env',root/'runtime/config/serve.env']
paths +=list((root/'kit').glob('*'))
paths +=[clone/r['relative'] for r in PAYLOAD['frozen']]
pins=[]
for p in paths:
 b=p.read_bytes();pins.append({'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
receipt={'status':'PREPARED_STOPPED_AWAITING_EXACT_FULL_MODEL_APPROVAL','run_id':plan['run_id'],'root':str(root),
 'diagnostic_container_id':created['Id'],'diagnostic_container_name':plan['diagnostic_name'],
 'diagnostic_state':'created','main_state_unchanged':api('GET','/containers/'+main+'/json')['State']['StartedAt']==d['State']['StartedAt'],
 'plan_sha256':plan_sha,'execution_pins':pins,'source_frozen_data_copied_inside_server':True,
 'model_weights_not_copied_or_uploaded':True,'new_model_or_device_execution':False,'approval_present':False}
(root/'control/PREPARED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt))
'''
result=invoke('target','PAYLOAD='+repr(payload)+'\n'+worker,120)
write(O/'shared-builtin-prepared-v164.json',result)
write(source/'LOCAL-PLAN.json',plan)
print(json.dumps({k:v for k,v in result.items() if k!='execution_pins'},ensure_ascii=False))
