"""Reviewed P-only lifecycle plans; all remote calls are made by exact ZCode launchers."""
import ast,hashlib,importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CURRENT=json.loads((ROOT/'config/C32-target-graph-runtime-v3/runtime.json').read_text())
STOP_RID='20261019T-L12B-stop-L10-P-owner-empty-v1'
def load(name,file,sha):
 p=ROOT/'scripts'/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def stage():return load('stage_L12B','stage_sglang_C32_packed_normal_and_rollback_v1.py','13545bf1c90368d57113dc810738da12c64d3efa5c01ba3fd38057e21cb1724f')
def bundle(kind):
 p=ROOT/('config/sglang-C32-native-HiCache-L12B-'+kind+'-stage-v1.json');expected={'candidate':'246b1c468df163a6a1c8b000af9a00104a4f500ba6e2683f2252e6dfd0e9037a','rollback':'fac91c1246f7fa0f48d512361a1a756e537a45e3d878a4d1d2792749dcc1bbe7'}[kind];assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 return json.loads(p.read_text())['bundles'][0]
def replace(code,values):
 return re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),code)
def pool(kind):
 m=load('pool_'+kind,'start_sglang_C32_target_graph_pool_v3.py','8637bad923b3c231cdf0e50616e91f4aab0a39de4673114fa5009d997e377ca3');b=bundle(kind)
 m.RID=b['runtime_run_id'];m.REMOTE='/data/wio/inference-foundry/glm53/runs/'+m.RID;m.STAGED=b['remote'];m.BUNDLE=b;m.FILES=b['files'];m.RUN=ROOT/'runs'/m.RID
 m.name=lambda n,meta=False:'glm53-L12B-'+kind+'-v1-20261019-'+str(n)
 m.invoke=lambda n,code,timeout=120:code
 if kind=='candidate':
  tree=ast.parse((ROOT/'scripts/start_sglang_C32_target_graph_pool_v3.py').read_text());f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='start');source=ast.unparse(f)
  needle="args += ['--entrypoint', '/bin/bash', IMAGE, '-c', 'set -eu; ' + boot]";assert source.count(needle)==1
  extra="args += ['-v', STAGED + '/native/libmf_hybm_accoffload.so:/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so:ro']\n    "
  source=source.replace(needle,extra+needle);exec(compile(source,'reviewed_P_optional_RO_file_mount','exec'),m.__dict__)
 return m
PREPARE=r'''import pathlib,subprocess,json,hashlib,socket
assert NODE in [34,35]
d=json.loads(subprocess.check_output(['docker','inspect',OLD],text=True))[0];assert d['Id']==OLD and not d['State']['Running'] and d['Image']==IMAGE and d['Config']['Labels']['run_id']==OLD_RID
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0 and r.stdout.count('No running processes')==8
for key,item in FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/key).read_bytes()).hexdigest()==item['sha256']
root=pathlib.Path(REMOTE);assert not root.exists();root.mkdir(parents=True)
files=[p for p in pathlib.Path('/data/wio/models/GLM-5.3-w8a8').rglob('*') if p.is_file()];assert len(files)==193 and sum(p.stat().st_size for p in files)==764240829290
available=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;assert available>1100*1024**3
print(json.dumps({'node':NODE,'empty_NPU_cards':8,'model_weights_bytes':764240829290,'MemAvailable':available,'new_runtime_root':str(root)}))
'''
def prepare_plan(kind,node):
 m=pool(kind);code=replace(PREPARE,{'NODE':node,'OLD':CURRENT['runtime_containers'][str(node)]['container_id'],'IMAGE':CURRENT['model_image'],'OLD_RID':CURRENT['shared_run_id'],'FILES':m.FILES,'STAGED':m.STAGED,'REMOTE':m.REMOTE});ast.parse(code);return code

def stop_plans():
 p=ROOT/'scripts/stop_sglang_C32_L04_for_target_graph_v1.py';s=importlib.util.spec_from_file_location('oldstop',p);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
 oldrid='20261018T-L09-sglang-C32-packed-normal-L04-rollback-v1';common=old.COMMON.replace(oldrid,CURRENT['shared_run_id']);body=old.STOP.replace(oldrid,CURRENT['shared_run_id'])
 needle="assert not profile.exists() or not any(p.is_file() for p in profile.rglob('*'))";assert body.count(needle)==1
 preservation="profile_files=[(str(p.relative_to(profile)),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(profile.rglob('*')) if p.is_file()];profile_preserved.update(files=len(profile_files),bytes=sum(x[1] for x in profile_files),metadata_sha256=hashlib.sha256(json.dumps(profile_files).encode()).hexdigest())"
 body=body.replace(needle,preservation)
 body=body.replace("copies={}\n","assert profile_files==[(str(p.relative_to(profile)),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(profile.rglob('*')) if p.is_file()]\ncopies={}\n")
 plans={}
 for n in [34,35]:
  item=CURRENT['runtime_containers'][str(n)];values={'HOSTNAME':stage().b.NODES[n][1],'CID':item['container_id'],'NAME':item['name'],'IMAGE':CURRENT['model_image'],'NODE':n,'META':CURRENT['meta_container_id'],'OUTDIR':'/data/wio/inference-foundry/glm53/runs/'+STOP_RID,'STAGED_BUNDLES':[bundle('candidate'),bundle('rollback')]}
  for key,text in [('precheck',common+old.PREFLIGHT),('stop',common+body)]:
   code=replace(text,values);ast.parse(code);plans[key+'-'+str(n)]=code
 return plans

def model_start_plan(kind,node):
 m=pool(kind)
 code=r'''import pathlib,json,subprocess,hashlib,socket,time
assert NODE in [34,35]
root=pathlib.Path(REMOTE);sup=json.loads((root/'supervisor-status.json').read_text());provider=json.loads((root/'provider-status.json').read_text());assert sup['run_id']==RID and sup['phase']=='POOL_READY_WAITING_MODEL_SIGNAL' and not sup['model_started'] and provider['phase']=='READY'
d=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0];assert d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and not d['HostConfig']['Privileged'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID
for key,item in FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/key).read_bytes()).hexdigest()==item['sha256']
assert provider['configuration']['dram_size']==({34:'136GB',35:'112GB'}[NODE]) and provider['configuration']['max_dram_size']=='160GB'
for port in [29430,30500,29898]:
 sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
assert not (root/'model.stdout.txt').exists() and not (root/'model.stderr.txt').exists()
with (root/'model-start-controller.once').open('x') as f:f.write(d['Id'])
r=subprocess.run(['docker','kill','--signal=USR1',d['Id']],capture_output=True,text=True,timeout=20);assert r.returncode==0
print(json.dumps({'node':NODE,'container_id':d['Id'],'model_start_signal_sent':True,'external_model_requests':0}))
'''
 code=replace(code,{'NODE':node,'REMOTE':m.REMOTE,'RID':m.RID,'NAME':m.name(node),'IMAGE':CURRENT['model_image'],'FILES':m.FILES,'STAGED':m.STAGED});ast.parse(code);return code

def observe_plan(kind,node):
 m=pool(kind)
 code=r'''import pathlib,subprocess,json,time,urllib.request,hashlib
root=pathlib.Path(REMOTE);opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.monotonic()+1800;last=0
while True:
 state=json.loads((root/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and 'model_exit_code' not in state,state
 healthy=False;gauges=[];events=[];warm=False;hook_pids=set()
 for stream in ['stdout','stderr']:
  p=root/('model.'+stream+'.txt')
  if p.exists():
   raw=p.read_text(errors='replace');warm=warm or ('Disaggregation warmup requests completed for all 4 DP ranks' in raw)
   for line in raw.splitlines():
    if 'GLM53_C32_NATIVE_HICACHE_INSTALLED=' in line:
     try:hook_pids.add(json.loads(line.split('GLM53_C32_NATIVE_HICACHE_INSTALLED=',1)[1])['pid'])
     except ValueError:pass
 try:
  with opener.open('http://172.16.12.'+str(NODE)+':30500/health',timeout=5) as f:healthy=f.status==200
  with opener.open('http://172.16.12.'+str(NODE)+':30500/metrics',timeout=5) as f:metric=f.read(8*1024**2+1);assert len(metric)<=8*1024**2
  gauges=[float(x.rsplit(' ',1)[1]) for x in metric.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
 except Exception:pass
 for p in (root/'audit').glob('sdma-*.jsonl'):
  raw=p.read_bytes();assert len(raw)<=16*1024**2 and (not raw or raw.endswith(b'\n'));events.extend(json.loads(line) for line in raw.splitlines())
 actual_pids={x['pid'] for x in events if x['event']=='initialize_complete'};registered={x['pid'] for x in events if x['event']=='model_state_registered'}
 native_ok=not NATIVE or len(actual_pids)==16 and actual_pids<=hook_pids
 done=healthy and bool(gauges) and not any(gauges) and len(actual_pids)==len(registered)==16 and native_ok and (NODE==35 or warm)
 if time.monotonic()-last>=30 or done:
  last=time.monotonic();print('GLM53_L12B_STARTUP_PROGRESS='+json.dumps({'node':NODE,'healthy':healthy,'idle':bool(gauges) and not any(gauges),'registered':len(registered),'native_hook_matches_actual_workers':native_ok,'warm_complete':warm,'done':done}),flush=True)
 if done:break
 assert time.monotonic()<deadline,'P startup bounded1800s exceeded'
 time.sleep(5)
d=json.loads(subprocess.check_output(['docker','inspect',NAME],text=True))[0];assert d['Image']==IMAGE and d['Config']['Labels']['run_id']==RID
print('GLM53_L12B_STARTUP_RESULT='+json.dumps({'project_id':'glm53','node':NODE,'pass':True,'container_id':d['Id'],'name':NAME,'run_id':RID,'actual_SDK_worker_PIDs':sorted(actual_pids),'registered_worker_PIDs':sorted(registered),'native_compatibility_hook_worker_PIDs':sorted(hook_pids),'external_model_requests':0,'formal_performance_qualified':False}))
'''
 code=replace(code,{'NODE':node,'REMOTE':m.REMOTE,'RID':m.RID,'NAME':m.name(node),'IMAGE':CURRENT['model_image'],'NATIVE':kind=='candidate'});ast.parse(code);return code

def plan_all():
 plans=stop_plans()
 for kind in ['candidate','rollback']:
  m=pool(kind)
  for n in [34,35]:
   for key,code in [('prepare',prepare_plan(kind,n)),('pool',m.start(n)),('preflight',m.native_preflight(n)),('start',model_start_plan(kind,n)),('observe',observe_plan(kind,n))]:ast.parse(code);plans[kind+'-'+key+'-'+str(n)]=code
 return plans
