"""Draft candidate stop for a necessary fresh temporal L04 C32 control; no performance rejection implied."""
import ast,concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261018T-L10-stop-entry-failed-v2-for-corrected-v3'
p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
spec=importlib.util.spec_from_file_location('inventory',p);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
p=ROOT/'config/C32-target-graph-runtime-v2/runtime.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='88b2cf0bd7b2094066f071c7b97515e50a0d566479ed2957079243cc4e4d7dec';RUNTIME=json.loads(p.read_text())
COMMON=r'''import pathlib,subprocess,json,hashlib,socket,urllib.request,time
assert socket.gethostname()==HOSTNAME
def cmd(args,timeout=30):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);assert r.returncode==0,(args,r.returncode,r.stderr);return r.stdout
def owned(cid,name):
 d=json.loads(cmd(['docker','inspect',cid]))[0]
 assert d['Id']==cid and d['Name']=='/'+name and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261018T-L10-sglang-C32-target-graph-fullmesh-v2' and not d['HostConfig']['Privileged']
 return d
def file_identity(path):
 assert path.is_file() and not path.is_symlink();h=hashlib.sha256();size=0
 with path.open('rb') as file:
  for block in iter(lambda:file.read(8*1024**2),b''):h.update(block);size+=len(block)
 return {'path':str(path),'bytes':size,'sha256':h.hexdigest()}
def idle():
 d=owned(CID,NAME);assert d['State']['Running']
 shared=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v2');supervisor=json.loads((shared/'supervisor-status.json').read_text());provider=json.loads((shared/'provider-status.json').read_text());assert provider['phase']=='READY'
 gauges=[];metric=b''
 if NODE in [34,35]:
  assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] and 'model_exit_code' not in supervisor
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open('http://172.16.12.'+str(NODE)+':30500/metrics',timeout=15) as response:metric=response.read(16*1024**2+1)
  assert len(metric)<=16*1024**2;gauges=[line for line in metric.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(float(line.rsplit(' ',1)[1])==0 for line in gauges)
 else:
  assert supervisor['phase']=='MODEL_EXITED_POOL_RETAINED' and supervisor['model_exit_code']==1
  error=(shared/'model.stderr.txt').read_text();assert 'entry.py' in error and "config['environment_intent']" in error and 'AssertionError' in error
 npu=cmd(['npu-smi','info']);rows=[]
 for line in npu.split('Process id',1)[-1].splitlines():
  cols=[x.strip() for x in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
   pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();rows.append(pid)
 assert len(rows)==(18 if NODE in [34,35] else 1),'Unexpected NPU context; preserve services'
 return {'container_id':CID,'state':d['State'],'supervisor':supervisor,'provider':provider,'idle_gauges':gauges,'metrics_sha256':hashlib.sha256(metric).hexdigest(),'owned_NPU_process_pids':rows}
'''
PREFLIGHT=r'''
for bundle in STAGED_BUNDLES:
 for relative,item in bundle['files'].items():assert hashlib.sha256((pathlib.Path(bundle['remote'])/relative).read_bytes()).hexdigest()==item['sha256']
out=idle();assert not pathlib.Path(OUTDIR).exists();out.update(node=NODE,model_requests=0,own_service_idle=True);print(json.dumps(out))
'''
STOP=r'''
before=idle();root=pathlib.Path(OUTDIR);assert not root.exists()
shared=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v2');profile=shared/'profile'
profile_preserved={'files':0,'bytes':0,'directory':str(profile),'retained_in_place':True}
assert not profile.exists() or not any(p.is_file() for p in profile.rglob('*'))
root.mkdir(parents=True)
cmd(['docker','stop','--time','240',CID],270)
after=owned(CID,NAME);assert not after['State']['Running'] and after['State']['Pid']==0
supervisor=json.loads((shared/'supervisor-status.json').read_text());provider=json.loads((shared/'provider-status.json').read_text())
assert supervisor['phase']=='STOPPED' and supervisor['provider_exit_code']==0 and provider['phase']=='CLOSE_COMPLETE' and provider['events'][-1]['rc']==0
npu=cmd(['npu-smi','info']);assert npu.count('No running processes')==8
copies={}
for directory in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']:
 destination=root/directory.rsplit('/',1)[1];r=subprocess.run(['docker','cp',CID+':'+directory,str(destination)],capture_output=True,text=True,timeout=120);copies[directory]={'rc':r.returncode,'stderr':r.stderr}
files=[file_identity(path) for path in root.rglob('*') if path.is_file() and not path.is_symlink()]
logs=[file_identity(shared/('model.'+stream+'.txt')) for stream in ['stdout','stderr']]
out={'node':NODE,'container_id':CID,'before':before,'after_state':after['State'],'supervisor':supervisor,'provider':provider,'all_NPUs_empty':True,'provider_closed0':True,'resource_shutdown_complete':True,'clean_model_shutdown':supervisor.get('model_exit_code')==0 and not supervisor.get('timeout_used') and after['State']['ExitCode']==0,'SDK_copy':copies,'preserved_SDK_files':files,'retained_complete_model_logs':logs,'old_model_and_diagnostic_files_retained':str(shared),'complete_profile_preserved':profile_preserved,'model_requests':0,'vllm_changed':False}
print(json.dumps(out))
'''
META_STOP=r'''
assert not owned(CID,NAME)['State']['Running'];meta=owned(META,'glm53-sglang-C32-target-graph-fullmesh-v2-meta-20261018-34');assert meta['State']['Running']
cmd(['docker','stop','--time','60',META],90);after=owned(META,'glm53-sglang-C32-target-graph-fullmesh-v2-meta-20261018-34');assert not after['State']['Running'] and after['State']['ExitCode']==0
root=pathlib.Path(OUTDIR)
with (root/'meta-final-stdout.txt').open('wb') as stdout,(root/'meta-final-stderr.txt').open('wb') as stderr:
 r=subprocess.run(['docker','logs',META],stdout=stdout,stderr=stderr,timeout=90);assert r.returncode==0
print(json.dumps({'container_id':META,'after_state':after['State'],'retained_meta_logs':[file_identity(root/('meta-final-'+stream+'.txt')) for stream in ['stdout','stderr']],'model_requests':0}))
'''
def render(node,body):
 item=RUNTIME['runtime_containers'][str(node)];values={'HOSTNAME':base.NODES[node][1],'CID':item['container_id'],'NAME':item['name'],'IMAGE':RUNTIME['model_image'],'NODE':node,'META':RUNTIME['meta_container_id'],'OUTDIR':'/data/wio/inference-foundry/glm53/runs/'+RID};values['STAGED_BUNDLES']=[json.loads((ROOT/path).read_text())['bundles'][0] for path in ['config/sglang-C32-target-graph-fullmesh-stage-v3.json','config/sglang-C32-target-graph-L04-rollback-stage-v1.json']];code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),COMMON+body);ast.parse(code);return code
def invoke(node,body,timeout):
 pod,_=base.NODES[node];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);(ROOT/'runs'/RID/f'mapping-{node}-{timeout}.json').write_text(json.dumps(mapped,indent=2)+'\n');assert mapped['rc']==0 and mapped['stdout']=='172.16.12.'+str(node)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);return base.invoke([command],render(node,body),timeout=timeout)
def main():
 stage=json.loads((ROOT/'runs/20261018T-L10-target-graph-stage-v3/receipt.json').read_text());assert stage['pass'] and len(stage['stages'])==4 and all(x==0 for x in stage['stages'].values()) and stage['fixed_image_CPU_qualification']['actual_entry_environment_contract_all4_pass']
 receipt_path=ROOT/'runs/20261018T-L10-target-graph-model-and-observe-v3/receipt.json';prior=json.loads(receipt_path.read_text());assert prior['models_completed'] and prior['observer_completed']
 for n in [36,41]:
  data=json.loads(json.loads((ROOT/('runs/20261018T-L10-target-graph-startup-v3/node-'+str(n)+'.json')).read_text())['stdout']);assert not data['startup_wait']['completed'] and data['supervisor_status']['model_exit_code']==1
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'precheck':{},'nodes':{},'meta_rc':None,'model_requests':0,'vllm_changed':False,'reason':'D entry declaration contradicted intended runtime before framework load; sourcecontract repaired with exactentry-loop CPU tests, no algorithm/shape/window change','max_SSH':2}
 def save(name,result):(run/name).write_text(json.dumps(result,indent=2)+'\n');return result
 try:
  for n in [34,35,36,41]:
   x=save('precheck-'+str(n)+'.json',invoke(n,PREFLIGHT,120));receipt['precheck'][str(n)]=x['rc'];assert x['rc']==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   tasks={pool.submit(invoke,n,STOP,650):n for n in [34,35,36,41]}
   for future in concurrent.futures.as_completed(tasks):
    n=tasks[future]
    try:x=future.result()
    except Exception as e:x={'error':repr(e)}
    save('node-'+str(n)+'.json',x);receipt['nodes'][str(n)]=x.get('rc',x.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
  assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values()),'Keep Meta if any owned shutdown unverified'
  x=save('meta-stop.json',invoke(34,META_STOP,180));receipt['meta_rc']=x['rc'];assert x['rc']==0
 finally:save('receipt.json',receipt)
if __name__=='__main__':main()
