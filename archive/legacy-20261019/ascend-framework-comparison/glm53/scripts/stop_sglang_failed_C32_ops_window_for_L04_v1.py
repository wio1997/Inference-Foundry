"""Once-only idle owned L04 shutdown, preserving all evidence and timeout status."""
import ast,concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261018T-L08-stop-failed-window-for-L04-v1'
p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
spec=importlib.util.spec_from_file_location('inventory',p);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
p=ROOT/'config/C32-ops-window-runtime-v2/runtime.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='19f1fa64ce0d55390488ff6f6da236779dcba4086e0a7740256dd56def5d8e67';RUNTIME=json.loads(p.read_text())
COMMON=r'''import pathlib,subprocess,json,hashlib,socket,urllib.request,time
assert socket.gethostname()==HOSTNAME
def cmd(args,timeout=30):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);assert r.returncode==0,(args,r.returncode,r.stderr);return r.stdout
def owned(cid,name):
 d=json.loads(cmd(['docker','inspect',cid]))[0]
 assert d['Id']==cid and d['Name']=='/'+name and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261018T-L08-sglang-C32-ops-window-v2' and not d['HostConfig']['Privileged']
 return d
def file_identity(path):
 assert path.is_file() and not path.is_symlink();h=hashlib.sha256();size=0
 with path.open('rb') as file:
  for block in iter(lambda:file.read(8*1024**2),b''):h.update(block);size+=len(block)
 return {'path':str(path),'bytes':size,'sha256':h.hexdigest()}
def idle():
 d=owned(CID,NAME);assert d['State']['Running']
 shared=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L08-sglang-C32-ops-window-v2');supervisor=json.loads((shared/'supervisor-status.json').read_text());provider=json.loads((shared/'provider-status.json').read_text());assert supervisor['model_started'] and provider['phase']=='READY'
 if NODE in [34,35]:assert supervisor['phase']=='MODEL_RUNNING' and 'model_exit_code' not in supervisor
 else:assert supervisor['phase'] in ['MODEL_RUNNING','MODEL_EXITED_POOL_RETAINED']
 if NODE in [36,41]:
  text=(shared/'model.stderr.txt').read_text(errors='replace');assert 'aclnnMoeDistributeDispatchV4 failed, error code is 207001' in text
  assert 'GLM53_C32_OPS_FIRST_USE=' not in text or '_npu_low_latency_combine' not in '\n'.join(l for l in text.splitlines() if 'GLM53_C32_OPS_FIRST_USE=' in l)
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));port=30500 if NODE in [34,35] else 30600
 gauges=[]
 if NODE in [34,35]:
  with opener.open('http://172.16.12.'+str(NODE)+':'+str(port)+'/metrics',timeout=15) as response:raw=response.read(16*1024**2+1)
  assert len(raw)<=16*1024**2;gauges=[line for line in raw.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(float(line.rsplit(' ',1)[1])==0 for line in gauges)
 else:raw=b'Model failed during native startup; no external model requests dispatched'
 npu=cmd(['npu-smi','info']);rows=[]
 for line in npu.split('Process id',1)[-1].splitlines():
  cols=[x.strip() for x in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
   pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();rows.append(pid)
 assert len(rows)==18 if NODE in [34,35] else len(rows)>=1,'Unexpected NPU context; preserve services'
 return {'container_id':CID,'state':d['State'],'supervisor':supervisor,'provider':provider,'idle_gauges':gauges,'metrics_sha256':hashlib.sha256(raw).hexdigest(),'owned_NPU_process_pids':rows}
'''
PREFLIGHT=r'''
for bundle in STAGED_BUNDLES:
 for relative,item in bundle['files'].items():assert hashlib.sha256((pathlib.Path(bundle['remote'])/relative).read_bytes()).hexdigest()==item['sha256']
out=idle();assert not pathlib.Path(OUTDIR).exists();out.update(node=NODE,model_requests=0,own_service_idle=True);print(json.dumps(out))
'''
STOP=r'''
before=idle();root=pathlib.Path(OUTDIR);assert not root.exists()
shared=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L08-sglang-C32-ops-window-v2');profile=shared/'profile'
profile_preserved={'files':0,'bytes':0,'directory':str(profile),'retained_in_place':True}
assert not profile.exists() or not any(p.is_file() for p in profile.rglob('*'))
root.mkdir(parents=True)
source_capture=None
if False:
 source_result=subprocess.run(['docker','exec',CID,'env','PYTHONDONTWRITEBYTECODE=1','python3','-c',NATIVE_SOURCE_PROBE],capture_output=True,text=True,timeout=90)
 source_capture={'rc':source_result.returncode,'stderr':source_result.stderr}
 if source_result.returncode==0:source_capture['data']=json.loads(source_result.stdout)
 else:source_capture['stdout']=source_result.stdout
 (root/'native-failure-group-sources.json').write_text(json.dumps(source_capture,indent=2)+'\n')
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
out={'node':NODE,'container_id':CID,'before':before,'after_state':after['State'],'supervisor':supervisor,'provider':provider,'all_NPUs_empty':True,'provider_closed0':True,'resource_shutdown_complete':True,'clean_model_shutdown':supervisor.get('model_exit_code')==0 and not supervisor.get('timeout_used') and after['State']['ExitCode']==0,'SDK_copy':copies,'preserved_SDK_files':files,'retained_complete_model_logs':logs,'old_model_and_diagnostic_files_retained':str(shared),'complete_profile_preserved':profile_preserved,'model_requests':0,'vllm_changed':False,'native_source_capture':source_capture}
print(json.dumps(out))
'''
META_STOP=r'''
assert not owned(CID,NAME)['State']['Running'];meta=owned(META,'glm53-sglang-C32-ops-window-v2-meta-20261018-34');assert meta['State']['Running']
cmd(['docker','stop','--time','60',META],90);after=owned(META,'glm53-sglang-C32-ops-window-v2-meta-20261018-34');assert not after['State']['Running'] and after['State']['ExitCode']==0
root=pathlib.Path(OUTDIR)
with (root/'meta-final-stdout.txt').open('wb') as stdout,(root/'meta-final-stderr.txt').open('wb') as stderr:
 r=subprocess.run(['docker','logs',META],stdout=stdout,stderr=stderr,timeout=90);assert r.returncode==0
print(json.dumps({'container_id':META,'after_state':after['State'],'retained_meta_logs':[file_identity(root/('meta-final-'+stream+'.txt')) for stream in ['stdout','stderr']],'model_requests':0}))
'''
def render(node,body):
 item=RUNTIME['runtime_containers'][str(node)];values={'HOSTNAME':base.NODES[node][1],'CID':item['container_id'],'NAME':item['name'],'IMAGE':RUNTIME['model_image'],'NODE':node,'META':RUNTIME['meta_container_id'],'OUTDIR':'/data/wio/inference-foundry/glm53/runs/'+RID};values['NATIVE_SOURCE_PROBE']="import pathlib,hashlib,json,sys\nroot=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt');site=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages')\npaths=[root/'speculative/eagle_worker_v2.py',root/'speculative/eagle_worker.py',root/'speculative/eagle_utils.py',root/'distributed/parallel_state.py',root/'hardware_backend/npu/utils.py',site/'torch_npu/distributed/distributed_c10d.py',site/'deep_ep/buffer.py']\nout=[];total=0\nfor p in paths:\n if not p.is_file():out.append({'path':str(p),'missing':True});continue\n raw=p.read_bytes();total+=len(raw);assert len(raw)<=2*1024**2 and total<=8*1024**2\n out.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'source':raw.decode()})\nassert not any(n.startswith(('torch','sglang','memcache_hybrid','memfabric_hybrid')) for n in sys.modules)\nprint(json.dumps({'project_id':'glm53','files':out,'framework_imports':False,'SDK_init_calls':0,'model_requests':0}))\n";values['STAGED_BUNDLES']=[json.loads((ROOT/path).read_text())['bundles'][0] for path in ['config/sglang-C32-ops-window-v2-stage-v2.json','config/sglang-C32-ops-window-L04-rollback-v2-stage-v2.json']];values['PROFILE_MANIFEST']=json.loads(json.loads((ROOT/('runs/20261017T2060-sglang-C32-D-two-step-native-v1/node-'+str(node)+'.json')).read_text())['stdout']).get('native_profile',{});code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),COMMON+body);ast.parse(code);return code
def invoke(node,body,timeout):
 pod,_=base.NODES[node];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);(ROOT/'runs'/RID/f'mapping-{node}-{timeout}.json').write_text(json.dumps(mapped,indent=2)+'\n');assert mapped['rc']==0 and mapped['stdout']=='172.16.12.'+str(node)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);return base.invoke([command],render(node,body),timeout=timeout)
def main():
 snapshot=ROOT/'runs/20261018T-L08-window-failure-snapshot-v1';receipt=json.loads((snapshot/'receipt.json').read_text());assert len(receipt['nodes'])==4 and all(v==0 for v in receipt['nodes'].values())
 for node in [36,41]:
  wire=json.loads((snapshot/f'node-{node}.json').read_text());assert wire['rc']==0;o=json.loads(wire['stdout']);assert o['container_id']==RUNTIME['runtime_containers'][str(node)]['container_id'] and o['model_requests']==0
  assert o['health'].get('status')!=200
  lines=[x['text'] for stream in ['stdout','stderr'] for x in o['logs'][stream]['error_contexts']]
  assert any('aclnnMoeDistributeDispatchV4 failed, error code is 207001' in line for line in lines)
  markers=[json.loads(line.split('GLM53_C32_OPS_FIRST_USE=',1)[1]) for stream in ['stdout','stderr'] for line in o['logs'][stream]['Ops_markers'] if 'GLM53_C32_OPS_FIRST_USE=' in line]
  assert len(markers)==16 and all(x['method']=='_npu_low_latency_dispatch' and x['original_method_returned'] for x in markers)
 for path in ['20261018T-L08-sglang-window-C32-functional-v2','20261018T-L08-sglang-window-long-v2','20261018T-L08-sglang-formal-C32-ops-window-v2']:
  assert not (ROOT/'runs'/path/'execution-once.lock').exists()
 stage=json.loads((ROOT/'runs/20261018T-L08-sglang-window-stage-v2/receipt.json').read_text());assert len(stage['staged'])==8 and all(v==0 for v in stage['staged'].values())
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'precheck':{},'nodes':{},'meta_rc':None,'model_requests':0,'vllm_changed':False,'max_SSH':1,'reason':'Native207001/unhealthy before external requests; deliberate cancellation of failed startup; not natural startup timeout'}
 def save(name,result):(run/name).write_text(json.dumps(result,indent=2)+'\n');return result
 try:
  for node in (34,35,36,41):
   result=save(f'precheck-{node}.json',invoke(node,PREFLIGHT,120));receipt['precheck'][str(node)]=result['rc'];assert result['rc']==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
   pending={pool.submit(invoke,node,STOP,650):node for node in (34,35,36,41)}
   for future in concurrent.futures.as_completed(pending):
    node=pending[future]
    try:result=future.result()
    except Exception as error:result={'error':repr(error)}
    save(f'node-{node}.json',result);receipt['nodes'][str(node)]=result.get('rc',result.get('error'));print(node,receipt['nodes'][str(node)],flush=True)
  assert len(receipt['nodes'])==4 and all(value==0 for value in receipt['nodes'].values()),'Keep Meta if any owned resource shutdown unverified'
  result=save('meta-stop.json',invoke(34,META_STOP,180));receipt['meta_rc']=result['rc'];assert result['rc']==0
 finally:save('receipt.json',receipt)
if __name__=='__main__':main()
