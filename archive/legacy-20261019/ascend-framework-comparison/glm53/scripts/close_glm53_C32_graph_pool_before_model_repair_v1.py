"""Close exactowned pre-model v1 pool after v2 contract fix has been staged."""
from pathlib import Path
import ast,concurrent.futures,hashlib,importlib.util,json,os,re,shlex
ROOT=Path(__file__).resolve().parents[1];RID='20261018T-L10-close-pre-model-pool-v1'
p=ROOT/'results/sglang-C32-target-graph-fullmesh-pool-capacity-v1.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='ae11f4d6bb7393097d876d677fe603a3b6af75fe1d4a8fa0418bb9f4c93d4875';CAP=json.loads(p.read_text());assert CAP['pass'] and not CAP['model_started']
p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777';spec=importlib.util.spec_from_file_location('inventory',p);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
COMMON=r'''import pathlib,subprocess,json,hashlib,socket
assert socket.gethostname()==HOSTNAME
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v1')
def cmd(args,timeout=30):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def owned(cid,name):
 r=cmd(['docker','inspect',cid]);assert r['rc']==0;d=json.loads(r['stdout'])[0]
 assert d['Id']==cid and d['Name']=='/'+name and d['Image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263' and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261018T-L10-sglang-C32-target-graph-fullmesh-v1'
 return d
def before():
 d=owned(CID,NAME);assert d['State']['Running'];s=json.loads((root/'supervisor-status.json').read_text());p=json.loads((root/'provider-status.json').read_text());assert s['phase']=='POOL_READY_WAITING_MODEL_SIGNAL' and s['model_started'] is False and p['phase']=='READY'
 assert not (root/'model.stdout.txt').exists() and not (root/'model.stderr.txt').exists() and not (root/'model-start-controller.once').exists()
 n=cmd(['npu-smi','info']);assert n['rc']==0
 for line in n['stdout'].split('Process id',1)[-1].splitlines():
  cols=[v.strip() for v in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(v.isdigit() for v in cols[1].split()) and cols[2].isdigit():assert CID in (pathlib.Path('/proc')/cols[2]/'cgroup').read_text(),'Foreign NPU workload; preserve pool'
 return {'container_state':d['State'],'supervisor':s,'provider':p,'models_never_started':True}
'''
PRE=r"""print(json.dumps(before()))"""
STOP=r'''
pre=before();archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-close-pre-model-pool-v1');archive.mkdir(exist_ok=False)
r=cmd(['docker','stop','--time','240',CID],270);assert r['rc']==0
state=owned(CID,NAME)['State'];assert not state['Running'] and state['Pid']==0
s=json.loads((root/'supervisor-status.json').read_text());p=json.loads((root/'provider-status.json').read_text());assert s['phase']=='STOPPED' and not s['model_started'] and s['provider_exit_code']==0 and p['phase']=='CLOSE_COMPLETE' and p['events'][-1]['rc']==0
n=cmd(['npu-smi','info']);assert n['rc']==0 and n['stdout'].count('No running processes')==8
copies={directory:cmd(['docker','cp',CID+':'+directory,str(archive/directory.rsplit('/',1)[-1])],120) for directory in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']}
print(json.dumps({'node':NODE,'container_id':CID,'before':pre,'after_state':state,'supervisor':s,'provider':p,'provider_closed0':True,'all_NPUs_empty':True,'resource_shutdown_complete':True,'models_never_started':True,'SDK_copy':copies,'external_model_requests':0}))
'''
META_STOP=r'''
assert not owned(CID,NAME)['State']['Running'];d=owned(META,'glm53-sglang-C32-target-graph-fullmesh-v1-meta-20261018-34');assert d['State']['Running']
r=cmd(['docker','stop','--time','60',META],90);assert r['rc']==0;d=owned(META,'glm53-sglang-C32-target-graph-fullmesh-v1-meta-20261018-34');assert not d['State']['Running'] and d['State']['ExitCode']==0
logs=cmd(['docker','logs',META],60);assert logs['rc']==0
for k in ['stdout','stderr']:(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-close-pre-model-pool-v1')/('meta-final-'+k+'.txt')).write_text(logs[k])
print(json.dumps({'container_id':META,'state_after':d['State'],'external_model_requests':0}))
'''
def render(n,body):
 item=CAP['providers'][str(n)];v={'HOSTNAME':b.NODES[n][1],'CID':item['container_id'],'NAME':item['name'],'NODE':n,'META':json.loads(CAP['meta']['stdout'])['container_id']};code=re.sub(r'\b(?:'+'|'.join(v)+r')\b',lambda m:repr(v[m[0]]),COMMON+body);ast.parse(code);return code
def invoke(n,body,timeout=180):
 pod=b.NODES[n][0];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.'+str(n)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);return b.invoke([command],render(n,body),timeout=timeout)
def main():
 stage=json.loads((ROOT/'runs/20261018T-L10-target-graph-stage-v2/receipt.json').read_text());assert stage['pass'] and len(stage['stages'])==4 and all(x==0 for x in stage['stages'].values()) and stage['fixed_image_CPU_qualification']['opaque_aux_None_and_Tensor_passed_unchanged']
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','precheck':{},'nodes':{},'meta_rc':None,'model_requests':0,'models_never_started':True}
 def save(n,v):(run/n).write_text(json.dumps(v,indent=2)+'\n');return v
 try:
  for n in [34,35,36,41]:
   x=save('precheck-'+str(n)+'.json',invoke(n,PRE));receipt['precheck'][str(n)]=x['rc'];assert x['rc']==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   tasks={pool.submit(invoke,n,STOP,500):n for n in [34,35,36,41]}
   for f in concurrent.futures.as_completed(tasks):
    n=tasks[f]
    try:x=f.result()
    except Exception as e:x={'error':repr(e)}
    save('node-'+str(n)+'.json',x);receipt['nodes'][str(n)]=x.get('rc',x.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
  assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values())
  x=save('meta-stop.json',invoke(34,META_STOP));receipt['meta_rc']=x['rc'];assert x['rc']==0
 finally:save('receipt.json',receipt)
if __name__=='__main__':main()
