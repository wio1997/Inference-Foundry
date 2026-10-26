"""Once-only idle owned SG1720 shutdown. Preserve native logs and all traces."""
import ast, concurrent.futures, hashlib, importlib.util, json, re, shlex
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1]
dependency=ROOT/'scripts/inventory_readonly.py'
assert hashlib.sha256(dependency.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',dependency);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RID='20261017T1952-stop-sglang-1720-for-metadata-v1';RUN=ROOT/'runs'/RID
OLD_RID='20261017T1720-sglang-shared-pool-v2'
OLD_ROOT='/data/wio/inference-foundry/glm53/runs/'+OLD_RID
REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
IDS={34:'10d14e531385739498eacfb08d75d6cc3b370cf78f74af14e9fae6736bbb744e',35:'eb04fad0928296da105ae2a0d90b16ef273550e17facebba56eab88ba6aa795d',36:'0d02ac7770dab23c14a55fe18b0dc8f7d5999b455926403b91227b9d95bcf20d',41:'f77216e73f51473383da9f6c7fe46e6ba40338db890ed0db656d18a032225fc6'}
META='b850ebfa32eeb44b21ea9a0cd44cee80e68a962df6276b5b77169da03289ea08'
COMMON=r'''import socket,pathlib,subprocess,json,hashlib,urllib.request,time,os
assert socket.gethostname()==HOST
def cmd(args,timeout=30):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);assert r.returncode==0,(args,r.returncode,r.stderr);return r.stdout
def owned(cid,name):
 d=json.loads(cmd(['docker','inspect',cid]))[0]
 assert d['Id']==cid and d['Name']=='/'+name and d['Image']==IMAGE
 assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==OLD_RID
 assert not d['HostConfig']['Privileged']
 return d
def idle():
 d=owned(CID,NAME);assert d['State']['Running']
 shared=pathlib.Path(OLD_ROOT)
 supervisor=json.loads((shared/'supervisor-status.json').read_text());provider=json.loads((shared/'provider-status.json').read_text())
 assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] and 'model_exit_code' not in supervisor
 assert provider['phase']=='READY'
 port=30500 if NODE in [34,35] else 30600
 with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}/metrics',timeout=15) as response:raw=response.read(16777217)
 assert len(raw)<=16777216
 rows=[l for l in raw.decode().splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
 assert rows and all(float(l.rsplit(' ',1)[1])==0 for l in rows),'Own service busy; do not stop'
 return {'container_id':CID,'state':d['State'],'supervisor':supervisor,'provider':provider,'idle_gauges':rows,'metrics_sha256':hashlib.sha256(raw).hexdigest()}
'''
PREFLIGHT=r'''
result=idle()
assert not pathlib.Path(REMOTE).exists(),'No replay/overwrite'
result.update(node=NODE,own_service_idle=True,model_requests=0)
print(json.dumps(result))
'''
STOP=r'''
before=idle();root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
# All profile artifacts remain node-side. Rename only this completed task subtree.
trace_move=None
if NODE in [36,41]:
 profile=pathlib.Path(OLD_ROOT)/'profile';source=profile/'20261017T1920-sglang-formal-C16-D-profile-v1'
 assert source.is_dir() and sorted(p.name for p in profile.iterdir())==[source.name]
 files=[p for p in source.rglob('*') if p.is_file()]
 assert len(list(source.rglob('analyse.done')))==16 and len(list(source.rglob('profiler_info_*.json')))==16
 assert len(files)=={36:7320,41:7318}[NODE]
 total=sum(p.stat().st_size for p in files);assert total=={36:41090536036,41:41114972647}[NODE]
 destination=pathlib.Path('/data/wio/inference-foundry/glm53/profile-archives')/(source.name+'-node'+str(NODE))
 assert not destination.exists();destination.parent.mkdir(parents=True,exist_ok=True)
 inode=source.stat().st_ino;device=source.stat().st_dev;source.rename(destination)
 assert destination.stat().st_ino==inode and destination.stat().st_dev==device and not list(profile.iterdir())
 trace_move={'source':str(source),'destination':str(destination),'files':len(files),'bytes':total,'same_inode':True,'no_trace_deleted':True,'prior_all32_SHA_evidence':'results/sglang-D-profile-all32-native-collection-review-v1.json'}
 (root/'profile-preservation.json').write_text(json.dumps(trace_move,indent=2)+'\n')
cmd(['docker','stop','--time','240',CID],270)
after=owned(CID,NAME);assert not after['State']['Running'] and after['State']['Pid']==0 and after['State']['ExitCode']==0,after['State']
shared=pathlib.Path(OLD_ROOT);supervisor=json.loads((shared/'supervisor-status.json').read_text());provider=json.loads((shared/'provider-status.json').read_text())
assert supervisor['phase']=='STOPPED' and supervisor['timeout_used'] is False and supervisor['provider_exit_code']==0,supervisor
assert provider['phase']=='CLOSE_COMPLETE' and provider['events'][-1]['rc']==0,provider
# The original complete model files remain at OLD_ROOT, including shutdown tail.
# Retain SDK final logs separately for the unresolved passive interval audit.
copied={}
for directory in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']:
 dst=root/directory.rsplit('/',1)[1]
 r=subprocess.run(['docker','cp',CID+':'+directory,str(dst)],capture_output=True,text=True,timeout=120)
 copied[directory]={'rc':r.returncode,'stderr':r.stderr}
manifest=[]
for p in root.rglob('*'):
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''):h.update(block)
  manifest.append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
npu=cmd(['npu-smi','info']);assert npu.count('No running processes')==8
print(json.dumps({'node':NODE,'container_id':CID,'before':before,'after_state':after['State'],'supervisor':supervisor,'provider':provider,'profile_preservation':trace_move,'SDK_copy':copied,'SDK_files':manifest,'npu_after':npu,'old_model_logs_retained':str(shared),'external_model_requests':0}))
'''
STOP_META=r'''
root=pathlib.Path(REMOTE)
assert not owned(CID,NAME)['State']['Running']
meta=owned(META,'glm53-sglang-shared-v2-meta-20261017-34');assert meta['State']['Running']
cmd(['docker','stop','--time','60',META],90)
after=owned(META,'glm53-sglang-shared-v2-meta-20261017-34');assert not after['State']['Running'] and after['State']['ExitCode']==0
raw=cmd(['docker','logs',META],60);p=root/'meta-final-stdout.txt';p.write_text(raw)
print(json.dumps({'container_id':META,'after_state':after['State'],'stdout_bytes':p.stat().st_size,'stdout_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'model_requests':0}))
'''
def render(node,body):
 values={'HOST':b.NODES[node][1],'NODE':node,'CID':IDS[node],'NAME':f'glm53-sglang-shared-v2-20261017-{node}','IMAGE':IMAGE,'OLD_RID':OLD_RID,'OLD_ROOT':OLD_ROOT,'REMOTE':REMOTE,'META':META}
 pattern=r'\b(?:'+'|'.join(values)+r')\b';code=re.sub(pattern,lambda m:repr(values[m[0]]),COMMON+body);ast.parse(code);return code
def invoke(node,body,timeout):
 pod,_=b.NODES[node];mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapping['rc']==0 and mapping['stdout']==f'172.16.12.{node}'
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],render(node,body),timeout=timeout)
def main():
 stage=json.loads((ROOT/'runs/20261017T1950-sglang-metadata-stage-v1/receipt.json').read_text())
 assert len(stage['staged'])==8 and all(x==0 for x in stage['staged'].values()) and len(stage['SDK_preflight'])==2 and all(x==0 for x in stage['SDK_preflight'].values())
 cpu=json.loads((ROOT/'results/sglang-routing-metadata-CPU-qualification-v1.json').read_text());assert cpu['pass'] and all(x['result']['pass'] for x in cpu['cases'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'precheck':{},'nodes':{},'meta_rc':None,'model_requests':0,'vllm_changed':False}
 def save(name,result):(RUN/name).write_text(json.dumps(result,indent=2)+'\n');return result
 try:
  for n in IDS:
   result=save(f'precheck-{n}.json',invoke(n,PREFLIGHT,120));receipt['precheck'][str(n)]=result['rc'];assert result['rc']==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures={pool.submit(invoke,n,STOP,600):n for n in IDS}
   for future in concurrent.futures.as_completed(futures):
    n=futures[future]
    try:result=future.result()
    except Exception as error:result={'error':repr(error)}
    save(f'node-{n}.json',result);receipt['nodes'][str(n)]=result.get('rc',result.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
  assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values()),'Preserve Meta if any model stop incomplete'
  result=save('meta-stop.json',invoke(34,STOP_META,150));receipt['meta_rc']=result['rc'];assert result['rc']==0
 finally:save('receipt.json',receipt)
if __name__=='__main__':main()
