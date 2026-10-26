"""Owned failed-candidate P shutdown and exact tested L10 P pool restore; D/Meta retained."""
import ast,hashlib,importlib.util,json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12C-reject-native-candidate-restore-L10-P-pools-v1';RUN=ROOT/'runs'/RID
STOP=r'''import pathlib,subprocess,json,hashlib,time,urllib.request
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0];assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261019T-L12C-candidate-v1' and not d['HostConfig']['Privileged']
root=pathlib.Path(REMOTE);archive=pathlib.Path(ARCHIVE);assert not archive.exists();archive.mkdir(parents=True)
sup=json.loads((root/'supervisor-status.json').read_text());assert sup['phase'] in ['MODEL_RUNNING','MODEL_EXITED_POOL_RETAINED'] and sup['run_id']=='20261019T-L12C-candidate-v1'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for n in [36,41]:
 with opener.open('http://172.16.12.'+str(n)+':30600/health',timeout=10) as f:assert f.status==200
 with opener.open('http://172.16.12.'+str(n)+':30600/metrics',timeout=10) as f:raw=f.read(8*1024**2+1);assert len(raw)<=8*1024**2
 values=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert values and not any(values)
# Assert every currently allocated NPU context belongs to this exact owned failed candidate.
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0;before=[]
for line in r.stdout.split('Process id',1)[-1].splitlines():
 cols=[x.strip() for x in line.split('|')]
 if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
  pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();before.append(pid)
r=subprocess.run(['docker','stop','--time','240',CID],capture_output=True,text=True,timeout=270);assert r.returncode==0,r.stderr
end=json.loads(subprocess.check_output(['docker','inspect',CID],text=True))[0];assert not end['State']['Running'] and end['State']['Pid']==0
sup=json.loads((root/'supervisor-status.json').read_text());provider=json.loads((root/'provider-status.json').read_text());assert sup['phase']=='STOPPED' and sup['provider_exit_code']==0 and provider['phase']=='CLOSE_COMPLETE' and provider['events'][-1]['rc']==0
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0 and r.stdout.count('No running processes')==8
scope={};exec(compile(STREAM,'stream','exec'),scope);logs={}
for stream in ['stdout','stderr']:
 p=root/('model.'+stream+'.txt')
 if p.exists():
  with p.open('rb') as f:logs[stream]=scope['summarize_log_stream'](f,archive/('model.'+stream+'.gz'))
print(json.dumps({'project_id':'glm53','node':NODE,'container_id':CID,'provider_closed0':True,'all_NPUs_empty':True,'resource_shutdown_complete':True,'supervisor':sup,'provider':provider,'retained_complete_logs':logs,'old_candidate_namespace_retained':str(root),'model_requests':0,'D_Meta_retained':True,'NPU_context_PIDs_before':before}))
'''
def plan():
 p=ROOT/'scripts/glm53_C32_L12C_P_transition_library_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='ac856e9ead23a242bf07466a2a6b5899cae67f43bc377c34e1e3ec106c7f87d1';s=importlib.util.spec_from_file_location('L12B',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);plans=m.plan_all()
 cap=json.loads((ROOT/'results/sglang-C32-L12C-selected-P-pool-capacity-v1.json').read_text());assert cap['selected_variant']=='candidate'
 stream=ROOT/'scripts/glm53_stream_C32_packed_native_logs_v1.py';assert hashlib.sha256(stream.read_bytes()).hexdigest()=='e03a299a23f50ee7d7fc7233c0f1cbb221043d8fde62c0cd3e5b04ea55cc5abc'
 for n in [34,35]:
  item=cap['providers'][str(n)];code=m.replace(STOP,{'CID':item['container_id'],'NAME':item['name'],'IMAGE':m.CURRENT['model_image'],'REMOTE':item['remote_root'],'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+RID,'NODE':n,'STREAM':stream.read_text()});ast.parse(code);plans['stop-candidate-'+str(n)]=code
 p=ROOT/'scripts/run_C32_L12C_controlled_P_transition_and_pool_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='007eb50f1a1a63292cf53bf3ebd8e8eb70d0e879177893d20a20f857f5c27157';s=importlib.util.spec_from_file_location('prior_transition',p);prior=importlib.util.module_from_spec(s);s.loader.exec_module(prior)
 for phase in ['before','after']:
  plans['rollback-meta-'+phase]=m.replace(prior.META,{'META_ID':cap['meta_container_id'],'IMAGE':m.CURRENT['model_image'],'SHARED':m.CURRENT['shared_run_id'],'OUTPUT':'/data/wio/inference-foundry/glm53/runs/'+RID+'-meta-'+phase});ast.parse(plans['rollback-meta-'+phase])
 return m,plans

def main():
 m,plans=plan();review=json.loads((ROOT/'results/C32-L12C-failed-candidate-P-restore-actual-review-v1.json').read_text());assert review['host_plans_sha256']=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in plans.items()}
 assert json.loads((ROOT/'results/C32-L12C-staged-candidate-and-tested-L10-rollback-v1.json').read_text())['pass']
 RUN.mkdir(exist_ok=False);(RUN/'execution-once.lock').write_text(str(os.getpid()));base=m.stage();receipt={'project_id':'glm53','max_additional_SSH':1,'serial':True,'selected_variant':'rollback','D_Meta_retained':True,'model_requests':0,'models_started':False,'stops':{},'pool':{},'preflight':{}}
 def save(n,o):(RUN/n).write_text(json.dumps(o,indent=2)+'\n')
 def call(n,key,timeout=180):
  r=base.invoke(n,plans[key],timeout);save(key+'.json',r);print(key,r['rc'],flush=True);assert r['rc']==0,(key,r.get('stderr'));return json.loads(r['stdout'])
 try:
  call(34,'rollback-meta-before',180)
  for n in [34,35]:
   d=call(n,'stop-candidate-'+str(n),650);assert d['resource_shutdown_complete'];receipt['stops'][str(n)]=0;save('receipt.json',receipt)
  for n in [34,35]:call(n,'rollback-prepare-'+str(n))
  for n in [34,35]:
   d=call(n,'rollback-pool-'+str(n),260);assert d['ready'];receipt['pool'][str(n)]=0;save('receipt.json',receipt)
  for n in [34,35]:
   d=call(n,'rollback-preflight-'+str(n),260);assert d['query_rc']==0;receipt['preflight'][str(n)]=0;save('receipt.json',receipt)
  call(34,'rollback-meta-after',180);receipt['pass']=True;receipt['phase']='ROLLBACK_P_POOLS_READY_AWAIT_INDEPENDENT_CAPACITY_AUDIT'
 finally:save('receipt.json',receipt)
 print(json.dumps(receipt))
if __name__=='__main__':main()
