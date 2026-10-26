"""Exact-once P-only controlled owner-empty native fixture and selected pool preparation."""
import ast,concurrent.futures,hashlib,importlib.util,json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12B-controlled-P-owner-empty-and-pool-v1';RUN=ROOT/'runs'/RID
LIB_SHA='0f59ff934fa11980d17b590948250b22baffb088cbbe4b2daab2493e6545d146'
def loadlib():
 p=ROOT/'scripts/glm53_C32_L12B_P_transition_library_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==LIB_SHA;s=importlib.util.spec_from_file_location('L12B',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
META=r'''import pathlib,subprocess,json,hashlib,re
r=subprocess.run(['docker','inspect',META_ID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0];assert d['Id']==META_ID and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED and d['State']['Running']
root=pathlib.Path(OUTPUT);assert not root.exists();root.mkdir(parents=True)
r=subprocess.run(['docker','cp',META_ID+':/var/log/memcache_hybrid',str(root/'memcache_hybrid')],capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr
files=[];critical=[]
for p in sorted((root/'memcache_hybrid').rglob('*')):
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<=256*1024**2
  selected=[line for line in raw.decode(errors='replace').splitlines() if re.search(r'Mount bm on loc|Unmount|UnMount|ock\.mmc\.meta\.lease_ttl_ms|ock\.mmc\.evict_threshold_',line,re.I)]
  files.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'critical_lines':selected});critical+=selected
print(json.dumps({'project_id':'glm53','container_id':META_ID,'Meta_unchanged':True,'files':files,'critical_lines':critical,'model_requests':0}))
'''
def fixture_plan(m):
 p=ROOT/'scripts/run_sglang_C32_native_hicache_NPU_fixture_v6.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='b47fb1d72fdfe2438f79ac8f37253e93c82501053e77f21a8c64b1867e86e097';s=importlib.util.spec_from_file_location('oldfixture',p);v6=importlib.util.module_from_spec(s);s.loader.exec_module(v6);_,values=v6.plan()
 prefix=r'''import socket,pathlib,subprocess,json,time,re,urllib.request,hashlib
assert socket.gethostname()==HOSTNAME
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Image']==IMAGE and d['Name']=='/'+NAME and not d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));idle={}
for n in [36,41]:
 url='http://172.16.12.'+str(n)+':30600'
 with opener.open(url+'/health',timeout=10) as f:assert f.status==200
 with opener.open(url+'/metrics',timeout=10) as f:raw=f.read(8*1024**2+1);assert len(raw)<=8*1024**2
 gauges=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges);idle[str(n)]=gauges
def owned():
 p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert p.returncode==0;rows=[]
 for line in p.stdout.split('Process id',1)[-1].splitlines():
  cols=[x.strip() for x in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():rows.append(int(cols[2]))
 return sorted(rows)
before=owned();assert before==[]
'''
 body=v6.HOST[v6.HOST.index('extension_source='):]
 body=body.replace("'--pid=container:'+CID,",'')
 body=body.replace('20261018T-L12-C32-native-HiCache-NPU-fixture-v6','20261019T-L12B-C32-native-owner-empty-fixture-v1').replace('glm53-C32-native-HiCache-fixture-v6-20261018-34','glm53-L12B-native-owner-empty-v1-20261019-34')
 code=m.replace(prefix+body,values);ast.parse(code);return code

def plan():
 m=loadlib();plans=m.plan_all();plans['fixture-owner-empty-34']=fixture_plan(m)
 for phase in ['before','after']:
  plans['meta-'+phase]=m.replace(META,{'META_ID':m.CURRENT['meta_container_id'],'IMAGE':m.CURRENT['model_image'],'SHARED':m.CURRENT['shared_run_id'],'OUTPUT':'/data/wio/inference-foundry/glm53/runs/'+RID+'-meta-'+phase})
  ast.parse(plans['meta-'+phase])
 # Retained D health ownership plans are the exact already executed read-only stage checks.
 p=ROOT/'scripts/stage_C32_native_hicache_L12B_and_rollback_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='8ca918a06f4d88c80e3b346526dc01cc923b2840db48ae3690af7661483e0309';s=importlib.util.spec_from_file_location('stagecheck',p);stage=importlib.util.module_from_spec(s);s.loader.exec_module(stage);_,_,health=stage.plan()
 for n in [34,35,36,41]:plans['health-'+str(n)]=health['health-'+str(n)]
 return m,plans

def main():
 m,plans=plan();review=json.loads((ROOT/'results/C32-L12B-controlled-P-transition-actual-review-v1.json').read_text());assert review['host_plans_sha256']=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in plans.items()}
 assert json.loads((ROOT/'results/C32-L12B-staged-candidate-and-tested-L10-rollback-v1.json').read_text())['pass']
 RUN.mkdir(exist_ok=False);(RUN/'execution-once.lock').write_text(str(os.getpid()));(ROOT/'runs'/m.STOP_RID).mkdir(exist_ok=False)
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'max_SSH':2,'model_requests':0,'D_Meta_vllm_retained':True,'phase':'PREFLIGHT','P_stopped':{},'selected_variant':None,'pool':{},'preflight':{},'models_started':False}
 base=m.stage()
 def save(name,obj):(RUN/name).write_text(json.dumps(obj,indent=2)+'\n');return obj
 def invoke(n,key,timeout=150):
  result=save(key+'.json',base.invoke(n,plans[key],timeout));assert result['rc']==0,(key,result.get('stderr'));return json.loads(result['stdout'])
 def batch(keys,timeout=150):
  values={}
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs={pool.submit(base.invoke,n,plans[key],timeout):(n,key) for n,key in keys}
   for f in concurrent.futures.as_completed(jobs):
    n,key=jobs[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    save(key+'.json',r);values[str(n)]=r;print(key,r.get('rc',r.get('error')),flush=True)
  return values
 def checkpoint():save('receipt.json',receipt)
 try:
  for n in [34,35,36,41]:invoke(n,'health-'+str(n))
  for n in [34,35]:invoke(n,'precheck-'+str(n))
  invoke(34,'meta-before',180);receipt['phase']='STOPPING_P';checkpoint()
  rows=batch([(n,'stop-'+str(n)) for n in [34,35]],650)
  receipt['P_stopped']={n:r.get('rc',r.get('error')) for n,r in rows.items()};checkpoint()
  assert all(r.get('rc')==0 and json.loads(r['stdout'])['resource_shutdown_complete'] for r in rows.values()),'Preserve stopped/remaining resources; no broad kill'
  receipt['phase']='OWNER_EMPTY_NATIVE_FIXTURE';checkpoint();native=invoke(34,'fixture-owner-empty-34',360)
  receipt['native_fixture_pass']=native['pass'];receipt['native_fixture_rc']=native['fixture_rc'];receipt['selected_variant']='candidate' if native['pass'] else 'rollback';kind=receipt['selected_variant'];receipt['phase']='PREPARING_'+kind.upper()+'_POOLS';checkpoint()
  for n in [34,35]:invoke(n,kind+'-prepare-'+str(n))
  rows=batch([(n,kind+'-pool-'+str(n)) for n in [34,35]],260);receipt['pool']={n:r.get('rc',r.get('error')) for n,r in rows.items()};checkpoint();assert all(r.get('rc')==0 and json.loads(r['stdout'])['ready'] for r in rows.values())
  rows=batch([(n,kind+'-preflight-'+str(n)) for n in [34,35]],260);receipt['preflight']={n:r.get('rc',r.get('error')) for n,r in rows.items()};checkpoint();assert all(r.get('rc')==0 and json.loads(r['stdout'])['query_rc']==0 for r in rows.values())
  invoke(34,'meta-after',180)
  for n in [36,41]:
   r=save('retained-D-health-'+str(n)+'.json',base.invoke(n,plans['health-'+str(n)],150));assert r['rc']==0
  receipt['phase']='POOLS_READY_AWAITING_INDEPENDENT_CAPACITY_AUDIT';receipt['pass']=True;checkpoint()
 finally:checkpoint()
 print(json.dumps(receipt))
if __name__=='__main__':main()
