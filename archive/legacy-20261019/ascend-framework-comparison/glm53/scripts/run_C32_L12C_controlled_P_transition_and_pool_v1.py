"""Exact-once P-only guarded mapping-reuse candidate pool transition; tested rollback staged."""
import ast,concurrent.futures,hashlib,importlib.util,json,os,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12C-controlled-P-transition-and-pool-v1';RUN=ROOT/'runs'/RID
LIB_SHA='ac856e9ead23a242bf07466a2a6b5899cae67f43bc377c34e1e3ec106c7f87d1'
def loadlib():
 p=ROOT/'scripts/glm53_C32_L12C_P_transition_library_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==LIB_SHA;s=importlib.util.spec_from_file_location('L12B',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
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
def plan():
 m=loadlib();plans=m.plan_all()
 for phase in ['before','after']:
  plans['meta-'+phase]=m.replace(META,{'META_ID':m.CURRENT['meta_container_id'],'IMAGE':m.CURRENT['model_image'],'SHARED':m.CURRENT['shared_run_id'],'OUTPUT':'/data/wio/inference-foundry/glm53/runs/'+RID+'-meta-'+phase})
  ast.parse(plans['meta-'+phase])
 # Retained D health ownership plans are the exact already executed read-only stage checks.
 p=ROOT/'scripts/stage_C32_native_hicache_L12C_and_rollback_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='2640e35f1db969f27c8005aded932404fb9263e2319bdc57100a33f3f49ab736';s=importlib.util.spec_from_file_location('stagecheck',p);stage=importlib.util.module_from_spec(s);s.loader.exec_module(stage);_,_,health=stage.plan()
 for n in [34,35,36,41]:plans['health-'+str(n)]=health['health-'+str(n)]
 return m,plans

def main():
 m,plans=plan();review=json.loads((ROOT/'results/C32-L12C-controlled-P-transition-actual-review-v1.json').read_text());assert review['host_plans_sha256']=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in plans.items()}
 assert json.loads((ROOT/'results/C32-L12C-staged-candidate-and-tested-L10-rollback-v1.json').read_text())['pass']
 assert json.loads((ROOT/'results/C32-L12C-combined-MC-native-mapping-reuse-qualified-v1.json').read_text())['pass']
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
  receipt['combined_native_MRE_qualified']='results/C32-L12C-combined-MC-native-mapping-reuse-qualified-v1.json';receipt['selected_variant']='candidate';kind='candidate';receipt['phase']='PREPARING_'+kind.upper()+'_POOLS';checkpoint()
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
