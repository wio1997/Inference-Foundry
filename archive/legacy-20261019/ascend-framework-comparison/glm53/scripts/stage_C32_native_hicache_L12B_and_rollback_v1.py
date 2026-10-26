"""Exact-once stage of private candidate and tested L10 rollback; no service change."""
import ast,base64,concurrent.futures,hashlib,importlib.util,json,os,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12B-stage-and-live-audit-v1';RUN=ROOT/'runs'/RID
MANIFESTS={'config/sglang-C32-native-HiCache-L12B-candidate-stage-v1.json':'246b1c468df163a6a1c8b000af9a00104a4f500ba6e2683f2252e6dfd0e9037a','config/sglang-C32-native-HiCache-L12B-rollback-stage-v1.json':'fac91c1246f7fa0f48d512361a1a756e537a45e3d878a4d1d2792749dcc1bbe7'}
def load():
 p=ROOT/'scripts/stage_sglang_C32_packed_normal_and_rollback_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='13545bf1c90368d57113dc810738da12c64d3efa5c01ba3fd38057e21cb1724f';s=importlib.util.spec_from_file_location('stage',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def bundles():
 out=[]
 for p,h in MANIFESTS.items():
  q=ROOT/p;assert hashlib.sha256(q.read_bytes()).hexdigest()==h;out+=json.loads(q.read_text())['bundles']
 for b in out:
  for v in b['files'].values():assert hashlib.sha256((ROOT/v['local']).read_bytes()).hexdigest()==v['sha256']
 return out
HEALTH=r'''import pathlib,subprocess,json,urllib.request,hashlib
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED and d['State']['Running'] and not d['HostConfig']['Privileged']
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs')/SHARED;s=json.loads((root/'supervisor-status.json').read_text());p=json.loads((root/'provider-status.json').read_text());assert s['phase']=='MODEL_RUNNING' and 'model_exit_code' not in s and p['phase']=='READY'
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url='http://172.16.12.'+str(NODE)+(':'+str(30500 if NODE in [34,35] else 30600))
with opener.open(url+'/health',timeout=10) as f:assert f.status==200
with opener.open(url+'/metrics',timeout=10) as f:raw=f.read(8*1024**2+1);assert len(raw)<=8*1024**2
values=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert values and all(x==0 for x in values)
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0;rows=[]
for line in r.stdout.split('Process id',1)[-1].splitlines():
 cols=[x.strip() for x in line.split('|')]
 if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
  pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();rows.append(pid)
assert len(rows)==18
print(json.dumps({'node':NODE,'container_id':CID,'healthy_idle':True,'NPU_context_PIDs':sorted(rows),'metrics_sha256':hashlib.sha256(raw).hexdigest(),'model_requests':0,'existing_models_changed':False}))
'''
BINARY=r'''import pathlib,base64,hashlib,json
p=pathlib.Path(REMOTE)/RELATIVE;assert not p.exists();raw=base64.b64decode(DATA);assert len(raw)==718752 and hashlib.sha256(raw).hexdigest()==SHA
p.parent.mkdir(exist_ok=False);p.write_bytes(raw);assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
print(json.dumps({'node':NODE,'private_optional_artifact_staged':True,'sha256':SHA,'bytes':len(raw),'SDK_files_changed':False}))
'''
CPU=r'''import ast,hashlib,importlib.util,json,os,sys
from pathlib import Path
root=Path('/opt/glm53/staged')
p=root/'python/qualify_image_cpu_v1.py';s=importlib.util.spec_from_file_location('unchanged_D_qual',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main()
assert not any(k.startswith(('torch','sglang','memfabric_hybrid','memcache_hybrid')) for k in sys.modules)
source=Path('/sgl-workspace/sglang/python/sglang/srt/mem_cache/pool_host/mla.py');assert hashlib.sha256(source.read_bytes()).hexdigest()=='4009474db3858a74ca89e314bb4ea3a7840f0ce884d7c422fcb4fe6131ef5041'
mem=source.with_name('npu_memfabric.py');assert hashlib.sha256(mem.read_bytes()).hexdigest()=='121ab59ea6f558ec41b69dcb5b859821f40e4425812e46cb73ed91055cc22d0a'
npu=source.parents[2]/'hardware_backend/npu/memory_pool_npu.py';tree=ast.parse(npu.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='NPUMLATokenToKVPoolHost');assert [ast.unparse(x) for x in cls.bases]==['MLATokenToKVPoolHost']
assert not any(isinstance(x,ast.FunctionDef) and x.name=='_transfer_ascendc_sparse_copy' for x in cls.body)
assert Path('/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so').stat().st_size==0
os.environ.update(GLM53_C32_NATIVE_HICACHE_MTP='1',SGLANG_HICACHE_HOST_MEM_BACKEND='memfabric',GLM53_NODE='34')
p=root/'python/glm53_C32_hicache_native_import_hook_v1.py';s=importlib.util.spec_from_file_location('P_hook',p);h=importlib.util.module_from_spec(s);s.loader.exec_module(h);h.activate();assert not any(k.startswith(('torch','sglang','memfabric_hybrid','memcache_hybrid')) for k in sys.modules)
print('GLM53_L12B_IMAGE_CPU='+json.dumps({'pass':True,'SDK_init_calls':0,'NPU_calls':0,'model_requests':0,'MLA_source_exact':True,'NPU_host_inherits_compatible_MLA':True,'stock_extension_placeholder_unchanged':True,'P_hook_lazy':True}))
'''
def plan():
 m=load();bs=bundles();runtime=json.loads((ROOT/'config/C32-target-graph-runtime-v3/runtime.json').read_text());checks={}
 for n in [34,35,36,41]:
  item=runtime['runtime_containers'][str(n)];c=m.replace(HEALTH,[('CID',item['container_id']),('NAME',item['name']),('NODE',n),('IMAGE',runtime['model_image']),('SHARED',runtime['shared_run_id'])]);ast.parse(c);checks['health-'+str(n)]=c
 e=bs[0]['private_optional_extension'];data=base64.b64encode((ROOT/e['local_artifact']).read_bytes()).decode()
 for n in [34,35]:
  c=m.replace(BINARY,[('REMOTE',bs[0]['remote']),('RELATIVE',e['relative']),('DATA',data),('SHA',e['sha256']),('NODE',n)]);ast.parse(c);checks['binary-'+str(n)]=c
 ast.parse(CPU);return m,bs,checks

def main():
 proof=json.loads((ROOT/'results/C32-L12B-local-source-CPU-review-v1.json').read_text());assert proof['pass']
 m,bs,checks=plan();review=json.loads((ROOT/'results/C32-L12B-stage-actual-review-v1.json').read_text());assert review['host_code_sha256']=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in checks.items()} and review['CPU_code_sha256']==hashlib.sha256(CPU.encode()).hexdigest()
 RUN.mkdir(exist_ok=False);(RUN/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','max_SSH':2,'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'existing_models_changed':False,'health':{},'stages':{},'extension':{}}
 def save(name,r):(RUN/name).write_text(json.dumps(r,indent=2)+'\n');return r
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs={pool.submit(m.invoke,n,checks['health-'+str(n)],120):n for n in [34,35,36,41]}
   for f in concurrent.futures.as_completed(jobs):
    n=jobs[f];r=save('health-'+str(n)+'.json',f.result());receipt['health'][str(n)]=r['rc'];assert r['rc']==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs={pool.submit(m.stage,b,n):(i,n) for i,b in enumerate(bs) for n in b['nodes']}
   for f in concurrent.futures.as_completed(jobs):
    i,n=jobs[f];key=str(i)+'-'+str(n);r=save('stage-'+key+'.json',f.result());receipt['stages'][key]=r['rc'];assert r['rc']==0;print('staged',key,flush=True)
  for n in [34,35]:
   r=save('extension-'+str(n)+'.json',m.invoke(n,checks['binary-'+str(n)],120));receipt['extension'][str(n)]=r['rc'];assert r['rc']==0
  args=['docker','run','--rm','--pull=never','--network=none','--cpus=2','--memory=4g','--pids-limit=256','--label','project_id=glm53','--name','glm53-L12B-CPU-20261019-36','-v',bs[0]['remote']+':/opt/glm53/staged:ro','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','python3',bs[0]['image'],'-c',CPU]
  code='import subprocess,json\nr=subprocess.run('+repr(args)+',capture_output=True,text=True,timeout=120);print(json.dumps({"rc":r.returncode,"stdout":r.stdout,"stderr":r.stderr}));raise SystemExit(r.returncode)'
  ast.parse(code);r=save('CPU-image.json',m.invoke(36,code,150));assert r['rc']==0;v=json.loads(r['stdout']);assert v['rc']==0;assert json.loads(v['stdout'].split('GLM53_L12B_IMAGE_CPU=')[-1])['pass'];receipt['CPU_pass']=True;receipt['pass']=True
 finally:save('receipt.json',receipt)
 print(json.dumps(receipt))
if __name__=='__main__':main()
