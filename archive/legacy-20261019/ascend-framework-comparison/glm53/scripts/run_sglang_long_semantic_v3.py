#!/usr/bin/env python3
"""Once-only native AIS long SG cold/warm/exact-known-key/cross-P-DP guards."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RID='20261017T1578-sglang-long-semantic-v3';RUN=b.ROOT/'runs'/RID;REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
CLIENT_IMAGE='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
META_IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
SG_IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
SDK='/data/wio/inference-foundry/glm53/overlays/sdk13-20261017'
IDS={34: '9844ec4193f7bc662ac43a62e2606a72f8299992283f27d2005a8ce9c643f805', 35: 'bf1859f9bad4d03c10e78a647703dcff951d4dc81dd53dd6e941ba0d97e928db', 36: '2af3259c99c29dd00a7e8ee84906c96c4657bd5050a9190ebcc093d49991dbad', 41: '9d3495a179b91716c8685ecb18a8e78001abbad6e7a90b4a64f329f33c319196'}
FROZEN={'aisbench_glm53_token_ids.py': 'd7157131f0b7a536731e2f9921db4740c2fb62668731b504c2c687342d09e13c', 'glm53_sglang_pd_gateway_v3.py': 'cb2d3581755ec6e55d9d5449a3f9b69c233d611edc42d7a1a6dbf6d891d5d431', 'glm53_sglang_long_semantic_client_v3.py': '4a86fa7387f8d6d91d451b9aa57cc3a546d995d8d75a157168cff75033e2872a', 'glm53_sglang_prefix_keys_v1.py': 'a2188e0205d85615a35ca3dd80a5339a7e3a28707c778ecc882aeab6ac3a15e7', 'glm53_sglang_semantic_meta_probe_v3.py': '012d06f9d9425f3a589b541df2b43fba19818b4c374f10b5144852955210ee24', 'glm53_request_terminal_v1.py': '34e9f176acea0bde847b349bf85442a12104461260145bead3529aef2ca82ba4', 'glm53_storage_proof_v1.py': 'eaef90335bc82d3e258c1c25808a3418f58526b5c8ad52d12fd3930f7b7e92c5'}
FILES={}
for name,sha in FROZEN.items():
 raw=(P.parent/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha;FILES[name]=raw.decode()
FILES['metadata-local.conf']='ock.mmc.meta_service_url = tcp://172.16.12.34:29780\nock.mmc.local_service.config_store_url = tcp://172.16.12.34:29781\nock.mmc.log_level = info\nock.mmc.local_service.world_size = 128\nock.mmc.local_service.protocol = device_sdma\nock.mmc.local_service.dram.size = 0GB\nock.mmc.local_service.max.dram.size = 16GB\n'
identity_path=b.ROOT/'runs/20261017T0610-aisbench-assets-v1/identity-summary.json'
assert hashlib.sha256(identity_path.read_bytes()).hexdigest()=='7cf76483c63972186dc4c589fe04e4d8c49d8fa8eb020aab49fc4a10b955bb8f'
IDENTITY=json.loads(identity_path.read_text())
PREREQS={'results/sglang-knownkey-fixture-v2.json': '9796e7cefc73698007bdbc5b30bbf7058dd39d8a15be6ecf5af128ec28ab3ed0', 'results/sglang-pd-sdma-matrix-v3.json': 'b4851a8283fe254e41aa940d81894d82a682794eb87d0c66a64ed15cb7012c6c', 'results/request-terminal-actual-fixture-v2.json': 'e07b66b2df193c641bb47d20a9687f6c0b747452ca32d4365f77fbeccc123942'}
for name,sha in PREREQS.items():assert hashlib.sha256((b.ROOT/name).read_bytes()).hexdigest()==sha

def invoke(node,code,timeout=100):
 pod,host=b.NODES[node];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']==f'172.16.12.{node}'
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def replace(code,values):
 for key,value in values:code=re.sub(r'\b'+key+r'\b',lambda _:repr(value),code)
 return code
def check(node):
 code=r'''import subprocess,json,urllib.request
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1562-sglang-correctness-v6-v3'
env=dict(x.split('=',1) for x in d['Config']['Env']);assert env['DEEP_USE_MODE']=='alltoall'
with urllib.request.urlopen(URL,timeout=10) as response:assert response.status==200;raw=response.read().decode()
gauges=[float(line.rsplit(' ',1)[1]) for line in raw.splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
assert gauges and all(x==0 for x in gauges),'Existing workload still active; no long client dispatch'
print(json.dumps({'node':NODE,'container_id':CID,'running':True,'DEEP_USE_MODE':'alltoall','idle_gauges':gauges}))
'''
 return invoke(node,replace(code,[('CID',IDS[node]),('NAME',f'glm53-sglang-correctness-v6-20261017-{node}'),('IMAGE',SG_IMAGE),('NODE',node),('URL',f'http://172.16.12.{node}:'+('30500' if node in [34,35] else '30600')+'/metrics')]))
def execute():
 args=['docker','run','--rm','--pull=never','--network=host','--pids-limit=512','--memory=8g','--cpus=4','--label','project_id=glm53','--label','run_id='+RID,'--name','glm53-sglang-long-semantic-v3-20261017-43','--cidfile',REMOTE+'/client.cid','--workdir','/run/glm53','-v','/data/wio/aisbench-offline-43-20261014:/work:ro','-v',REMOTE+':/run/glm53','-v','/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1:/semantic:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONNOUSERSITE=1','-e','PYTHONPATH=/run/glm53:/work/benchmark','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','CUDA_VISIBLE_DEVICES=','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HOME=/run/glm53/hf','-e','XDG_CACHE_HOME=/run/glm53/cache','-e','MPLCONFIGDIR=/run/glm53/mpl','-e','TORCH_HOME=/run/glm53/torch','-e','NLTK_DATA=/work/nltk_data','-e','TIKTOKEN_CACHE_DIR=/work/tiktoken_cache','--entrypoint','timeout',CLIENT_IMAGE,'4000','/work/venv/bin/python','/run/glm53/glm53_sglang_long_semantic_client_v3.py']
 boot='export PYTHONPATH=/opt/glm53/sdk13:/run/glm53; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; exec timeout 150 python3 /run/glm53/glm53_sglang_semantic_meta_probe_v3.py'
 meta_args=['docker','run','--rm','-i','--pull=never','--network=host','--memory=4g','--cpus=2','--pids-limit=512','--label','project_id=glm53','--label','run_id='+RID,'--workdir','/run/glm53','-v',REMOTE+':/run/glm53:ro','-v',SDK+':/opt/glm53/sdk13:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','MMC_LOCAL_CONFIG_PATH=/run/glm53/metadata-local.conf','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','CUDA_VISIBLE_DEVICES=','--entrypoint','/bin/bash',META_IMAGE,'-c',boot]
 code=r'''import pathlib,json,hashlib,subprocess,sys,time
for image in (CLIENT_IMAGE,META_IMAGE):
 r=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and r.stdout.strip()==image
core=pathlib.Path('/data/wio/aisbench-offline-43-20261014/benchmark/ais_bench/benchmark')
for entry in IDENTITY['files']:assert hashlib.sha256((core/entry['path']).read_bytes()).hexdigest()==entry['sha256']
dataset=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1')
assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()=='ac8fbf5e4425c349440c85789d9a4be7137579a0b2a4072f8a2adf00c9bf6ece'
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
for name,text in FILES.items():
 assert pathlib.PurePosixPath(name).name==name
 (root/name).write_text(text)
 if name in FROZEN:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==FROZEN[name]
status={'client_rc':None,'gates':{}};deadline=time.monotonic()+4000
with (root/'client-stdout.txt').open('w') as out,(root/'client-stderr.txt').open('w') as err:
 process=subprocess.Popen(ARGS,stdout=out,stderr=err,text=True)
 try:
  while process.poll() is None:
   for index in (0,1):
    request_path=root/f'need-metadata-{index}.json'
    if request_path.is_file() and str(index) not in status['gates']:
     request=json.loads(request_path.read_text());record=json.loads((dataset/f'semantic-{index}.json').read_text())
     assert request['index']==index and request['namespace']==f'glm53-sem-sglang-v6-20261017-case{index}-warm'
     assert request['prefix_ids']==record['prefix_ids'] and request['prefix_sha256_le_u32']==record['prefix_sha256_le_u32']
     try:
      meta=subprocess.run(META_ARGS[:2]+['--name',f'glm53-sglang-semantic-meta-v3-20261017-43-{index}']+META_ARGS[2:],input=json.dumps(request),capture_output=True,text=True,timeout=160)
      (root/f'metadata-{index}-stdout.txt').write_text(meta.stdout);(root/f'metadata-{index}-stderr.txt').write_text(meta.stderr)
      lines=[x[len('GLM53_PREFIX_METADATA='):] for x in meta.stdout.splitlines() if x.startswith('GLM53_PREFIX_METADATA=')];result=json.loads(lines[0]) if len(lines)==1 else {}
      gate={'rc':meta.returncode,'pass':bool(meta.returncode==0 and result.get('pass') and result.get('close_rc')==0),'metadata':result,'stdout_sha256':hashlib.sha256(meta.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(meta.stderr.encode()).hexdigest()}
     except Exception as error:gate={'rc':-1,'pass':False,'error':repr(error)}
     status['gates'][str(index)]=gate;temp=root/f'metadata-gate-{index}.tmp';temp.write_text(json.dumps(gate)+'\n');temp.replace(root/f'metadata-gate-{index}.json')
   if time.monotonic()>deadline:raise TimeoutError('CPUclient4000s timeout; preserve active container and logs, no automatic cancellation')
   time.sleep(.2)
  status['client_rc']=process.returncode
 except Exception as error:status['error']=repr(error)
 finally:(root/'semantic-phase-status.json').write_text(json.dumps(status,indent=2)+'\n')
artifacts={}
for path in sorted(root.iterdir()):
 if path.is_file():
  data=path.read_bytes();artifacts[path.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
  if path.name in ('semantic-result.json','semantic-phase-status.json') or path.name.startswith('metadata-gate-'):artifacts[path.name]['text']=data.decode()
print('GLM53_LONG_ARTIFACTS='+json.dumps(artifacts),flush=True)
print((root/'client-stderr.txt').read_text()[-12000:],file=sys.stderr,end='')
print('GLM53_CLIENT_EXIT='+json.dumps(status),flush=True)
sys.exit(0 if status.get('client_rc')==0 and len(status['gates'])==2 and all(x['pass'] for x in status['gates'].values()) else 1)
'''
 return invoke(43,replace(code,[('CLIENT_IMAGE',CLIENT_IMAGE),('META_IMAGE',META_IMAGE),('IDENTITY',IDENTITY),('REMOTE',REMOTE),('FILES',FILES),('FROZEN',FROZEN),('ARGS',args),('META_ARGS',meta_args)]),4120)
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'files_sha256':FROZEN,'prerequisites_sha256':PREREQS,'phase':'FUNCTIONAL_ONLY_NOT_PERFORMANCE','model_precheck':{},'rc':None,'logical_client_cases_planned':6,'backend_model_calls_planned':12,'successful_output_tokens_if_all_pass':2408,'formal_requests':0}
 try:
  for node in IDS:
   r=check(node);(RUN/f'precheck-{node}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['model_precheck'][node]=r['rc'];assert r['rc']==0,'Model identity/idle gate failed; no long client starts'
  (RUN/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
  r=execute();(RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n');receipt['rc']=r['rc']
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('long functional',receipt['rc'],flush=True)
if __name__=='__main__':main()
