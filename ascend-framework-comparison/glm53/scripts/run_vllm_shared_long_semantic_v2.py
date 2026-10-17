#!/usr/bin/env python3
"""Once-only native AIS long vLLM cold/warm/exact-known-key/cross-P-DP guards."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RID='20261017T1779-vllm-shared-long-semantic-v2';RUN=b.ROOT/'runs'/RID;REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
CLIENT_IMAGE='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
META_IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
VLLM_IMAGE=META_IMAGE
SDK='/data/wio/inference-foundry/glm53/overlays/sdk13-20261017'
IDS={30: 'a654305d634fb392660cee17797c15f31af90df477c0025839a59f985e1c073c', 31: '7919640401f16f87c8edb1a71c7e78915c8d76441a428606bf7edec11379bc0e', 43: '0b3266fcc1f72005a763fdbd5191cecb564d67b448508c2e1d42e965a2973779', 45: 'c1f460d681f216677d48e3ca881c72bd3a325f10b19d2d83cba299d47dae8a99'}
FROZEN={'aisbench_glm53_token_ids.py': 'd7157131f0b7a536731e2f9921db4740c2fb62668731b504c2c687342d09e13c', 'glm53_vllm_pd_gateway_v3.py': 'b7ee8ea8266ecf17d5af774af6e54fcb025c5bf6def22a4b0952bf6251f16967', 'glm53_vllm_long_semantic_client_v4.py': 'c2115ca9344ccf1809a49899e2ee898ca9a48426fdd0b93fc1f21a706367f9b8', 'glm53_vllm_semantic_meta_probe_v4.py': '10d668029ae9b968f1395ed33b1a2e8f3fd8432c11f4c625202c6ecf7e30b7ee', 'glm53_request_terminal_v1.py': '34e9f176acea0bde847b349bf85442a12104461260145bead3529aef2ca82ba4'}
FILES={}
for name,sha in FROZEN.items():
 raw=(P.parent/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha;FILES[name]=raw.decode()
native_config=b.ROOT/'config/vllm-correctness-v10/local.conf';assert hashlib.sha256(native_config.read_bytes()).hexdigest()=='07f071d38ab9f2d81d9dadd4d301fb4fa2234f6d11afa5c7badf087faffc5d7f'
FILES['metadata-local.conf']=native_config.read_text();FROZEN['metadata-local.conf']='07f071d38ab9f2d81d9dadd4d301fb4fa2234f6d11afa5c7badf087faffc5d7f'
assert 'meta_service_url = tcp://172.16.12.30:29680' in FILES['metadata-local.conf'] and 'config_store_url = tcp://172.16.12.30:29681' in FILES['metadata-local.conf']
identity_path=b.ROOT/'runs/20261017T0610-aisbench-assets-v1/identity-summary.json'
assert hashlib.sha256(identity_path.read_bytes()).hexdigest()=='7cf76483c63972186dc4c589fe04e4d8c49d8fa8eb020aab49fc4a10b955bb8f'
IDENTITY=json.loads(identity_path.read_text())
PREREQS={'results/vllm-shared-startup-v1.json': 'db64b2127c6ec3f84e6ec273cf0339ab62c0aec8e9ef9e8337779c19589ca953', 'results/vllm-shared-short-matrix-v1.json': 'ebf25ab6884a2aaf303ddc6f655fe7e1775528159a496c60c668afcf5816627f', 'results/request-terminal-actual-fixture-v2.json': 'e07b66b2df193c641bb47d20a9687f6c0b747452ca32d4365f77fbeccc123942'}
for name,sha in PREREQS.items():assert hashlib.sha256((b.ROOT/name).read_bytes()).hexdigest()==sha

def invoke(node,code,timeout=100):
 pod,host=b.NODES[node];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']==f'172.16.12.{node}'
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def replace(code,values):
 table=dict(values);pattern=r'\b(?:'+'|'.join(re.escape(k) for k in table)+r')\b';code=re.sub(pattern,lambda match:repr(table[match.group(0)]),code);__import__('ast').parse(code);return code
def check(node):
 path=P.parent/'glm53_vllm_shared_model_identity_v3.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='24a93a46cfc19946e4e52cc2d255e92ea552c127b560d12ee1fea7108a18fb89'
 spec=importlib.util.spec_from_file_location('identity',path);identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
 code=identity.HOST+r"""
import urllib.request
with urllib.request.urlopen(URL,timeout=10) as response:assert response.status==200;raw=response.read(8388609);assert len(raw)<=8388608
lines=raw.decode().splitlines();gauges=[float(line.rsplit(' ',1)[1]) for line in lines if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
assert gauges and all(x==0 for x in gauges),'Existing workload active; no long client dispatch'
print(json.dumps({'node':NODE,'container_id':CID,'actual_model_identity':model_identity,'idle_gauges':gauges}))
"""
 return invoke(node,replace(code,[('CID',IDS[node]),('NAME',f'glm53-vllm-shared-v4-20261017-{node}'),('IMAGE',VLLM_IMAGE),('NODE',node),('URL',f'http://172.16.12.{node}:'+('30300' if node in [30,31] else '30400')+'/metrics')]))
def execute():
 args=['docker','run','--rm','--pull=never','--network=host','--pids-limit=512','--memory=8g','--cpus=4','--label','project_id=glm53','--label','run_id='+RID,'--name','glm53-vllm-shared-long-semantic-v2-20261017-43','--cidfile',REMOTE+'/client.cid','--workdir','/run/glm53','-v','/data/wio/aisbench-offline-43-20261014:/work:ro','-v',REMOTE+':/run/glm53','-v','/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1:/semantic:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONNOUSERSITE=1','-e','PYTHONPATH=/run/glm53:/work/benchmark','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','CUDA_VISIBLE_DEVICES=','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HOME=/run/glm53/hf','-e','XDG_CACHE_HOME=/run/glm53/cache','-e','MPLCONFIGDIR=/run/glm53/mpl','-e','TORCH_HOME=/run/glm53/torch','-e','NLTK_DATA=/work/nltk_data','-e','TIKTOKEN_CACHE_DIR=/work/tiktoken_cache','--entrypoint','timeout',CLIENT_IMAGE,'4000','/work/venv/bin/python','/run/glm53/glm53_vllm_long_semantic_client_v4.py']
 boot='export PYTHONHASHSEED=0; export PYTHONPATH=/opt/glm53/sdk13:/run/glm53:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; exec timeout 150 python3 /run/glm53/glm53_vllm_semantic_meta_probe_v4.py'
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
     assert request['index']==index and request['namespace']==f'glm53-sem-vllm-shared-v2-20261017T1779-case{index}-warm'
     assert request['prefix_ids']==record['prefix_ids'] and request['prefix_sha256_le_u32']==record['prefix_sha256_le_u32']
     try:
      meta=subprocess.run(META_ARGS[:2]+['--name',f'glm53-vllm-shared-semantic-meta-v2-20261017-43-{index}']+META_ARGS[2:],input=json.dumps(request),capture_output=True,text=True,timeout=160)
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
