#!/usr/bin/env python3
"""Three actual74400/1 prefix warm/knownkeys/alternateDPread gates, not performance."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0950-long-semantic-v1'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T0950-long-semantic-v1'
CLIENT_IMAGE='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
MODEL_IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
IDS={30:'b3cf90053554bfeec3e65cfd0a11c69afaf456eb077f3482d6e7b590e63f5219',31:'be7b34b50bf325741fdbfa33fdc1070bda627df1082d8c1e67daa63ae197dde8',43:'d6cc48ee6b8c993b5a583456c4389213bc57d595054186976c3c79994bfcc294',45:'8f844997dbb88cbbc8d8dc1f28d170f3bee7199204dd87ad3f944ee98a66181b'}
FROZEN={'aisbench_glm53_token_ids.py': 'd7157131f0b7a536731e2f9921db4740c2fb62668731b504c2c687342d09e13c', 'glm53_vllm_pd_gateway_v2.py': '799ed5ce70dd9f89289319f78e5a03fecdef0ddba9e9cf238920c72a48074e2f', 'glm53_long_semantic_client_v1.py': 'c0b4106e428b2b74dcbc67566da8c064597f201f75a3aede0f67a6290c0737bd', 'glm53_semantic_meta_probe_v1.py': 'cea13a86aa5eed5e91fdfd04a67f91c6b8d9cbce7b63c87250ee97911f23f99b'}
FILES={name:(P.parent/name).read_text() for name in FROZEN}
for name,text in FILES.items():assert hashlib.sha256(text.encode()).hexdigest()==FROZEN[name]
identity_path=b.ROOT/'runs/20261017T0610-aisbench-assets-v1/identity-summary.json'
assert hashlib.sha256(identity_path.read_bytes()).hexdigest()=='7cf76483c63972186dc4c589fe04e4d8c49d8fa8eb020aab49fc4a10b955bb8f'
IDENTITY=json.loads(identity_path.read_text())
qualification_path=b.ROOT/'results/model-pd-matrix-v2.json'
assert hashlib.sha256(qualification_path.read_bytes()).hexdigest()=='0717220a75848258439e45993b07262247848b0419640c83ced7ed5802c97312'
assert json.loads(qualification_path.read_text())['status']=='ALL_DP_SMALL_PD_STATE_BYTES_AND_GREEDY_OUTPUT_PASSED'


def invoke(node,code,timeout):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code='import socket\nassert socket.gethostname()=='+repr(hostname)+'\n'+code
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],code,timeout=timeout)


def precheck(node):
 code='''import subprocess,json,urllib.request,sys
raw=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=15)
assert raw.returncode==0;data=json.loads(raw.stdout)[0]
assert data['Id']==CID and data['Name']==NAME and data['Image']==IMAGE and data['State']['Running']
assert data['Config']['Labels']['project_id']=='glm53' and data['Config']['Labels']['run_id']=='20261017T0840-vllm-correctness-v3'
with urllib.request.urlopen(URL,timeout=3) as response:assert response.status==200
print(json.dumps({'id':CID,'node':NODE,'health':200,'state':data['State']}))
'''
 import re
 for key,value in [('CID',IDS[node]),('NAME',f'/glm53-vllm-correctness-v3-20261017-{node}'),('IMAGE',MODEL_IMAGE),('NODE',node),('URL',f'http://172.16.12.{node}:'+('30300' if node in (30,31) else '30400')+'/health')]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 return invoke(node,code,75)


def execute_client():
 args=['docker','run','--rm','--pull=never','--network=host','--pids-limit=512','--memory=8g','--cpus=4',
       '--label','project_id=glm53','--label','run_id=20261017T0950-long-semantic-v1',
       '--name','glm53-long-semantic-v1-20261017-43','--cidfile',REMOTE+'/client.cid','--workdir','/run/glm53',
       '-v','/data/wio/aisbench-offline-43-20261014:/work:ro','-v',REMOTE+':/run/glm53',
       '-v','/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1:/semantic:ro',
       '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONNOUSERSITE=1','-e','PYTHONPATH=/run/glm53:/work/benchmark',
       '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false',
       '-e','CUDA_VISIBLE_DEVICES=','-e','ASCEND_RT_VISIBLE_DEVICES=',
       '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HOME=/run/glm53/hf',
       '-e','XDG_CACHE_HOME=/run/glm53/cache','-e','MPLCONFIGDIR=/run/glm53/mpl','-e','TORCH_HOME=/run/glm53/torch',
       '-e','NLTK_DATA=/work/nltk_data','-e','TIKTOKEN_CACHE_DIR=/work/tiktoken_cache',
       '--entrypoint','timeout',CLIENT_IMAGE,'4000','/work/venv/bin/python','/run/glm53/glm53_long_semantic_client_v1.py']
 code=r'''import pathlib,json,hashlib,subprocess,sys,time
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15)
assert image.returncode==0 and image.stdout.strip()==IMAGE
core=pathlib.Path('/data/wio/aisbench-offline-43-20261014/benchmark/ais_bench/benchmark')
for entry in IDENTITY['files']:assert hashlib.sha256((core/entry['path']).read_bytes()).hexdigest()==entry['sha256']
dataset=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1')
assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()=='ac8fbf5e4425c349440c85789d9a4be7137579a0b2a4072f8a2adf00c9bf6ece'
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
for name,text in FILES.items():
 assert pathlib.PurePosixPath(name).name==name
 (root/name).write_text(text)
raw=subprocess.run(['docker','inspect',MODEL_CID],capture_output=True,text=True,timeout=15);assert raw.returncode==0
own=json.loads(raw.stdout)[0];assert own['Id']==MODEL_CID and own['Name']=='/glm53-vllm-correctness-v3-20261017-43' and own['Image']==MODEL_IMAGE and own['Config']['Labels']['project_id']=='glm53' and own['State']['Running']
bootstrap='export PYTHONPATH=/opt/glm53/sdk13:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; export PYTHONDONTWRITEBYTECODE=1; exec python3 -c "$1"'
status={'client_rc':None,'gates':{}};deadline=time.monotonic()+4000
with (root/'client-stdout.txt').open('w') as out,(root/'client-stderr.txt').open('w') as err:
 process=subprocess.Popen(ARGS,stdout=out,stderr=err,text=True)
 try:
  while process.poll() is None:
   for index in (0,1):
    request_path=root/f'need-metadata-{index}.json'
    if request_path.is_file() and str(index) not in status['gates']:
     request=json.loads(request_path.read_text());record=json.loads((dataset/f'semantic-{index}.json').read_text())
     assert request['index']==index and request['namespace']=='glm53-sem-vllm-v3-20261017-case'+str(index)+'-warm'
     assert request['prefix_ids']==record['prefix_ids'] and request['prefix_sha256_le_u32']==record['prefix_sha256_le_u32']
     meta=subprocess.run(['docker','exec','-i',MODEL_CID,'bash','-c',bootstrap,'glm53-meta',FILES['glm53_semantic_meta_probe_v1.py']],input=json.dumps(request),capture_output=True,text=True,timeout=100)
     (root/f'metadata-{index}-stdout.txt').write_text(meta.stdout);(root/f'metadata-{index}-stderr.txt').write_text(meta.stderr)
     lines=[x[len('GLM53_PREFIX_METADATA='):] for x in meta.stdout.splitlines() if x.startswith('GLM53_PREFIX_METADATA=')]
     result=json.loads(lines[0]) if len(lines)==1 else {}
     gate={'rc':meta.returncode,'pass':bool(meta.returncode==0 and result.get('pass') and result.get('close_rc')==0),'metadata':result,'stdout_sha256':hashlib.sha256(meta.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(meta.stderr.encode()).hexdigest()}
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
 import re
 for key,value in [('IMAGE',CLIENT_IMAGE),('IDENTITY',IDENTITY),('REMOTE',REMOTE),('FILES',FILES),('ARGS',args),('MODEL_CID',IDS[43]),('MODEL_IMAGE',MODEL_IMAGE)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 return invoke(43,code,4120)


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'files_sha256':FROZEN,
          'phase':'FUNCTIONAL_ONLY_NOT_PERFORMANCE','model_precheck':{},'rc':None,'logical_client_cases_planned':8,'backend_model_calls_planned':14,'formal_requests':0}
 try:
  for node in IDS:
   result=precheck(node);(RUN/f'precheck-{node}.json').write_text(json.dumps(result,indent=2))
   receipt['model_precheck'][node]=result['rc'];assert result['rc']==0,'Existing model gate failed; no long client starts'
  (RUN/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
  result=execute_client();(RUN/'node-43.json').write_text(json.dumps(result,indent=2)+'\n');receipt['rc']=result['rc']
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('long functional',receipt['rc'],flush=True)


if __name__=='__main__':main()
