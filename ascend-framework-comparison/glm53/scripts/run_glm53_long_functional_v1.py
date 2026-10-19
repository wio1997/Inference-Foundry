#!/usr/bin/env python3
"""First actual80K/600 native-client functional cases, not formal performance."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0680-long-functional-v1'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T0680-long-functional-v1'
CLIENT_IMAGE='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
MODEL_IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
IDS={30:'7c5a8fa28d185ff0aef35a1330e737934adc0d6cdb232919e2e05901c9f866d4',31:'64779f17401c305adcaa13955e1003eba84e44d27670eb88324ebd2a1e7763b0',43:'d6894b55dc4fc874aa6187bb4723b8dec43cdb21bb321b7702d114c6757d142e',45:'3ba6b5d6f580bc18cdb45190b1fcd07950cac2b176a0c4d930d01a492074c3a9'}
FROZEN={'aisbench_glm53_token_ids.py':'d7157131f0b7a536731e2f9921db4740c2fb62668731b504c2c687342d09e13c',
        'glm53_vllm_pd_gateway.py':'de81c0a5ee20faa97662e9b49f33b4a476ad52c9431f89dbd52f7acd595ce815',
        'glm53_long_functional_client.py':'817e0cefe0e4cf2bceb60510baaeabc635a8fee35d9acca97321c5bd5f592f43'}
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
assert data['Config']['Labels']['project_id']=='glm53' and data['Config']['Labels']['run_id']=='20261017T0550-vllm-correctness-v2'
with urllib.request.urlopen(URL,timeout=3) as response:assert response.status==200
print(json.dumps({'id':CID,'node':NODE,'health':200,'state':data['State']}))
'''
 import re
 for key,value in [('CID',IDS[node]),('NAME',f'/glm53-vllm-correctness-v2-20261017-{node}'),('IMAGE',MODEL_IMAGE),('NODE',node),('URL',f'http://172.16.12.{node}:'+('30300' if node in (30,31) else '30400')+'/health')]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 return invoke(node,code,75)


def execute_client():
 args=['docker','run','--rm','--pull=never','--network=host','--pids-limit=512','--memory=8g','--cpus=4',
       '--label','project_id=glm53','--label','run_id=20261017T0680-long-functional-v1',
       '--name','glm53-long-functional-v1-20261017-43','--cidfile',REMOTE+'/client.cid','--workdir','/run/glm53',
       '-v','/data/wio/aisbench-offline-43-20261014:/work:ro','-v',REMOTE+':/run/glm53',
       '-v','/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts/C16:/dataset:ro',
       '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONNOUSERSITE=1','-e','PYTHONPATH=/run/glm53:/work/benchmark',
       '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false',
       '-e','CUDA_VISIBLE_DEVICES=','-e','ASCEND_RT_VISIBLE_DEVICES=',
       '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HOME=/run/glm53/hf',
       '-e','XDG_CACHE_HOME=/run/glm53/cache','-e','MPLCONFIGDIR=/run/glm53/mpl','-e','TORCH_HOME=/run/glm53/torch',
       '-e','NLTK_DATA=/work/nltk_data','-e','TIKTOKEN_CACHE_DIR=/work/tiktoken_cache',
       '--entrypoint','timeout',CLIENT_IMAGE,'4000','/work/venv/bin/python','/run/glm53/glm53_long_functional_client.py']
 code='''import pathlib,json,hashlib,subprocess,sys
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15)
assert image.returncode==0 and image.stdout.strip()==IMAGE
core=pathlib.Path('/data/wio/aisbench-offline-43-20261014/benchmark/ais_bench/benchmark')
for entry in IDENTITY['files']:
 assert hashlib.sha256((core/entry['path']).read_bytes()).hexdigest()==entry['sha256']
dataset=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts/C16')
assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()=='dff4e3943c14f8d12c1964f0aad9ed419441e6b94274f40507d2fdd34f4e3ab9'
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
for name,text in FILES.items():
 assert pathlib.PurePosixPath(name).name==name
 (root/name).write_text(text)
result=subprocess.run(ARGS,capture_output=True,text=True,timeout=4050)
print(result.stdout,end='');print(result.stderr,file=sys.stderr,end='')
artifacts={}
for path in sorted(root.glob('metrics-*.txt')):
 data=path.read_bytes();artifacts[path.name]={'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()}
for name in ['functional-progress.json','functional-result.json','client.cid']:
 path=root/name
 if path.is_file():
  data=path.read_bytes();artifacts[name]={'sha256':hashlib.sha256(data).hexdigest(),'text':data.decode()}
print('GLM53_LONG_ARTIFACTS='+json.dumps(artifacts),flush=True)
sys.exit(result.returncode)
'''
 import re
 for key,value in [('IMAGE',CLIENT_IMAGE),('IDENTITY',IDENTITY),('REMOTE',REMOTE),('FILES',FILES),('ARGS',args)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 return invoke(43,code,4120)


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'files_sha256':FROZEN,
          'phase':'FUNCTIONAL_ONLY_NOT_PERFORMANCE','model_precheck':{},'rc':None,'logical_client_cases_planned':4,'backend_model_calls_planned':7,'formal_requests':0}
 try:
  for node in IDS:
   result=precheck(node);(RUN/f'precheck-{node}.json').write_text(json.dumps(result,indent=2))
   receipt['model_precheck'][node]=result['rc'];assert result['rc']==0,'Existing model gate failed; no long client starts'
  (RUN/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
  result=execute_client();(RUN/'node-43.json').write_text(json.dumps(result,indent=2)+'\n');receipt['rc']=result['rc']
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('long functional',receipt['rc'],flush=True)


if __name__=='__main__':main()
