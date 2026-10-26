#!/usr/bin/env python3
"""One bounded native SGLang functional PD matrix, no benchmark acceptance."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1380-sglang-pd-matrix-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1380-sglang-pd-matrix-v1';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
IDS={34:'b91ac6ca6fffa3caa5ab08869997b0590f53846db48723991b9189cd7d248aeb',35:'b3cae6b8479be6db5a2bd001eade675330f01b238176292abdca172fec93c529',36:'d0b3515f3cb028e15aae787631bf807b63f394dbc57fd5ae782023a676588915',41:'3cb55401fe1bc03e21709048285d4d77439cf2e228640009a8957bfb2b6e885c'}
FILES={n:(P.parent/n).read_text() for n in ['probe_sglang_pd_matrix_payload_v1.py','glm53_sglang_pd_gateway_v1.py']}
FROZEN={'probe_sglang_pd_matrix_payload_v1.py':'8c74018c1691d8195de41798545b74fb9a0096d54e07e909600a974ec6eba743','glm53_sglang_pd_gateway_v1.py':'d3caaad39d41b2e09ab4d0f874f1acc5fcfcc5666a1a10f2f80b4a40018f02a4'}
for n,v in FILES.items():assert hashlib.sha256(v.encode()).hexdigest()==FROZEN[n],n

def invoke(node,code,timeout=100):
 pod,host=b.NODES[node];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']==f'172.16.12.{node}'
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def check(node):
 code=r'''import subprocess,json
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1320-sglang-correctness-v4'
env=dict(x.split('=',1) for x in d['Config']['Env']);assert env['DEEP_USE_MODE']=='alltoall'
print(json.dumps({'node':NODE,'running':True,'native_strategy_selector':'alltoall'}))
'''
 for k,v in [('CID',IDS[node]),('NAME',f'glm53-sglang-correctness-v4-20261017-{node}'),('IMAGE',IMAGE),('NODE',node)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return invoke(node,code)
def execute():
 args=['docker','run','--rm','--pull=never','--network=host','--cpus=4','--memory=4g','--pids-limit=256','--label','project_id=glm53','--label','run_id=20261017T1380-sglang-pd-matrix-v1','--name','glm53-sglang-pd-matrix-v1-20261017-34','-v',REMOTE+':/run/glm53','-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/run/glm53','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','CUDA_VISIBLE_DEVICES=','--entrypoint','timeout',IMAGE,'3100','python3','/run/glm53/probe_sglang_pd_matrix_payload_v1.py']
 code=r'''import subprocess,pathlib,json,hashlib,sys
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay';root.mkdir(parents=True)
for name,text in FILES.items():
 p=root/name;p.write_text(text);assert hashlib.sha256(p.read_bytes()).hexdigest()==FROZEN[name]
with (root/'client-stdout.txt').open('w') as out,(root/'client-stderr.txt').open('w') as err:r=subprocess.run(ARGS,stdout=out,stderr=err,text=True,timeout=3150)
result=root/'result.json';data=json.loads(result.read_text()) if result.is_file() else None
print('GLM53_SG_MATRIX_ARTIFACTS='+json.dumps({'client_rc':r.returncode,'result':data,'artifacts':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file()}}),flush=True)
print((root/'client-stderr.txt').read_text()[-16000:],file=sys.stderr,end='');sys.exit(r.returncode)
'''
 for k,v in [('REMOTE',REMOTE),('FILES',FILES),('FROZEN',FROZEN),('ARGS',args)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return invoke(34,code,timeout=3220)
def main():
 qualified=json.loads((b.ROOT/'results/sglang-v4-startup.json').read_text());assert qualified['nodes']['34']['native_warmup_completion_logs'] and qualified['nodes']['36']['native_warmup_completion_logs']
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'prechecks':{},'rc':None,'formal_requests':0}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   fs={pool.submit(check,n):n for n in IDS}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f];r=f.result();(RUN/f'precheck-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['prechecks'][n]=r['rc'];assert r['rc']==0
  r=execute();(RUN/'node-34.json').write_text(json.dumps(r,indent=2)+'\n');receipt['rc']=r['rc'];print('matrix',r['rc'],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
