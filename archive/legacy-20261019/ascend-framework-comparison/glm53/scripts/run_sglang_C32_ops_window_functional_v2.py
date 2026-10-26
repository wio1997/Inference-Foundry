#!/usr/bin/env python3
"""One bounded native SGLang functional PD matrix, no benchmark acceptance."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261018T-L08-sglang-window-C32-functional-v2';REMOTE='/data/wio/inference-foundry/glm53/runs/20261018T-L08-sglang-window-C32-functional-v2';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
capacity=json.loads((b.ROOT/'results/sglang-C32-ops-window-pool-capacity-v2.json').read_text());assert capacity['pass']
IDS={int(n):item['container_id'] for n,item in capacity['providers'].items()}
FILES={n:(P.parent/n).read_text() for n in ['probe_sglang_C32_ops_window_pd_payload_v2.py','glm53_sglang_pd_gateway_v2.py']}
FROZEN={'probe_sglang_C32_ops_window_pd_payload_v2.py': '322d9a5eadb11cedada83335ef4fd23a624a20adefc811e4ff57e5b354fb00e7', 'glm53_sglang_pd_gateway_v2.py': '80922bbe499ddf061b34297bd728e71693241724bc982b527b341eab6faf264f'}
for n,v in FILES.items():assert hashlib.sha256(v.encode()).hexdigest()==FROZEN[n],n

def invoke(node,code,timeout=100):
 pod,host=b.NODES[node];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']==f'172.16.12.{node}'
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def check(node):
 identity_path=P.parent/'glm53_C32_ops_window_model_identity_v2.py'
 assert hashlib.sha256(identity_path.read_bytes()).hexdigest()=='ca80d9e4575f70d491e46f8da79ee87738e0baaaaefd897e7e8d184af1afad6c'
 spec=importlib.util.spec_from_file_location('model_identity',identity_path);identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
 code=identity.HOST+"\nprint(json.dumps({'node':NODE,'container_id':CID,'actual_model_identity':model_identity,'supervisor':supervisor,'provider_phase':provider['phase']}))\n"
 for k,v in [('CID',IDS[node]),('NAME',f'glm53-sglang-C32-ops-window-v2-20261018-{node}'),('IMAGE',IMAGE),('NODE',node)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return invoke(node,code)
def execute():
 args=['docker','run','--rm','--pull=never','--network=host','--cpus=4','--memory=4g','--pids-limit=256','--label','project_id=glm53','--label','run_id=20261018T-L08-sglang-window-C32-functional-v2','--name','glm53-sglang-C32-window-functional-v2-20261018-34','-v',REMOTE+':/run/glm53','-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/run/glm53','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','CUDA_VISIBLE_DEVICES=','--entrypoint','timeout',IMAGE,'3100','python3','/run/glm53/probe_sglang_C32_ops_window_pd_payload_v2.py']
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
 fixture=json.loads((b.ROOT/'runs/20261017T1400-sglang-gateway-fixture-v2/receipt.json').read_text());assert fixture['rc']==0
 audit=b.ROOT/'runs/20261018T-L08-sglang-window-startup-v2'
 for node in [34,35,36,41]:
  outer=json.loads((audit/f'node-{node}.json').read_text());assert outer['rc']==0;o=json.loads(outer['stdout']);assert o['model']['state']['Running']
  metrics=o['/metrics'].get('body','');assert metrics
  gauges=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(x==0 for x in gauges)
 qualified=json.loads((b.ROOT/'results/sglang-C32-ops-window-startup-v2.json').read_text());assert qualified['pass'] and qualified['all32_D_actual_native_ops_dispatch_combine_verified'] and qualified['runtime_CIDs']=={str(n):cid for n,cid in IDS.items()}
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'prechecks':{},'rc':None,'formal_requests':0}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
   fs={pool.submit(check,n):n for n in IDS}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f];r=f.result();(RUN/f'precheck-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['prechecks'][n]=r['rc'];assert r['rc']==0
  r=execute();(RUN/'node-34.json').write_text(json.dumps(r,indent=2)+'\n');receipt['rc']=r['rc'];print('matrix',r['rc'],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
