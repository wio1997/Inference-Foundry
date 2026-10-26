#!/usr/bin/env python3
"""One bounded native SGLang functional PD matrix, no benchmark acceptance."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1735-sglang-shared-pd-matrix-v6';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1735-sglang-shared-pd-matrix-v6';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
IDS={34: '10d14e531385739498eacfb08d75d6cc3b370cf78f74af14e9fae6736bbb744e', 35: 'eb04fad0928296da105ae2a0d90b16ef273550e17facebba56eab88ba6aa795d', 36: '0d02ac7770dab23c14a55fe18b0dc8f7d5999b455926403b91227b9d95bcf20d', 41: 'f77216e73f51473383da9f6c7fe46e6ba40338db890ed0db656d18a032225fc6'}
FILES={n:(P.parent/n).read_text() for n in ['probe_sglang_pd_matrix_payload_v2.py','glm53_sglang_pd_gateway_v2.py']}
FROZEN={'probe_sglang_pd_matrix_payload_v2.py': 'e663d0c66de2bb281754e7fac1bfedc47a25325e76bd564e9b12b2822fec67fa', 'glm53_sglang_pd_gateway_v2.py': '80922bbe499ddf061b34297bd728e71693241724bc982b527b341eab6faf264f'}
for n,v in FILES.items():assert hashlib.sha256(v.encode()).hexdigest()==FROZEN[n],n

def invoke(node,code,timeout=100):
 pod,host=b.NODES[node];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']==f'172.16.12.{node}'
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def check(node):
 identity_path=P.parent/'glm53_shared_model_identity_v1.py'
 assert hashlib.sha256(identity_path.read_bytes()).hexdigest()=='e89e2a6a45878328ea0772849bf629a76a177abbfb58fb27988c36c8cf6a6ec4'
 spec=importlib.util.spec_from_file_location('model_identity',identity_path);identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
 code=identity.HOST+"\nprint(json.dumps({'node':NODE,'container_id':CID,'actual_model_identity':model_identity,'supervisor':supervisor,'provider_phase':provider['phase']}))\n"
 for k,v in [('CID',IDS[node]),('NAME',f'glm53-sglang-shared-v2-20261017-{node}'),('IMAGE',IMAGE),('NODE',node)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return invoke(node,code)
def execute():
 args=['docker','run','--rm','--pull=never','--network=host','--cpus=4','--memory=4g','--pids-limit=256','--label','project_id=glm53','--label','run_id=20261017T1735-sglang-shared-pd-matrix-v6','--name','glm53-sglang-shared-pd-matrix-v6-20261017-34','-v',REMOTE+':/run/glm53','-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/run/glm53','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','CUDA_VISIBLE_DEVICES=','--entrypoint','timeout',IMAGE,'3100','python3','/run/glm53/probe_sglang_pd_matrix_payload_v2.py']
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
 audit=b.ROOT/'runs/20261017T1731-sglang-shared-startup-checkpoint-v2'
 for node in [34,35,36,41]:
  outer=json.loads((audit/f'node-{node}.json').read_text());assert outer['rc']==0;o=json.loads(outer['stdout']);assert o['model']['state']['Running']
  metrics=o['/metrics'].get('body','');assert metrics
  gauges=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(x==0 for x in gauges)
 assert hashlib.sha256((b.ROOT/'results/sglang-shared-startup-v2.json').read_bytes()).hexdigest()=='6af878738b6522792292cca5e7c4ecbb74030e6b16a6148ade87b4685d916ddf'
 qualified=json.loads((b.ROOT/'results/sglang-shared-startup-v2.json').read_text());assert qualified['nodes']['34']['native_warmup_completion_logs'] and qualified['nodes']['36']['native_warmup_completion_logs']
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
