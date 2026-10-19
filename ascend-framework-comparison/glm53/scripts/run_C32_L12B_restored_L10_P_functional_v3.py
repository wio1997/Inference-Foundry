#!/usr/bin/env python3
"""One bounded native SGLang functional PD matrix, no benchmark acceptance."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261019T-L12B-sglang-restored-L10-C32-functional-v3';REMOTE='/data/wio/inference-foundry/glm53/runs/20261019T-L12B-sglang-restored-L10-C32-functional-v3';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
capacity=json.loads((b.ROOT/'results/sglang-C32-L12B-restored-L10-P-pool-capacity-v1.json').read_text());assert capacity['pass']
IDS={int(n):item['container_id'] for n,item in capacity['providers'].items()}
FILES={n:(P.parent/n).read_text() for n in ['probe_C32_L12B_restored_pd_payload_v1.py','glm53_sglang_pd_gateway_v2.py']}
FROZEN={'probe_C32_L12B_restored_pd_payload_v1.py': 'ddc2a14f223f34e464882e0e8391a2b2431faa9936c25d3e1b02edf67b412c56', 'glm53_sglang_pd_gateway_v2.py': '80922bbe499ddf061b34297bd728e71693241724bc982b527b341eab6faf264f'}
for n,v in FILES.items():assert hashlib.sha256(v.encode()).hexdigest()==FROZEN[n],n

def invoke(node,code,timeout=100):
 pod,host=b.NODES[node];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']==f'172.16.12.{node}'
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def check(node):
 identity_path=P.parent/'glm53_C32_L12B_mixed_model_identity_v2.py'
 assert hashlib.sha256(identity_path.read_bytes()).hexdigest()=='c0189bfd2da9ce89cd479f092716d93efbbea66b729e39ac8b5240f411753320'
 spec=importlib.util.spec_from_file_location('mixed_identity',identity_path);identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
 code=identity.render(node,capacity)+"\nprint(json.dumps({'node':"+repr(node)+",'container_id':"+repr(IDS[node])+",'actual_model_identity':model_identity,'supervisor':supervisor,'provider_phase':provider['phase']}))\n"
 code+="\nimport urllib.request\nopener=urllib.request.build_opener(urllib.request.ProxyHandler({}))\nwith opener.open("+repr('http://172.16.12.'+str(node)+(':'+str(30500 if node in [34,35] else 30600))+'/metrics')+",timeout=10) as f: raw=f.read(8*1024**2+1);assert len(raw)<=8*1024**2\ngauges=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges)\n"
 return invoke(node,code)
def execute():
 args=['docker','run','--rm','--pull=never','--network=host','--cpus=4','--memory=4g','--pids-limit=256','--label','project_id=glm53','--label','run_id=20261019T-L12B-sglang-restored-L10-C32-functional-v3','--name','glm53-L12B-restored-L10-C32-functional-v3-20261019-34','-v',REMOTE+':/run/glm53','-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/run/glm53','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','CUDA_VISIBLE_DEVICES=','--entrypoint','timeout',IMAGE,'3100','python3','/run/glm53/probe_C32_L12B_restored_pd_payload_v1.py']
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
def plan():
 original=globals()['invoke'];globals()['invoke']=lambda n,code,timeout=100:code
 try:return {**{'identity-'+str(n):check(n) for n in [34,35,36,41]},'client':execute()}
 finally:globals()['invoke']=original

def main():
 fixture=json.loads((b.ROOT/'runs/20261017T1400-sglang-gateway-fixture-v2/receipt.json').read_text());assert fixture['rc']==0
 qualified=json.loads((b.ROOT/'results/sglang-C32-L12B-restored-L10-startup-native-v3.json').read_text());assert qualified['pass'] and qualified['all64_actual_SDMA_and_BF16_layout_verified'] and qualified['all32_D_nativeOps_fullmesh_and_targetGraph_retained'] and qualified['all32_P_legacy_host_IO_and_no_native_extension_verified']
 assert qualified['runtime_CIDs']=={str(n):cid for n,cid in IDS.items()}
 review=json.loads((b.ROOT/'results/C32-L12B-restored-short-actual-review-v3.json').read_text());assert review['host_codes_sha256']=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in plan().items()}
 RUN.mkdir(exist_ok=False)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'prechecks':{},'rc':None,'formal_requests':0}
 try:
  for n in [34,35,36,41]:
   r=check(n);(RUN/('precheck-'+str(n)+'.json')).write_text(json.dumps(r,indent=2)+'\n');receipt['prechecks'][str(n)]=r['rc'];assert r['rc']==0
  r=execute();(RUN/'node-34.json').write_text(json.dumps(r,indent=2)+'\n');receipt['rc']=r['rc'];print('matrix',r['rc'],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
