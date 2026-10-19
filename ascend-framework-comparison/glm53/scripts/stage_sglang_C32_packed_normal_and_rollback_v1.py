#!/usr/bin/env python3
"""Stage new own files and validate SDK config objects; no engines or NPU access."""
import base64,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261018T-L09-sglang-packed-normal-stage-v1'
MANIFESTS={'config/sglang-C32-packed-normal-candidate-stage-v1.json': '0319ebb838216e81910bf896ded1ae0d7a3609aa8e30573935a0fd8bb79fe757', 'config/sglang-C32-packed-normal-rollback-stage-v1.json': '36e1e2b3cfd4facf964f222446799bbfb4ddd78e52cdc78d79983a71b622fa44'}
BUNDLES=[]
for path,expected in MANIFESTS.items():
 M=b.ROOT/path;assert hashlib.sha256(M.read_bytes()).hexdigest()==expected;BUNDLES.extend(json.loads(M.read_text())['bundles'])
for bundle in BUNDLES:
 for f in bundle['files'].values():assert hashlib.sha256((b.ROOT/f['local']).read_bytes()).hexdigest()==f['sha256']
def invoke(node,code,timeout=120):
 pod,host=b.NODES[node];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']==f'172.16.12.{node}'
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def replace(code,pairs):
 for key,value in pairs:code=re.sub(r'\b'+key+r'\b',lambda _:repr(value),code)
 return code
def stage(bundle,node):
 files={name:{'data':base64.b64encode((b.ROOT/entry['local']).read_bytes()).decode(),'sha256':entry['sha256']} for name,entry in bundle['files'].items()}
 code=r'''import pathlib,subprocess,json,base64,hashlib
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and r.stdout.strip()==IMAGE
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
for name,item in FILES.items():
 p=pathlib.PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
 assert hashlib.sha256(base64.b64decode(item['data'])).hexdigest()==item['sha256']
root.mkdir(parents=True)
for name,item in FILES.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(item['data']));assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
print(json.dumps({'node':NODE,'framework':FRAMEWORK,'remote':str(root),'files':len(FILES),'new_source_staged':True,'existing_models_changed':False}))
'''
 return invoke(node,replace(code,[('IMAGE',bundle['image']),('REMOTE',bundle['remote']),('FILES',files),('NODE',node),('FRAMEWORK',bundle['framework'])]))
def preflight(bundle):
 fw=bundle['framework'];node=bundle['nodes'][0];remote=bundle['remote']
 args=['docker','run','--rm','--pull=never','--network=none','--cpus=2','--memory=4g','--pids-limit=256','--tmpfs','/run/glm53:rw,size=32m','--label','project_id=glm53','--name',f'glm53-{fw}-buffer-dedup-config-20261017-{node}','-v',remote+':/opt/glm53/staged:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1']
 if fw=='vllm':args+=['-v','/data/wio/inference-foundry/glm53/overlays/sdk13-20261017:/opt/glm53/sdk13:ro']
 boot='set -eu; '
 if fw=='vllm':boot+='export PYTHONPATH=/opt/glm53/sdk13:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; '
 boot+='exec python3 /opt/glm53/staged/sdk-preflight.py --framework "$1" --root /opt/glm53/staged'
 args+=['--entrypoint','/bin/bash',bundle['image'],'-c',boot,'glm53-sdk-config',fw]
 code=r'''import pathlib,hashlib,subprocess,sys
root=pathlib.Path(REMOTE)
for name,entry in FILES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==entry['sha256']
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=90);print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)
'''
 return invoke(node,replace(code,[('REMOTE',remote),('FILES',bundle['files']),('ARGS',args)]),timeout=150)
def main():
 cpu=json.loads((b.ROOT/'results/C32-L09-packed-dispatch-image-CPU-qualification-v1.json').read_text());assert cpu['pass'] and cpu['actual_NPU_correctness']=='PENDING' and cpu['rank_quant_cases']==448
 hook=json.loads((b.ROOT/'results/C32-L09-packed-hook-pure-qualification-v1.json').read_text());assert hook['rc']==0 and hook['result']['pass'] and len(hook['result']['cases'])==7
 restored=json.loads((b.ROOT/'results/sglang-C32-window-L04-rollback-startup-v2.json').read_text());assert restored['pass'] and restored['all32_D_actual_first_use_verified']
 previous=json.loads((b.ROOT/'runs/20261017T2064-sglang-C32-native-ops-stage-v1/receipt.json').read_text());assert len(previous['SDK_preflight'])==2 and all(v==0 for v in previous['SDK_preflight'].values())
 # SDK configuration/nativeCPUobjects/source unchanged; do not repeat their preflight.
 for bundle in BUNDLES:
  for name,entry in bundle['files'].items():
   if name.startswith('provider-config/') or name in ['provider.py','sdk-preflight.py','config/meta.conf','config/local.conf']:
    original=json.loads((b.ROOT/'config/sglang-C32-native-ops-rollback-v1-stage-v1.json').read_text())['bundles'][0]['files'][name]
    assert entry['sha256']==original['sha256']
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'manifest_sha256':MANIFESTS,'staged':{},'SDK_preflight':{},'SDK_configuration_preflight_reused_by_exact_source':True,'model_requests':0,'SDK_init_calls':0,'existing_services_changed':False}
 try:
  for bundle in BUNDLES:
   fw=bundle['framework'];key=bundle['remote'].rsplit('/',1)[1]
   for node in bundle['nodes']:
    r=stage(bundle,node);(RUN/f'{key}-stage-{node}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['staged'][bundle['remote']+':'+str(node)]=r['rc'];assert r['rc']==0
   receipt['SDK_preflight'][bundle['remote']]='REUSED_UNCHANGED2064_CONFIG_AND_SDK_SOURCE_NOT_RERUN'
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
