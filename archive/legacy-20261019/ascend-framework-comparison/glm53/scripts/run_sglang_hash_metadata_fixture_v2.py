#!/usr/bin/env python3
"""One CPU-only native C++ hash oracle build and SDK metadata connectivity fixture."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1432-sglang-knownkey-metadata-fixture-v2';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1432-sglang-knownkey-metadata-fixture-v2';IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
SDK='/data/wio/inference-foundry/glm53/overlays/sdk13-20261017'
FILES={};FROZEN={'glm53_sglang_prefix_keys_v1.py':'a2188e0205d85615a35ca3dd80a5339a7e3a28707c778ecc882aeab6ac3a15e7','sglang_hash_metadata_cpu_fixture_v2.py':'99ee363ba341e17663ff52752ab3c4c68a9c1b7a0f8840b9fb320aa6692e7a92'}
for name,sha in FROZEN.items():
 raw=(P.parent/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha;FILES[name]=raw.decode()
idx=json.loads((b.ROOT/'runs/20261017T1420-sglang-cache-key-source-v1/code/index.json').read_text());NATIVE={}
for item in idx:
 if item['local'] in ['0-radix_cache.py','2-hash_binding.cpp','3-native_hash.py','7-utils.py']:
  raw=(b.ROOT/'runs/20261017T1420-sglang-cache-key-source-v1/code'/item['local']).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'];name=Path(item['path']).name;FILES[name]=raw.decode();NATIVE[name]=item['sha256']
assert len(NATIVE)==4;FILES['native-sha.json']=json.dumps(NATIVE,indent=2)+'\n'
FILES['metadata-local.conf']='ock.mmc.meta_service_url = tcp://172.16.12.34:29780\nock.mmc.local_service.config_store_url = tcp://172.16.12.34:29781\nock.mmc.log_level = info\nock.mmc.local_service.world_size = 128\nock.mmc.local_service.protocol = device_sdma\nock.mmc.local_service.dram.size = 0GB\nock.mmc.local_service.max.dram.size = 16GB\n'
HOST=r'''import socket,pathlib,subprocess,hashlib,json,sys
assert socket.gethostname()==HOSTNAME
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20);assert r.returncode==0 and r.stdout.strip()==IMAGE
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay';root.mkdir(parents=True)
for name,text in FILES.items():assert pathlib.PurePosixPath(name).name==name;(root/name).write_text(text)
with (root/'client-stdout.txt').open('w') as out,(root/'client-stderr.txt').open('w') as err:r=subprocess.run(ARGS,stdout=out,stderr=err,text=True,timeout=350)
result=json.loads((root/'result.json').read_text()) if (root/'result.json').is_file() else None
hash_result=json.loads((root/'hash-result.json').read_text()) if (root/'hash-result.json').is_file() else None
print('GLM53_HASH_FIXTURE_ARTIFACTS='+json.dumps({'client_rc':r.returncode,'result':result,'hash_result':hash_result,'artifacts':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file()}}),flush=True)
print((root/'client-stderr.txt').read_text()[-12000:],file=sys.stderr,end='');sys.exit(r.returncode)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,hostname=b.NODES[43];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']=='172.16.12.43'
 boot='export PYTHONPATH=/opt/glm53/sdk13:/run/glm53; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; exec timeout 320 python3 /run/glm53/sglang_hash_metadata_cpu_fixture_v2.py'
 args=['docker','run','--rm','--pull=never','--network=host','--memory=4g','--cpus=2','--pids-limit=512','--label','project_id=glm53','--name','glm53-knownkey-metadata-fixture-v2-20261017-43','--workdir','/run/glm53','-v',REMOTE+':/run/glm53','-v',SDK+':/opt/glm53/sdk13:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','MMC_LOCAL_CONFIG_PATH=/run/glm53/metadata-local.conf','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','CUDA_VISIBLE_DEVICES=','--entrypoint','/bin/bash',IMAGE,'-c',boot]
 code=HOST
 for k,v in [('HOSTNAME',hostname),('IMAGE',IMAGE),('REMOTE',REMOTE),('FILES',FILES),('ARGS',args)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 r=b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=410)
 (RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0,'NPU_devices':False,'init_bm':False},indent=2)+'\n');print('fixture',r['rc'],flush=True)
if __name__=='__main__':main()
