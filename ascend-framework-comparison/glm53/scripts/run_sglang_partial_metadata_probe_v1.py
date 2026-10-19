#!/usr/bin/env python3
"""Hardware-free metadata observation of existing failed-run keys, no warmup."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RID='20261017T1535-sglang-partial-prefix-metadata-v1';RUN=b.ROOT/'runs'/RID;REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
FROZEN={'glm53_sglang_prefix_keys_v1.py':'a2188e0205d85615a35ca3dd80a5339a7e3a28707c778ecc882aeab6ac3a15e7','sglang_partial_prefix_metadata_v1.py':'0934ec98e6037bdade933f298b2cd17e837da92fd231f51029f46d34256d307c'}
FILES={}
for name,sha in FROZEN.items():
 raw=(P.parent/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha;FILES[name]=raw.decode()
FILES['metadata-local.conf']='ock.mmc.meta_service_url = tcp://172.16.12.34:29780\nock.mmc.local_service.config_store_url = tcp://172.16.12.34:29781\nock.mmc.log_level = info\nock.mmc.local_service.world_size = 128\nock.mmc.local_service.protocol = device_sdma\nock.mmc.local_service.dram.size = 0GB\nock.mmc.local_service.max.dram.size = 16GB\n'
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,host=b.NODES[43];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.43'
 boot='export PYTHONPATH=/opt/glm53/sdk13:/run/glm53; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; exec timeout 120 python3 /run/glm53/sglang_partial_prefix_metadata_v1.py'
 args=['docker','run','--rm','--pull=never','--network=host','--memory=4g','--cpus=2','--pids-limit=512','--label','project_id=glm53','--label','run_id='+RID,'--name','glm53-sglang-partial-metadata-v1-20261017-43','--workdir','/run/glm53','-v',REMOTE+':/run/glm53:ro','-v','/data/wio/inference-foundry/glm53/overlays/sdk13-20261017:/opt/glm53/sdk13:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-v','/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1:/semantic:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','MMC_LOCAL_CONFIG_PATH=/run/glm53/metadata-local.conf','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','ASCEND_RT_VISIBLE_DEVICES=','-e','CUDA_VISIBLE_DEVICES=','--entrypoint','/bin/bash',IMAGE,'-c',boot]
 code=r'''import socket,pathlib,subprocess,json,hashlib,sys
assert socket.gethostname()==HOST
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and r.stdout.strip()==IMAGE
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
for name,text in FILES.items():(root/name).write_text(text)
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=150)
(root/'stdout.txt').write_text(r.stdout);(root/'stderr.txt').write_text(r.stderr)
print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)
'''
 for k,v in [('HOST',host),('IMAGE',IMAGE),('REMOTE',REMOTE),('FILES',FILES),('ARGS',args)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 r=b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=210)
 (RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n')
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0,'NPU_devices':False,'metadata_only':True},indent=2)+'\n')
 print('metadata',r['rc'],flush=True)
if __name__=='__main__':main()
