#!/usr/bin/env python3
"""One isolated CPU/client-only local-loopback AISBench fixture, zero inference."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0640-aisbench-client-fixture-v1'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T0640-aisbench-client-fixture-v1'
IMAGE='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
FROZEN={'aisbench_glm53_token_ids.py':'d7157131f0b7a536731e2f9921db4740c2fb62668731b504c2c687342d09e13c','aisbench_cpu_fixture.py':'8a4637d3e428a12c69ad9c91f17e3b6fdf5c0ac6ae947bc13dd2ec483d1611cf'}
FILES={name:(P.parent/name).read_text() for name in FROZEN}
for name,text in FILES.items():assert hashlib.sha256(text.encode()).hexdigest()==FROZEN[name]
identity_path=b.ROOT/'runs/20261017T0610-aisbench-assets-v1/identity-summary.json'
assert hashlib.sha256(identity_path.read_bytes()).hexdigest()=='7cf76483c63972186dc4c589fe04e4d8c49d8fa8eb020aab49fc4a10b955bb8f'
IDENTITY=json.loads(identity_path.read_text())


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,hostname=b.NODES[43]
 m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert m['rc']==0 and m['stdout']=='172.16.12.43'
 args=['docker','run','--rm','--pull=never','--network=none','--pids-limit=512','--memory=8g','--cpus=4',
       '--label','project_id=glm53','--name','glm53-aisbench-fixture-v1-20261017-43','--workdir','/run/glm53',
       '-v','/data/wio/aisbench-offline-43-20261014:/work:ro','-v',REMOTE+':/run/glm53',
       '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONNOUSERSITE=1','-e','PYTHONPATH=/run/glm53:/work/benchmark',
       '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false',
       '-e','CUDA_VISIBLE_DEVICES=','-e','ASCEND_RT_VISIBLE_DEVICES=',
       '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HOME=/run/glm53/hf',
       '-e','XDG_CACHE_HOME=/run/glm53/cache','-e','MPLCONFIGDIR=/run/glm53/mpl','-e','TORCH_HOME=/run/glm53/torch',
       '-e','NLTK_DATA=/work/nltk_data','-e','TIKTOKEN_CACHE_DIR=/work/tiktoken_cache',
       '--entrypoint','timeout',IMAGE,'180','/work/venv/bin/python','/run/glm53/aisbench_cpu_fixture.py']
 code='''import socket,pathlib,json,hashlib,subprocess,sys
assert socket.gethostname()==EXPECTED
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15)
assert image.returncode==0 and image.stdout.strip()==IMAGE
asset=pathlib.Path('/data/wio/aisbench-offline-43-20261014/benchmark/ais_bench/benchmark')
for entry in IDENTITY['files']:
 assert hashlib.sha256((asset/entry['path']).read_bytes()).hexdigest()==entry['sha256']
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
for name,text in FILES.items():
 assert pathlib.PurePosixPath(name).name==name
 (root/name).write_text(text)
result=subprocess.run(ARGS,capture_output=True,text=True,timeout=210)
print(result.stdout,end='');print(result.stderr,file=sys.stderr,end='')
sys.exit(result.returncode)
'''
 import re
 for key,value in [('EXPECTED',hostname),('IMAGE',IMAGE),('IDENTITY',IDENTITY),('REMOTE',REMOTE),('FILES',FILES),('ARGS',args)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 result=b.invoke([command],code,timeout=270);(RUN/'node-43.json').write_text(json.dumps(result,indent=2)+'\n')
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'files_sha256':FROZEN,'rc':result['rc'],'model_requests':0,'npu_compute_requests':0,'old_assets_readonly':True,'qualification':'CLIENT_FIXTURE_ONLY'}
 (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('fixture',result['rc'],flush=True)


if __name__=='__main__':main()
