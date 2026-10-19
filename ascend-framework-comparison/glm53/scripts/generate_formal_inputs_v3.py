#!/usr/bin/env python3
"""Once-only CPU input generation in fixed rc1, no NPU or model requests."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0615-formal-inputs-v3'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3'
PAYLOAD=(P.parent/'prepare_formal_inputs_v3.py').read_text()
PAYLOAD_SHA='27bb57b035aa4a4ee4a1a122ac9c518ca1889fb7b90d55b6fdcc895f7ffe65ed'
assert hashlib.sha256(PAYLOAD.encode()).hexdigest()==PAYLOAD_SHA
IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
GENERATE_CODE='''import runpy,json,importlib.metadata as md
module=runpy.run_path(REMOTE+'/generator.py')
print('GLM53_INPUT_RUNTIME='+json.dumps({'tokenizers':md.version('tokenizers')}),flush=True)
for concurrency in (16,32,64):
 result=module['generate'](REMOTE+'/artifacts/C'+str(concurrency),'/data/wio/models/GLM-5.3-w8a8/tokenizer.json',concurrency)
 print('GLM53_INPUT_DATASET='+json.dumps({k:v for k,v in result.items() if k not in ('prefixes','formal')}),flush=True)
'''.replace('REMOTE',repr(REMOTE))


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,hostname=b.NODES[43]
 m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert m['rc']==0 and m['stdout']=='172.16.12.43'
 args=['docker','run','--rm','--pull=never','--network=none','--pids-limit=512','--memory=8g','--cpus=8',
       '--label','project_id=glm53','--name','glm53-input-generator-v3-20261017-43',
       '-v',REMOTE+':'+REMOTE,
       '-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/data/wio/models/GLM-5.3-w8a8/tokenizer.json:ro',
       '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false',
       '-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','timeout',IMAGE,'900','python3','-c',GENERATE_CODE]
 code='''import pathlib,socket,subprocess,json,sys,hashlib
assert socket.gethostname()==EXPECTED
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15)
assert image.returncode==0 and image.stdout.strip()==IMAGE
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
tokenizer=pathlib.Path('/data/wio/models/GLM-5.3-w8a8/tokenizer.json')
assert hashlib.sha256(tokenizer.read_bytes()).hexdigest()=='19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
root.mkdir(parents=True);(root/'generator.py').write_text(PAYLOAD)
result=subprocess.run(ARGS,capture_output=True,text=True,timeout=930)
print(result.stdout,end='');print(result.stderr,file=sys.stderr,end='')
manifests={}
for concurrency in (16,32,64):
 path=root/'artifacts'/('C'+str(concurrency))/'manifest.json'
 if path.is_file():manifests[concurrency]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'data':json.loads(path.read_text())}
print('GLM53_INPUT_MANIFESTS='+json.dumps(manifests),flush=True)
sys.exit(result.returncode)
'''
 import re
 for key,value in [('EXPECTED',hostname),('IMAGE',IMAGE),('REMOTE',REMOTE),('PAYLOAD',PAYLOAD),('ARGS',args)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 result=b.invoke([command],code,timeout=990);(RUN/'node-43.json').write_text(json.dumps(result,indent=2)+'\n')
 manifests={}
 for line in result['stdout'].splitlines():
  if line.startswith('GLM53_INPUT_MANIFESTS='):manifests=json.loads(line.split('=',1)[1])
 for concurrency,data in manifests.items():(RUN/('C'+concurrency+'-manifest.json')).write_text(json.dumps(data,indent=2)+'\n')
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'generator_sha256':PAYLOAD_SHA,
          'rc':result['rc'],'generated_concurrency':list(manifests),'model_requests':0,'npu_compute_requests':0,'remote':REMOTE}
 (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('generator',result['rc'],list(manifests),flush=True)


if __name__=='__main__':main()
