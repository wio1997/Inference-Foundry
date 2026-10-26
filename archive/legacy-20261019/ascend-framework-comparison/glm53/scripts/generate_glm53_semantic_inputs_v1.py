#!/usr/bin/env python3
"""Once-only CPU tokenizer guard-dataset generation; formal448inputs unchanged."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('b',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0920-semantic-inputs-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T0920-semantic-inputs-v1';IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
PAYLOAD=(P.parent/'prepare_glm53_semantic_inputs_v1.py').read_text();assert hashlib.sha256(PAYLOAD.encode()).hexdigest()=='7a04e5f521ed38b64372b3d74abb9a866a2c61444dbc6a26890496ac7cb519f4'
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,host=b.NODES[43];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.43'
 args=['docker','run','--rm','--pull=never','--network=none','--pids-limit=512','--memory=8g','--cpus=4','--label','project_id=glm53','--name','glm53-semantic-inputs-v1-20261017-43','--workdir','/run/glm53','-v',REMOTE+':/run/glm53','-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro','-v','/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts/C16:/dataset:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HOME=/run/glm53/hf','-e','XDG_CACHE_HOME=/run/glm53/cache','--entrypoint','timeout',IMAGE,'180','python3','/run/glm53/generator.py']
 code=r'''import socket,subprocess,pathlib,json,hashlib,sys
assert socket.gethostname()==HOST
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert image.returncode==0 and image.stdout.strip()==IMAGE
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False);(root/'generator.py').write_text(PAYLOAD)
with (root/'generator-stdout.txt').open('w') as out,(root/'generator-stderr.txt').open('w') as err:r=subprocess.run(ARGS,stdout=out,stderr=err,text=True,timeout=210)
(root/'generator-exit.json').write_text(json.dumps({'rc':r.returncode}))
print((root/'generator-stdout.txt').read_text()[-20000:],end='');print((root/'generator-stderr.txt').read_text()[-12000:],file=sys.stderr,end='')
p=root/'manifest.json'
if p.is_file():print('GLM53_SEMANTIC_MANIFEST_EVIDENCE='+json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'manifest':json.loads(p.read_text())}),flush=True)
sys.exit(r.returncode)
'''
 import re
 for k,v in [('HOST',host),('IMAGE',IMAGE),('REMOTE',REMOTE),('PAYLOAD',PAYLOAD),('ARGS',args)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 r=b.invoke([cmd],code,timeout=270);(RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0,'npu_compute_requests':0},indent=2)+'\n');print('semanticgenerator',r['rc'],flush=True)
if __name__=='__main__':main()
