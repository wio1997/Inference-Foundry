"""Once-only CPU43 fifth-tail inputs, after accepted C32 control/native closure."""
import ast, base64, hashlib, importlib.util, json, shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261017T2056-sglang-C32-fifth-suffix-inputs-v1'
REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
FILES={
 'generator.py':('scripts/glm53_C32_fifth_suffix_generator_v1.py','6b8d1d6bd256f1eb7a15dfb922236d73122bf2a9e664d9d7d9bbbcef30f344f6'),
 'reference.py':('scripts/prepare_formal_inputs_v3.py','27bb57b035aa4a4ee4a1a122ac9c518ca1889fb7b90d55b6fdcc895f7ffe65ed')}
HOST=r'''import pathlib,subprocess,json,hashlib,base64,socket
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
root=pathlib.Path(REMOTE);assert not root.exists();root.mkdir(parents=True)
image_check=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20)
assert image_check.returncode==0 and image_check.stdout.strip()==IMAGE
for name,item in FILES.items():
 raw=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['sha256'];(root/name).write_bytes(raw)
args=['docker','run','--rm','--pull=never','--network=none','--pids-limit=512','--memory=8g','--cpus=4','--label','project_id=glm53','--label','run_id='+RID,'--name','glm53-C32-fifth-suffix-inputs-v1-20261017-43',
 '-v',REMOTE+':/run/glm53','-v','/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts/C32:/canonical:ro',
 '-v','/data/wio/models/GLM-5.3-w8a8/tokenizer.json:/tokenizer.json:ro',
 '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','TOKENIZERS_PARALLELISM=false','-e','PYTHONDONTWRITEBYTECODE=1',
 '--entrypoint','timeout',IMAGE,'900','python3','/run/glm53/generator.py','--canonical','/canonical','--output','/run/glm53/artifacts',
 '--generator-reference','/run/glm53/reference.py','--generator-reference-sha256','27bb57b035aa4a4ee4a1a122ac9c518ca1889fb7b90d55b6fdcc895f7ffe65ed','--tokenizer','/tokenizer.json']
p=subprocess.run(args,capture_output=True,text=True,timeout=960)
(root/'generator-stdout.txt').write_text(p.stdout);(root/'generator-stderr.txt').write_text(p.stderr)
out={'project_id':'glm53','rc':p.returncode,'stderr':p.stderr,'model_requests':0,'SDK_init_calls':0,'NPU_devices_exposed':False}
if p.returncode==0:
 proof=json.loads(p.stdout);raw=(root/'artifacts/manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==proof['manifest_sha256']
 out['input_proof']=proof
else:out['stdout']=p.stdout
print(json.dumps(out))
'''
def render():
 payload={}
 for name,(relative,sha) in FILES.items():
  raw=(ROOT/relative).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
  payload[name]={'base64':base64.b64encode(raw).decode(),'sha256':sha}
 import re
 values={'REMOTE':REMOTE,'RID':RID,'IMAGE':IMAGE,'FILES':payload}
 source=re.sub(r'\b(?:REMOTE|RID|IMAGE|FILES)\b',lambda m:repr(values[m[0]]),HOST)
 ast.parse(source);return source
def main():
 certificate=json.loads((ROOT/'results/sglang-C32-L04-control-v3-measurement-v1.json').read_text())
 assert certificate['measurement_valid'] and certificate['run_id']=='20261017T2052-sglang-formal-C32-L04-control-v3' and certificate['formal_requests']==128
 receipt=json.loads((ROOT/'runs/20261017T2054-sglang-C32-L04-control-native-v3/receipt.json').read_text());assert set(receipt['nodes'])=={'34','35','36','41','43'} and all(v==0 for v in receipt['nodes'].values())
 run=ROOT/'runs'/RID;run.mkdir(parents=True,exist_ok=True)
 with (run/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
 path=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
 spec=importlib.util.spec_from_file_location('inventory',path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
 pod,_=base.NODES[43];mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapping['rc']==0 and mapping['stdout']=='172.16.12.43'
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 result=base.invoke([command],render(),timeout=1000)
 (run/'node-43.json').write_text(json.dumps(result,indent=2)+'\n')
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'node43_rc':result['rc'],'model_requests':0,'SDK_init_calls':0}
 if result['rc']==0:
  actual=json.loads(result['stdout']);receipt['generator_rc']=actual['rc']
  if actual['rc']==0:
   proof=actual['input_proof'];(run/'manifest-proof.json').write_text(json.dumps(proof,indent=2)+'\n');receipt['manifest_sha256']=proof['manifest_sha256']
 (run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
