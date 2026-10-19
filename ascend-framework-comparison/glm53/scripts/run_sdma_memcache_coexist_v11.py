#!/usr/bin/env python3
"""One reviewed SDMA SDK pair on vLLM nodes30/43; local ZCode only."""
import concurrent.futures,hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0330-sdma-memcache-coexist-v11'
PAYLOAD=(P.parent/'sdma_memcache_coexist_v10.py').read_text()
assert hashlib.sha256(PAYLOAD.encode()).hexdigest()=='0bef66ae59001b9a589fe36355b1ede741d298495606568a3df20a0333f670b6'
GUARD=(P.parent/'memfabric_import_guard.py').read_text()
assert hashlib.sha256(GUARD.encode()).hexdigest()=='641601d437f5c610a85fbf551c1db023d22bfb92bea8ce468483b69d656fc150'
HELPER=(P.parent/'memcache_probe_helpers_v11.py').read_text()
assert hashlib.sha256(HELPER.encode()).hexdigest()=='bbe740e9c446f4dc12c72ae544cd2cffae4bc715b69e168858012480cbba5fdc'
MCGUARD=(P.parent/'memcache_import_guard.py').read_text()
assert hashlib.sha256(MCGUARD.encode()).hexdigest()=='aeb6dbc8bfa95a49aef8b17372188fc7736be3b8d1204233dd64cc0c935f2902'
PAYLOAD=GUARD+'\n'+MCGUARD+'\nGLM53_MF_GUARD_SOURCE='+repr(GUARD)+'\nGLM53_MC_GUARD_SOURCE='+repr(MCGUARD)+'\n'+HELPER+'\n'+PAYLOAD
BOOTSTRAP=(P.parent/'memfabric_bootstrap.sh').read_text()
assert hashlib.sha256(BOOTSTRAP.encode()).hexdigest()=='33d1a8cbb2ee1b10f6d0af659c973e918b2a10f044eede883423e65619197308'
IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
def execute(n):
 pod,hostname=b.NODES[n];ip='172.16.12.'+str(n)
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']==ip
 name='glm53-sdma-memcache-v11-20261017-'+str(n)
 args=['docker','run','--rm','--pull=never','--network=host','--shm-size=1g','--pids-limit=512','--memory=8g','--cpus=4','--ulimit','memlock=-1','--label','project_id=glm53','--name',name]
 for dev in ['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
 args+=['-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','ASCEND_RT_VISIBLE_DEVICES=0','--entrypoint','/bin/bash',IMAGE,'-c',BOOTSTRAP,'glm53-bootstrap',PAYLOAD,'Prefill' if n==30 else 'Decode','172.16.12.30',ip]
 hc='''import socket,subprocess,sys,pathlib,json,os
assert socket.gethostname()==HOSTNAME
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=10)
assert image.returncode==0 and image.stdout.strip()==IMAGE
info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
assert info.returncode==0 and info.stdout.count('No running processes')==8, 'NPU occupied; no changes'
for port in [38220,38221,38222,38223,38400,38401,38402]:
 s=socket.socket();s.bind(('0.0.0.0',port));s.close()
root=pathlib.Path('/data/wio/inference-foundry/glm53')
locks=root/'locks';locks.mkdir(parents=True,exist_ok=True)
lock=locks/'sdma-memcache-v11-20261017.once'
with lock.open('x') as f:f.write(json.dumps({'project_id':'glm53','pid':os.getpid(),'container':NAME}))
print('GLM53_PRECHECK_JSON='+json.dumps({'hostname':socket.gethostname(),'npu_before':info.stdout,'lock':str(lock)}),flush=True)
try:
 r=subprocess.run(ARGS,text=True,capture_output=True,timeout=160)
 print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');rc=r.returncode
except subprocess.TimeoutExpired as e:
 for raw,stream in [(e.stdout,sys.stdout),(e.stderr,sys.stderr)]:
  if raw:print(raw.decode(errors='replace') if isinstance(raw,bytes) else raw,file=stream,end='')
 print(str(e),file=sys.stderr);rc=124
 # Bound cleanup to exact owned diagnostic name and label, never another container.
 own=subprocess.run(['docker','inspect',NAME,'--format','{{index .Config.Labels "project_id"}}'],capture_output=True,text=True,timeout=10)
 if own.returncode==0 and own.stdout.strip()=='glm53':
  subprocess.run(['docker','stop','--time','5',NAME],capture_output=True,text=True,timeout=15)
  print('GLM53_OWNED_TIMEOUT_CLEANUP='+NAME,flush=True)
after=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
print('GLM53_POSTCHECK_JSON='+json.dumps({'rc':after.returncode,'npu_after':after.stdout}),flush=True)
sys.exit(rc)
'''.replace('HOSTNAME',repr(hostname)).replace('IMAGE',repr(IMAGE)).replace('NAME',repr(name)).replace('ARGS',repr(args))
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],hc,timeout=230)
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 results={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  fs={pool.submit(execute,n):n for n in (30,43)}
  for f in concurrent.futures.as_completed(fs):
   n=fs[f]
   try:r=f.result()
   except Exception as e:r={'error':repr(e)}
   (RUN/('node-%s.json'%n)).write_text(json.dumps(r,indent=2));results[n]=r.get('rc',r.get('error'));print(n,results[n],flush=True)
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'payload_sha256':hashlib.sha256(PAYLOAD.encode()).hexdigest(),'results':results,'qualification':'SDK_ONLY_NOT_MODEL_PD','model_requests':0},indent=2))
if __name__=='__main__':main()
