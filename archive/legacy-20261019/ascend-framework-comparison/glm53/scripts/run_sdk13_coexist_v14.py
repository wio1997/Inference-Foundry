#!/usr/bin/env python3
"""One reviewed SDMA SDK pair on vLLM nodes30/43; local ZCode only."""
import concurrent.futures,hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0410-sdk13-coexist-v14'
PAYLOAD=(P.parent/'sdma_memcache_coexist_v14.py').read_text()
assert hashlib.sha256(PAYLOAD.encode()).hexdigest()=='813715ef86515c1eac6665c2ee4517fc1796f274306afe9fcff4d06fac243dda'
HELPER=(P.parent/'memcache_probe_helpers_v14.py').read_text()
assert hashlib.sha256(HELPER.encode()).hexdigest()=='9fbfc3a71fa9637b39ed523013b319b5c16f6902b9f5685acf0dbf3128a58e06'
PAYLOAD=HELPER+'\n'+PAYLOAD
BOOTSTRAP=(P.parent/'sdk13_bootstrap.sh').read_text()
assert hashlib.sha256(BOOTSTRAP.encode()).hexdigest()=='7e7de87c2b2ba0b81b4af8b5c9a06820cd7985a392c14e742c0271887e6d7b51'
IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
def execute(n):
 pod,hostname=b.NODES[n];ip='172.16.12.'+str(n)
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']==ip
 name='glm53-sdk13-coexist-v14-20261017-'+str(n)
 args=['docker','run','--rm','--pull=never','--network=host','--shm-size=1g','--pids-limit=512','--memory=8g','--cpus=4','--ulimit','memlock=-1','--label','project_id=glm53','--name',name]
 args+=['-v','/data/wio/inference-foundry/glm53/overlays/sdk13-20261017:/opt/glm53/sdk13:ro']
 for dev in ['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
 args+=['-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','ASCEND_RT_VISIBLE_DEVICES=0','--entrypoint','/bin/bash',IMAGE,'-c',BOOTSTRAP,'glm53-bootstrap',PAYLOAD,'Prefill' if n==30 else 'Decode','172.16.12.30',ip]
 hc='''import socket,subprocess,sys,pathlib,json,os
assert socket.gethostname()==HOSTNAME
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=10)
assert image.returncode==0 and image.stdout.strip()==IMAGE
info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
assert info.returncode==0 and info.stdout.count('No running processes')==8, 'NPU occupied; no changes'
for port in [38220,38221,38222,38223]:
 s=socket.socket();s.bind(('0.0.0.0',port));s.close()
if True:
 overlay=pathlib.Path('/data/wio/inference-foundry/glm53/overlays/sdk13-20261017')
 import hashlib
 assert (overlay/'archive.sha256').read_text().strip()=='f858a88e60e33f2c954bd9920b56b96193e2be53749fa16e83316d461cb731b3'
 manifest=json.loads((overlay/'_glm53_overlay_manifest.json').read_text())
 for item in manifest['files']:
  assert hashlib.sha256((overlay/item['path']).read_bytes()).hexdigest()==item['sha256']
root=pathlib.Path('/data/wio/inference-foundry/glm53')
locks=root/'locks';locks.mkdir(parents=True,exist_ok=True)
lock=locks/'sdk13-coexist-v14-20261017.once'
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

META_NAME='glm53-sdk13-meta-v14-20261017-30'
REMOTE_RUN='/data/wio/inference-foundry/glm53/runs/20261017T0410-sdk13-coexist-v14'
META_CODE="""import memcache_hybrid as mc,pathlib,sys
assert mc.__file__.startswith('/opt/glm53/sdk13/')
print('GLM53_META_SDK='+mc.__file__+' VERSION='+pathlib.Path(mc.__file__).with_name('VERSION').read_text(),flush=True)
sys.exit(mc.MetaService.main())
"""

def meta_action(start, container_id=None):
 pod,hostname=b.NODES[30]
 m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert m['rc']==0 and m['stdout']=='172.16.12.30'
 if start:
  args=['docker','run','-d','--pull=never','--network=host','--pids-limit=512','--memory=3g','--cpus=2','--label','project_id=glm53','--name',META_NAME,
        '-v','/data/wio/inference-foundry/glm53/overlays/sdk13-20261017:/opt/glm53/sdk13:ro',
        '-v',REMOTE_RUN+'/meta.conf:/config/meta.conf:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
        '-e','MMC_META_CONFIG_PATH=/config/meta.conf','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
        '--entrypoint','/bin/bash',IMAGE,'-c','set -eu\nexport PYTHONPATH=/opt/glm53/sdk13:${PYTHONPATH:-}\nexport LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}\nexport PYTHONDONTWRITEBYTECODE=1\nexec timeout 240 python3 -c "$1"','glm53-meta',META_CODE]
  hc='''import socket,pathlib,subprocess,json,sys,hashlib
assert socket.gethostname()==HOSTNAME
sdk=pathlib.Path('/data/wio/inference-foundry/glm53/overlays/sdk13-20261017')
assert (sdk/'archive.sha256').read_text().strip()=='f858a88e60e33f2c954bd9920b56b96193e2be53749fa16e83316d461cb731b3'
manifest=json.loads((sdk/'_glm53_overlay_manifest.json').read_text())
for item in manifest['files']:
 assert hashlib.sha256((sdk/item['path']).read_bytes()).hexdigest()==item['sha256']
for port in [38400,38401,38402]:
 s=socket.socket();s.bind(('0.0.0.0',port));s.close()
root=pathlib.Path(REMOTE_RUN);root.mkdir(parents=True,exist_ok=False)
(root/'meta.conf').write_text('ock.mmc.meta_service_url = tcp://172.16.12.30:38400\\nock.mmc.meta_service.config_store_url = tcp://172.16.12.30:38401\\nock.mmc.meta_service.metrics_url = http://172.16.12.30:38402\\nock.mmc.log_level = info\\n')
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=30)
print(json.dumps({'rc':r.returncode,'container_id':r.stdout.strip(),'stderr':r.stderr}));sys.exit(r.returncode)
'''.replace('HOSTNAME',repr(hostname)).replace('REMOTE_RUN',repr(REMOTE_RUN)).replace('ARGS',repr(args))
 else:
  assert container_id and len(container_id)==64
  hc='''import socket,subprocess,json,sys
assert socket.gethostname()==HOSTNAME
cid=CID
own=subprocess.run(['docker','inspect',cid,'--format','{{index .Config.Labels "project_id"}} {{.Name}}'],capture_output=True,text=True,timeout=10)
assert own.returncode==0 and own.stdout.strip()=='glm53 /'+NAME
stopped=subprocess.run(['docker','stop','--time','10',cid],capture_output=True,text=True,timeout=20)
logs=subprocess.run(['docker','logs',cid],capture_output=True,text=True,timeout=10)
state=subprocess.run(['docker','inspect',cid,'--format','{{json .State}}'],capture_output=True,text=True,timeout=10)
result={'stop_rc':stopped.returncode,'stop_stderr':stopped.stderr,'stdout':logs.stdout,'stderr':logs.stderr,'state':state.stdout,'inspect_rc':state.returncode}
if stopped.returncode==0:
 removed=subprocess.run(['docker','rm',cid],capture_output=True,text=True,timeout=10);result['remove_rc']=removed.returncode
print(json.dumps(result));sys.exit(0 if stopped.returncode==0 else 1)
'''.replace('HOSTNAME',repr(hostname)).replace('CID',repr(container_id)).replace('NAME',repr(META_NAME))
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],hc,timeout=80)

def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 results={};cid=None;meta_stop=None
 try:
  started=meta_action(True);(RUN/'meta-start.json').write_text(json.dumps(started,indent=2));assert started['rc']==0
  details=json.loads(started['stdout']);assert details['rc']==0;cid=details['container_id']
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(execute,n):n for n in (30,43)}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/('node-%s.json'%n)).write_text(json.dumps(r,indent=2));results[n]=r.get('rc',r.get('error'));print(n,results[n],flush=True)
 finally:
  if cid:
   meta_stop=meta_action(False,cid);(RUN/'meta-stop.json').write_text(json.dumps(meta_stop,indent=2))
  (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'payload_sha256':hashlib.sha256(PAYLOAD.encode()).hexdigest(),'results':results,'meta_stop_rc':None if meta_stop is None else meta_stop['rc'],'model_requests':0,'qualification':'SDK_CACHE_ONLY_NOT_MODEL_PD'},indent=2))
if __name__=='__main__':main()
