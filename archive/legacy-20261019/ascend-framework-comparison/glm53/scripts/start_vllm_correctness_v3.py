#!/usr/bin/env python3
"""Reviewed once-only real-model startup; no inference or benchmark in launcher."""
import concurrent.futures
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P = Path(__file__).resolve()
prior = P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest() == 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
spec = importlib.util.spec_from_file_location('base',prior)
b = importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
RUN = b.ROOT/'runs/20261017T0840-vllm-correctness-v3'
REMOTE = '/data/wio/inference-foundry/glm53/runs/20261017T0840-vllm-correctness-v3'
OVERLAY = '/data/wio/inference-foundry/glm53/overlays/vllm-sdma-v4-20261017'
SDK = '/data/wio/inference-foundry/glm53/overlays/sdk13-20261017'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
ENTRY = (P.parent/'vllm_correctness_entry.py').read_text()
assert hashlib.sha256(ENTRY.encode()).hexdigest() == '9595457ff22b2838a60650fd1006ab6ac21d0a8215cd21904bc2ab434491ada7'
stage_data = (b.ROOT/'config/vllm-overlay-stage-manifest-v4.json').read_bytes()
assert hashlib.sha256(stage_data).hexdigest() == 'b00b5893d67846dde1dde5b350351cfa4ad081215644198746e79e7d6fc17e04'
STAGED = json.loads(stage_data)
for entry in STAGED.values():
    assert hashlib.sha256((b.ROOT/entry['local']).read_bytes()).hexdigest() == entry['sha256']
inventory = (b.ROOT/'results/inventory-20261017.json').read_bytes()
INVENTORY_SHA = 'e6a8e981802cdf88199a0aca5413cfaf1df8b51eb2b834c13999cd72c3042581'
assert hashlib.sha256(inventory).hexdigest() == INVENTORY_SHA
METADATA = json.loads(inventory)['nodes'][0]['metadata_sha256']
SOURCE = json.loads((b.ROOT/'config/overlays/vllm-sdma-v3/manifest.json').read_text())
BOOTSTRAP = '''set -eu
export PYTHONPATH=/opt/glm53/source/python:/opt/glm53/sdk13:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1
exec python3 /run/glm53/entry.py "$1"
'''


def invoke_host(node, code, timeout=120):
    pod, hostname = b.NODES[node]
    mapping = b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
    assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
    code = 'import socket\nassert socket.gethostname()=='+repr(hostname)+'\n'+code
    command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
    return b.invoke([command],code,timeout=timeout)


def precheck(node):
    ports = list(range(26000,26036))+list(range(30000,30032))+[30300,29410,29580,29680,29681,29682] if node in (30,31) else list(range(27000,27048))+list(range(31000,31032))+[30400,29420]
    code = '''import pathlib,subprocess,json,hashlib,os
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15)
assert image.returncode==0 and image.stdout.strip()==IMAGE
sdk=pathlib.Path(SDK)
assert (sdk/'archive.sha256').read_text().strip()=='f858a88e60e33f2c954bd9920b56b96193e2be53749fa16e83316d461cb731b3'
for item in json.loads((sdk/'_glm53_overlay_manifest.json').read_text())['files']:
 assert hashlib.sha256((sdk/item['path']).read_bytes()).hexdigest()==item['sha256']
overlay=pathlib.Path(OVERLAY)
for name,item in STAGED.items():
 assert hashlib.sha256((overlay/name).read_bytes()).hexdigest()==item['sha256']
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8');files=[p for p in model.rglob('*') if p.is_file()]
assert len(files)==193 and sum(p.stat().st_size for p in files)==764240829290
for name,sha in METADATA.items():
 assert hashlib.sha256((model/name).read_bytes()).hexdigest()==sha
config=json.loads((model/'config.json').read_text());assert config['model_type']=='glm_moe_dsa' and config['architectures']==['GlmMoeDsaForCausalLM']
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
assert npu.returncode==0 and npu.stdout.count('No running processes')==8,'NPU occupied; no service changes'
for port in PORTS:
 s=socket.socket();s.bind(('0.0.0.0',port));s.close()
net=subprocess.run(['ip','-j','address','show'],capture_output=True,text=True,timeout=15);assert net.returncode==0
interfaces=[i['ifname'] for i in json.loads(net.stdout) if any(a.get('local')==IP for a in i.get('addr_info',[]))]
assert len(interfaces)==1,'Exact control interface must be unique'
memory=pathlib.Path('/proc/meminfo').read_text();available=int(next(l.split()[1] for l in memory.splitlines() if l.startswith('MemAvailable:')))*1024
assert available>800*1024**3,'Less than 800GiB available host RAM; preserve workload'
root.mkdir(parents=True);(root/'rpc').mkdir();(root/'cache').mkdir();(root/'entry.py').write_text(ENTRY)
(root/'precheck.json').write_text(json.dumps({'node':NODE,'image':IMAGE,'npu':npu.stdout,'meminfo':memory,'interface':interfaces[0],'model_metadata_sha256':METADATA},indent=2))
print(json.dumps({'node':NODE,'root':str(root),'interface':interfaces[0],'npu_empty_cards':8,'available_host_bytes':available,'model_metadata_verified':True}))
'''
    for key,value in [('REMOTE',REMOTE),('IMAGE',IMAGE),('SDK',SDK),('OVERLAY',OVERLAY),('STAGED',STAGED),('METADATA',METADATA),('PORTS',ports),('IP','172.16.12.'+str(node)),('ENTRY',ENTRY),('NODE',node)]:
        code=code.replace(key,repr(value))
    return invoke_host(node,code,timeout=160)


def start_meta():
    name='glm53-vllm-correctness-v3-meta-20261017-30'
    boot='''set -eu
export PYTHONPATH=/opt/glm53/sdk13:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1
exec python3 -c 'import memcache_hybrid as mc,sys; assert mc.__file__.startswith("/opt/glm53/sdk13/"); sys.exit(mc.MetaService.main())'
'''
    args=['docker','run','-d','--pull=never','--network=host','--pids-limit=2048','--memory=8g','--cpus=4',
          '--label','project_id=glm53','--label','run_id=20261017T0840-vllm-correctness-v3','--name',name,
          '-v',SDK+':/opt/glm53/sdk13:ro','-v',OVERLAY+'/config/meta.conf:/config/meta.conf:ro',
          '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
          '-e','MMC_META_CONFIG_PATH=/config/meta.conf','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
          '--log-opt','max-size=128m','--log-opt','max-file=8','--entrypoint','/bin/bash',IMAGE,'-c',boot]
    code='''import subprocess,time,json,sys
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=30)
assert r.returncode==0,r.stderr
cid=r.stdout.strip();assert len(cid)==64
ready=False;deadline=time.monotonic()+30
while time.monotonic()<deadline:
 try:
  with socket.create_connection(('172.16.12.30',29680),timeout=1):ready=True;break
 except OSError:time.sleep(.25)
state=subprocess.run(['docker','inspect',cid,'--format','{{json .State}}'],capture_output=True,text=True,timeout=10)
print(json.dumps({'container_id':cid,'name':NAME,'ready_tcp':ready,'state':state.stdout,'run_rc':r.returncode}))
sys.exit(0 if ready else 1)
'''.replace('ARGS',repr(args)).replace('NAME',repr(name))
    return invoke_host(30,code,timeout=90)


def start_model(node,interface):
    name=f'glm53-vllm-correctness-v3-20261017-{node}'
    args=['docker','run','-d','--pull=never','--network=host','--shm-size=64g','--memory=1700g','--pids-limit=32768',
          '--ulimit','memlock=-1','--ulimit','nofile=655360:655360',
          '--label','project_id=glm53','--label','run_id=20261017T0840-vllm-correctness-v3','--name',name,
          '--log-opt','max-size=128m','--log-opt','max-file=8',
          '-v',SDK+':/opt/glm53/sdk13:ro','-v',OVERLAY+'/source:/opt/glm53/source:ro',
          '-v',OVERLAY+'/config:/opt/glm53/config:ro','-v',REMOTE+':/run/glm53',
          '-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro',
          '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
    for entry in SOURCE['files']:
        args+=['-v',OVERLAY+'/source/'+entry['local']+':'+entry['container_path']+':ro']
    for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
        args+=['--device',dev]
    env={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'VLLM_HOST_IP':f'172.16.12.{node}','HCCL_IF_IP':f'172.16.12.{node}',
         'HCCL_SOCKET_IFNAME':interface,'GLOO_SOCKET_IFNAME':interface,
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn',
         'VLLM_CACHE_ROOT':'/run/glm53/cache','VLLM_RPC_BASE_PATH':'/run/glm53/rpc',
         'MMC_LOCAL_CONFIG_PATH':'/opt/glm53/config/local.conf','MMC_META_CONFIG_PATH':'/opt/glm53/config/meta.conf'}
    for key,value in env.items():args+=['-e',key+'='+value]
    args+=['--entrypoint','/bin/bash',IMAGE,'-c',BOOTSTRAP,'glm53-serve',str(node)]
    code='''import subprocess,json,pathlib,sys
root=pathlib.Path(REMOTE)
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
assert npu.returncode==0 and npu.stdout.count('No running processes')==8,'NPU became occupied; no model start'
with (root/'start-model.once').open('x') as f:f.write(NAME)
result=subprocess.run(ARGS,capture_output=True,text=True,timeout=45)
out={'node':NODE,'name':NAME,'run_rc':result.returncode,'container_id':result.stdout.strip(),'stderr':result.stderr,'args':ARGS}
(root/'start-result.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out));sys.exit(result.returncode)
'''
    for key,value in [('REMOTE',REMOTE),('NAME',name),('NODE',node),('ARGS',args)]:code=code.replace(key,repr(value))
    return invoke_host(node,code,timeout=90)


def main():
    # Do not start any resource unless the prior actual preflight completed.
    preflight=json.loads((b.ROOT/'runs/20261017T0830-vllm-cache-batch-preflight-v3/receipt.json').read_text())
    assert preflight['preflight_rc']==0 and all(x==0 for x in preflight['nodes'].values())
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),
             'model_requests':0,'phase':'STARTUP_ONLY_NOT_MODEL_CORRECTNESS','prechecks':{},'models':{},'meta':None}
    try:
        interfaces={}
        for node in (30,31,43,45):
            result=precheck(node);(RUN/f'precheck-{node}.json').write_text(json.dumps(result,indent=2))
            receipt['prechecks'][node]=result['rc'];assert result['rc']==0,'Precheck failed; no service starts'
            interfaces[node]=json.loads(result['stdout'])['interface']
        result=start_meta();(RUN/'meta-start.json').write_text(json.dumps(result,indent=2));receipt['meta']=result['rc']
        assert result['rc']==0,'Meta not ready; no model starts'
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            fs={pool.submit(start_model,node,interfaces[node]):node for node in (30,31,43,45)}
            for future in concurrent.futures.as_completed(fs):
                node=fs[future]
                try:result=future.result()
                except Exception as e:result={'error':repr(e)}
                (RUN/f'node-{node}.json').write_text(json.dumps(result,indent=2));receipt['models'][node]=result.get('rc',result.get('error'))
                print('model start',node,receipt['models'][node],flush=True)
    finally:
        (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
