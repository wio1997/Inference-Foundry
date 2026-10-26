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
RUN = b.ROOT/'runs/20261017T1561-sglang-correctness-v6-v2'
REMOTE = '/data/wio/inference-foundry/glm53/runs/20261017T1561-sglang-correctness-v6-v2'
OVERLAY = '/data/wio/inference-foundry/glm53/overlays/sglang-sdma-v6-20261017'
IMAGE = 'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
ENTRY=(P.parent/'sglang_correctness_entry_v1.py').read_text()
assert hashlib.sha256(ENTRY.encode()).hexdigest()=='3c5652b0c7143387f47c7add7c600e0ee085a74cd6a1632f0ef92201fdd5c3a8'
inventory = (b.ROOT/'results/inventory-20261017.json').read_bytes()
INVENTORY_SHA = 'e6a8e981802cdf88199a0aca5413cfaf1df8b51eb2b834c13999cd72c3042581'
assert hashlib.sha256(inventory).hexdigest() == INVENTORY_SHA
METADATA = json.loads(inventory)['nodes'][0]['metadata_sha256']
SOURCE_PATH=b.ROOT/'config/overlays/sglang-sdma-v1/manifest.json'
assert hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()=='0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528'
SOURCE=json.loads(SOURCE_PATH.read_text())
STAGED={'source/sglang/srt/disaggregation/ascend/transfer_engine.py': 'b8824bd9c8365bda4c584d18e3fb0d19a47f083d00c163c65ded3f95db5570b8', 'source/sglang/srt/disaggregation/ascend/conn.py': 'b49d78a04966c13e95758d30ee66574436e0034dfc24fdb12ecad2bbc5dfd547', 'python/glm53_sglang_sdma_audit.py': '22ae47b2c4d6bef5dd83fd1377766e35aef51a61259331db20690a984bd92948', 'manifest.json': '0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528', 'config/meta.conf': '4dd3177275a91891bac5ef561d33273e9b545cbd182ba351cb00098d31701f0b', 'config/local.conf': '503c58548e227107bf8d6fde95fa1c78e72eeaa89239af2200455d4906c7c728', 'config/node-34.json': '16599139e9f3c2d29c3ecfa1096bb120ed1dd17b219952b6ee90512a6e148562', 'config/node-35.json': '430ec01bbf382b5ce2e76040e3536cee5ed05d74865479fda8633d03d4b38513', 'config/node-36.json': '84cd938dd1cd2b1cbb815f06e49f49fd6c3659466e01e5535cce211c9f5a7bce', 'config/node-41.json': '6eb96125ab8d2446bda2a5ed56c53035a92a92aa9c38e4fd8e142160dd080aef'}
BOOTSTRAP = '''set -eu
export PYTHONPATH=/opt/glm53/source/python:${PYTHONPATH:-}
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
    ports=[29430,30500,29898,29590,29780,29781,29782] if node in (34,35) else [29440,30600]
    code = '''import pathlib,subprocess,json,hashlib,os
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
image=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15)
assert image.returncode==0 and image.stdout.strip()==IMAGE
overlay=pathlib.Path(OVERLAY)
for name,sha in STAGED.items():
 assert hashlib.sha256((overlay/name).read_bytes()).hexdigest()==sha
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
    import re
    for key,value in [('REMOTE',REMOTE),('IMAGE',IMAGE),('OVERLAY',OVERLAY),('STAGED',STAGED),('METADATA',METADATA),('PORTS',ports),('IP','172.16.12.'+str(node)),('ENTRY',ENTRY),('NODE',node)]:
        code=re.sub(r'\b'+key+r'\b',lambda _:repr(value),code)
    return invoke_host(node,code,timeout=160)


def start_meta():
    name='glm53-sglang-correctness-v6-meta-20261017-34'
    boot='''set -eu
export PYTHONDONTWRITEBYTECODE=1
exec python3 -c 'import memcache_hybrid as mc,sys; from importlib.metadata import version; assert version("memcache_hybrid")=="1.3.0"; sys.exit(mc.MetaService.main())'
'''
    args=['docker','run','-d','--pull=never','--network=host','--pids-limit=2048','--memory=8g','--cpus=4',
          '--label','project_id=glm53','--label','run_id=20261017T1561-sglang-correctness-v6-v2','--name',name,
          '-v',OVERLAY+'/config/meta.conf:/config/meta.conf:ro',
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
  with socket.create_connection(('172.16.12.34',29780),timeout=1):ready=True;break
 except OSError:time.sleep(.25)
state=subprocess.run(['docker','inspect',cid,'--format','{{json .State}}'],capture_output=True,text=True,timeout=10)
print(json.dumps({'container_id':cid,'name':NAME,'ready_tcp':ready,'state':state.stdout,'run_rc':r.returncode}))
sys.exit(0 if ready else 1)
'''.replace('ARGS',repr(args)).replace('NAME',repr(name))
    return invoke_host(34,code,timeout=90)


def start_model(node,interface):
    name=f'glm53-sglang-correctness-v6-20261017-{node}'
    args=['docker','run','-d','--pull=never','--network=host','--shm-size=64g','--memory=1700g','--pids-limit=32768',
          '--ulimit','memlock=-1','--ulimit','nofile=655360:655360',
          '--label','project_id=glm53','--label','run_id=20261017T1561-sglang-correctness-v6-v2','--name',name,
          '--log-opt','max-size=128m','--log-opt','max-file=8',
          '-v',OVERLAY+':/opt/glm53/source:ro','-v',REMOTE+':/run/glm53',
          '-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro',
          '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
    for entry in SOURCE['files']:
        args+=['-v',OVERLAY+'/'+entry['local']+':'+entry['container_path']+':ro']
    for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
        args+=['--device',dev]
    config_path=b.ROOT/f'config/sglang-correctness-v7/node-{node}.json'
    assert hashlib.sha256(config_path.read_bytes()).hexdigest()==STAGED[f'config/node-{node}.json']
    config=json.loads(config_path.read_text())
    env={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'HCCL_IF_IP':f'172.16.12.{node}','HCCL_SOCKET_IFNAME':interface,'GLOO_SOCKET_IFNAME':interface,
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/run/glm53/hf','XDG_CACHE_HOME':'/run/glm53/cache',
         'PYTHONDONTWRITEBYTECODE':'1','SGLANG_MOONCAKE_SEND_AUX_TCP':'0',
         'MMC_LOCAL_CONFIG_PATH':'/opt/glm53/source/config/local.conf','MMC_META_CONFIG_PATH':'/opt/glm53/source/config/meta.conf',
         **config['environment_intent']}
    for key,value in env.items():args+=['-e',key+'='+value]
    args+=['--entrypoint','/bin/bash',IMAGE,'-c',BOOTSTRAP,'glm53-sglang-serve',str(node)]
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
    import re
    for key,value in [('REMOTE',REMOTE),('NAME',name),('NODE',node),('ARGS',args)]:code=re.sub(r'\b'+key+r'\b',lambda _:repr(value),code)
    return invoke_host(node,code,timeout=90)


def main():
    # Do not start any resource unless the prior actual preflight completed.
    preflight=json.loads((b.ROOT/'runs/20261017T1555-sglang-buffer-only-preflight-v3/receipt.json').read_text())
    assert preflight['preflight_rc']==0 and all(x==0 for x in preflight['stage_rc'].values())
    cleanup=json.loads((b.ROOT/'runs/20261017T1551-stop-sglang-v5-v2/receipt.json').read_text());assert cleanup['meta_rc']==0 and all(x==0 for x in cleanup['nodes'].values())
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),
             'model_requests':0,'phase':'STARTUP_ONLY_NOT_MODEL_CORRECTNESS','prechecks':{},'models':{},'meta':None}
    try:
        interfaces={}
        for node in (34,35,36,41):
            result=precheck(node);(RUN/f'precheck-{node}.json').write_text(json.dumps(result,indent=2))
            receipt['prechecks'][node]=result['rc'];assert result['rc']==0,'Precheck failed; no service starts'
            interfaces[node]=json.loads(result['stdout'])['interface']
        result=start_meta();(RUN/'meta-start.json').write_text(json.dumps(result,indent=2));receipt['meta']=result['rc']
        assert result['rc']==0,'Meta not ready; no model starts'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fs={pool.submit(start_model,node,interfaces[node]):node for node in (34,35,36,41)}
            for future in concurrent.futures.as_completed(fs):
                node=fs[future]
                try:result=future.result()
                except Exception as e:result={'error':repr(e)}
                (RUN/f'node-{node}.json').write_text(json.dumps(result,indent=2));receipt['models'][node]=result.get('rc',result.get('error'))
                print('model start',node,receipt['models'][node],flush=True)
    finally:
        (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
