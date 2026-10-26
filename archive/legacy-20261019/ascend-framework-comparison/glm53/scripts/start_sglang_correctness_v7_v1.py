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
RUN = b.ROOT/'runs/20261017T1655-sglang-correctness-v7-v1'
REMOTE = '/data/wio/inference-foundry/glm53/runs/20261017T1655-sglang-correctness-v7-v1'
OVERLAY = '/data/wio/inference-foundry/glm53/overlays/sglang-sdma-v8-20261017'
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
POOL_PROVIDERS={35: {'container_id': '9ac4b37d5f2ce2f2edf8dbe16e506aeac8362e75b88bcf9579a97e07e5f6ef00', 'name': 'glm53-sglang-four-node-pool-v1-20261017-35', 'local_service_id': 0, 'configured_dram_size': '112GB'}, 34: {'container_id': '264d475e8173858cbcbee2769503ffd984124ff349c4e47d8f53afcb3bd0a59a', 'name': 'glm53-sglang-four-node-pool-v1-20261017-34', 'local_service_id': 1, 'configured_dram_size': '136GB'}, 36: {'container_id': '35ea19fc7edf105ccfa4b3c3c5de23ca3d86dd651965a4a3b77551b8e29b441e', 'name': 'glm53-sglang-four-node-pool-v1-20261017-36', 'local_service_id': 2, 'configured_dram_size': '160GB'}, 41: {'container_id': '3dbc3dee24e8aa88740f812ff9f1b48105b35309729ab40b8ed6c256f7e2eaa8', 'name': 'glm53-sglang-four-node-pool-v1-20261017-41', 'local_service_id': 3, 'configured_dram_size': '160GB'}}
STAGED={'manifest.json': '0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528', 'source/sglang/srt/disaggregation/ascend/transfer_engine.py': 'b8824bd9c8365bda4c584d18e3fb0d19a47f083d00c163c65ded3f95db5570b8', 'source/sglang/srt/disaggregation/ascend/conn.py': 'b49d78a04966c13e95758d30ee66574436e0034dfc24fdb12ecad2bbc5dfd547', 'python/glm53_sglang_sdma_audit.py': '22ae47b2c4d6bef5dd83fd1377766e35aef51a61259331db20690a984bd92948', 'config/node-34.json': 'e4cf3a230adf4576503de024396b7d9bce8021bd7af76558fe5d2e6bddc8445b', 'config/node-35.json': '30421dd6bcd907e53df7576b745240119a2a680e4e7483a42068052c115b11ea', 'config/node-36.json': 'a7952cb70e7026eae8f06188acfd333e6dedab168c990b2f798307b763e91238', 'config/node-41.json': '37507c8a5a9a37bbaa23bd3dcd13878b1480b010fb28659437f7054eb3d3eb4c', 'config/meta.conf': '67a02b220fc98a4b4fccfa9c86a497c5aa09f7fcd019c7675f2622bfd9850d86', 'config/local.conf': '405679fd396268145d85b1bdeee79ae93015237881fc1cf85b76ae2cbe54a8a6', 'config/manifest.json': '5e91a30b3ad6fd1bcbf708e654e7743ddc9e7ebb71c3efa789c729db629c2e1b', 'provider-config/node-34.json': '3f19f67aa8397607a1a6e0812bef2dcd324838a8c7a125f9856ae07b35e048ff', 'provider-config/node-35.json': '5552021e12049ade8b33a595247364e448263e2eec758b8b79cac5c53d42b9ae', 'provider.py': 'f18ffd988cf0df34b71f5aeec619cc53b853308fbbedc7fea0e786330328bf22', 'sdk-preflight.py': '0ee0cd46bc1db73655e85cd0834502bb5fd505bbb4119d16211acabfc276d2ef', 'entry.py': '3c5652b0c7143387f47c7add7c600e0ee085a74cd6a1632f0ef92201fdd5c3a8', 'provider-config/node-36.json': 'd7ac0fdfa40e3aa34bca197dc483504d2128115eae6fd9830f7c935e5e8c8b41', 'provider-config/node-41.json': '34be6e73525764bd27d1d9751b02d1b91920929bb0202939b155ec30029a0a6c'}

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
    ports=[29430,30500,29898,29590] if node in (34,35) else [29440,30600]
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
provider=json.loads(subprocess.run(['docker','inspect',PROVIDER_CID],capture_output=True,text=True,timeout=20).stdout)[0]
assert provider['Id']==PROVIDER_CID and provider['Name']=='/'+PROVIDER_NAME and provider['Image']==IMAGE and provider['Config']['Labels']['project_id']=='glm53' and provider['Config']['Labels']['run_id']=='20261017T1645-sglang-four-node-pool-start-v1' and provider['State']['Running']
provider_pid=int(provider['State']['Pid']);assert provider_pid>1
provider_status=json.loads(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1645-sglang-four-node-pool-start-v1/provider-status.json').read_text());assert provider_status['phase']=='READY' and provider_status['configuration']['node']==NODE
import re
entries=re.findall(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|',npu.stdout.split('Process id',1)[-1],re.M)
assert npu.returncode==0 and {int(x[2]) for x in entries}<={provider_pid} and len({int(x[0]) for x in entries})+npu.stdout.count('No running processes')==8,'Only exact owned standaloneprovider NPU PID allowed; abort on unrelated/unparsed processes'
if NODE==34:
 meta=json.loads(subprocess.run(['docker','inspect','df63a3efe328cf69d9b72125b185b7eebe4b93ec3feb3c8099ccb2fcbcdf6dca'],capture_output=True,text=True,timeout=20).stdout)[0]
 assert meta['Id']=='df63a3efe328cf69d9b72125b185b7eebe4b93ec3feb3c8099ccb2fcbcdf6dca' and meta['Name']=='/'+'glm53-sglang-four-node-pool-v1-meta-20261017-34' and meta['Image']==IMAGE and meta['Config']['Labels']['project_id']=='glm53' and meta['Config']['Labels']['run_id']=='20261017T1645-sglang-four-node-pool-start-v1' and meta['State']['Running']
with socket.create_connection(('172.16.12.34',29780),timeout=3):pass

for port in PORTS:
 s=socket.socket();s.bind(('0.0.0.0',port));s.close()
net=subprocess.run(['ip','-j','address','show'],capture_output=True,text=True,timeout=15);assert net.returncode==0
interfaces=[i['ifname'] for i in json.loads(net.stdout) if any(a.get('local')==IP for a in i.get('addr_info',[]))]
assert len(interfaces)==1,'Exact control interface must be unique'
memory=pathlib.Path('/proc/meminfo').read_text();available=int(next(l.split()[1] for l in memory.splitlines() if l.startswith('MemAvailable:')))*1024
assert available>800*1024**3,'Less than 800GiB available host RAM; preserve workload'
root.mkdir(parents=True);(root/'rpc').mkdir();(root/'cache').mkdir();(root/'entry.py').write_text(ENTRY)
(root/'precheck.json').write_text(json.dumps({'node':NODE,'image':IMAGE,'npu':npu.stdout,'meminfo':memory,'interface':interfaces[0],'model_metadata_sha256':METADATA},indent=2))
print(json.dumps({'node':NODE,'root':str(root),'interface':interfaces[0],'permitted_provider_host_pid':provider_pid,'other_NPU_processes':0,'available_host_bytes':available,'model_metadata_verified':True}))
'''
    import re
    for key,value in [('REMOTE',REMOTE),('IMAGE',IMAGE),('OVERLAY',OVERLAY),('STAGED',STAGED),('METADATA',METADATA),('PORTS',ports),('IP','172.16.12.'+str(node)),('ENTRY',ENTRY),('NODE',node),('PROVIDER_CID',POOL_PROVIDERS[node]['container_id']),('PROVIDER_NAME',POOL_PROVIDERS[node]['name'])]:
        code=re.sub(r'\b'+key+r'\b',lambda _:repr(value),code)
    return invoke_host(node,code,timeout=160)


def start_model(node,interface):
    name=f'glm53-sglang-correctness-v7-20261017-{node}'
    args=['docker','run','-d','--pull=never','--network=host','--shm-size=64g','--memory=1700g','--pids-limit=32768',
          '--ulimit','memlock=-1','--ulimit','nofile=655360:655360',
          '--label','project_id=glm53','--label','run_id=20261017T1655-sglang-correctness-v7-v1','--name',name,
          '--log-opt','max-size=128m','--log-opt','max-file=8',
          '-v',OVERLAY+':/opt/glm53/source:ro','-v',REMOTE+':/run/glm53',
          '-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro',
          '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
    for entry in SOURCE['files']:
        args+=['-v',OVERLAY+'/'+entry['local']+':'+entry['container_path']+':ro']
    for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
        args+=['--device',dev]
    config_path=b.ROOT/f'config/sglang-correctness-v9/node-{node}.json'
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
provider=json.loads(subprocess.run(['docker','inspect',PROVIDER_CID],capture_output=True,text=True,timeout=20).stdout)[0]
assert provider['Id']==PROVIDER_CID and provider['Name']=='/'+PROVIDER_NAME and provider['Image']==IMAGE and provider['Config']['Labels']['project_id']=='glm53' and provider['Config']['Labels']['run_id']=='20261017T1645-sglang-four-node-pool-start-v1' and provider['State']['Running']
provider_pid=int(provider['State']['Pid']);assert provider_pid>1
provider_status=json.loads(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1645-sglang-four-node-pool-start-v1/provider-status.json').read_text());assert provider_status['phase']=='READY' and provider_status['configuration']['node']==NODE
import re
entries=re.findall(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|',npu.stdout.split('Process id',1)[-1],re.M)
assert npu.returncode==0 and {int(x[2]) for x in entries}<={provider_pid} and len({int(x[0]) for x in entries})+npu.stdout.count('No running processes')==8,'Only exact owned standaloneprovider NPU PID allowed; abort on unrelated/unparsed processes'

with (root/'start-model.once').open('x') as f:f.write(NAME)
result=subprocess.run(ARGS,capture_output=True,text=True,timeout=45)
out={'node':NODE,'name':NAME,'run_rc':result.returncode,'container_id':result.stdout.strip(),'stderr':result.stderr,'args':ARGS}
(root/'start-result.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out));sys.exit(result.returncode)
'''
    import re
    for key,value in [('REMOTE',REMOTE),('NAME',name),('NODE',node),('ARGS',args),('IMAGE',IMAGE),('PROVIDER_CID',POOL_PROVIDERS[node]['container_id']),('PROVIDER_NAME',POOL_PROVIDERS[node]['name'])]:code=re.sub(r'\b'+key+r'\b',lambda _:repr(value),code)
    return invoke_host(node,code,timeout=90)


def main():
    pool=json.loads((b.ROOT/'results/sglang-four-node-pool-capacity-v1.json').read_text())
    assert hashlib.sha256((b.ROOT/'results/sglang-four-node-pool-capacity-v1.json').read_bytes()).hexdigest()=='7e91f93a151b88b6d5312108ab7935ec9cf744980535769584a39fe788a9636c'
    assert pool['actual_MC_bytes']==568*1024**3
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'phase':'STARTUP_ONLY_NOT_MODEL_CORRECTNESS','prechecks':{},'models':{},'existing_pool':pool}
    try:
        interfaces={}
        for node in (34,35,36,41):
            result=precheck(node);(RUN/f'precheck-{node}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['prechecks'][node]=result['rc'];assert result['rc']==0,'Precheckfailed; no model starts'
            interfaces[node]=json.loads(result['stdout'])['interface']
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures={executor.submit(start_model,node,interfaces[node]):node for node in (34,35,36,41)}
            for future in concurrent.futures.as_completed(futures):
                node=futures[future]
                try:result=future.result()
                except Exception as e:result={'error':repr(e)}
                (RUN/f'node-{node}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['models'][node]=result.get('rc',result.get('error'));print('modelstart',node,receipt['models'][node],flush=True)
    finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
