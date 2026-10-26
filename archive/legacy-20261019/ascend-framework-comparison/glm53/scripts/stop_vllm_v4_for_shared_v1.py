#!/usr/bin/env python3
"""Fence idle metrics, retain raw logs, stop only owned v1 services D/P/Meta."""
import concurrent.futures
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1745-stop-vllm-v4-for-shared-v1'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1745-stop-vllm-v4-for-shared-v1'
IDS={30: 'c37c2e631f60ff4a4517d129f0a5341a56ab06c0d448463b466ca3c35bb06786', 31: '2f48868b06679ffd0750c5165fcd64840fbc9de43b6d4647e4b0bdbba47555dd', 43: '101a44f8ff8913bc5d103fb28ac3c09754005652ef06f805195187dd17db8a3c', 45: '5abe1c23a1a68e0e0e033048fe4d57864347213604b49ff89ea1c294fc861e12'}
META='2ee8e748d0328653b0e9a2bc50c394dddd103f80572753de7da2f491280da251'
STAGED='/data/wio/inference-foundry/glm53/overlays/vllm-sdma-v8-20261017'
STAGED_FILES={'source/manifest.json': {'local': 'config/overlays/vllm-sdma-v5/manifest.json', 'sha256': '1fc1d1e722b6dbb2d302980963c0285458e52acfa38abc95359828857a24d1ed'}, 'source/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py': {'local': 'config/overlays/vllm-sdma-v5/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py', 'sha256': '4f48b24b5ea0dff5ff208dc6fd39016375b8d25b2cd382663f918df4653ad505'}, 'source/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py': {'local': 'config/overlays/vllm-sdma-v5/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py', 'sha256': 'ed6f512a16e1c6883a974dcc4f3a55a305fc0371f31c1679eff4858bc4e10dd8'}, 'source/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py': {'local': 'config/overlays/vllm-sdma-v5/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py', 'sha256': '3b77f2ea980f6056b8d03861254718a29e4e18909b18fa465e77b8f18dc9116e'}, 'source/distributed/kv_transfer/ascend_multi_connector.py': {'local': 'config/overlays/vllm-sdma-v5/distributed/kv_transfer/ascend_multi_connector.py', 'sha256': 'cba51d6a0a9fb3fdc911fac5e22b24afd3c5f9ceb45e6c242261060a5d411f99'}, 'source/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py': {'local': 'config/overlays/vllm-sdma-v5/distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py', 'sha256': '1f29ff09cb00ffae0163f9af013c5bcef34dc30845b2b47ee245d9f8d0db450a'}, 'source/python/glm53_sdma_transport.py': {'local': 'config/overlays/vllm-sdma-v5/python/glm53_sdma_transport.py', 'sha256': 'ac0e3ad4f142c5954dbc1c1b09110b1d8018e0ab1701666b7b75e5072f0ce645'}, 'source/python/glm53_runtime_lifecycle.py': {'local': 'config/overlays/vllm-sdma-v5/python/glm53_runtime_lifecycle.py', 'sha256': 'de677307f474a613ab0833e00851b569e902f06f4443b73b8739bf3b510d1628'}, 'source/python/glm53_persistent_audit.py': {'local': 'config/overlays/vllm-sdma-v5/python/glm53_persistent_audit.py', 'sha256': '4c3edd9e8041b7c0f078674a3942892c1fce0b143d42bc6a7ae48910df5802cc'}, 'source/python/glm53_memcache_native.py': {'local': 'config/overlays/vllm-sdma-v5/python/glm53_memcache_native.py', 'sha256': 'e6200f49c12b51f42567446de6ec72e0c40780927e25eb068dc7fec65629233e'}, 'config/node-30.json': {'local': 'config/vllm-correctness-v9/node-30.json', 'sha256': '2164a1946162dbb5b7dee78f3646e0f1020c7aac2d74294f69ad07045a71448a'}, 'config/node-31.json': {'local': 'config/vllm-correctness-v9/node-31.json', 'sha256': 'e037393666416e7d80e1997fdaa199839d38b12c4a358831f238881fad8f7982'}, 'config/node-43.json': {'local': 'config/vllm-correctness-v9/node-43.json', 'sha256': '7f1f765429bd71f82166cc51e013429ca2ebd0810d7350c039a99e9ad55b5871'}, 'config/node-45.json': {'local': 'config/vllm-correctness-v9/node-45.json', 'sha256': '73eb8c142d6d018e8edada6c43da2216389cf5f08bde9aaaed6ec8f624e7ae61'}, 'config/meta.conf': {'local': 'config/vllm-correctness-v9/meta.conf', 'sha256': 'c81368740b1afe4b83b22a8edb26d258afc23383f623931adbec52d2959e9cac'}, 'config/local.conf': {'local': 'config/vllm-correctness-v9/local.conf', 'sha256': 'f90cf8929feb8d505db561a7171a27d0819ea40e31c0ea135ed2ba272cfa338e'}, 'config/manifest.json': {'local': 'config/vllm-correctness-v9/manifest.json', 'sha256': 'a7a5c06771416da79d25369a5e008e38353369c3a2b94902323b81a4a00b94bf'}, 'provider-config/node-30.json': {'local': 'config/standalone-pool-providers-v3/vllm-node-30.json', 'sha256': '81842762cea37468740e5e12892d43b48ace9d36810c303028ab9b26b576de21'}, 'provider-config/node-31.json': {'local': 'config/standalone-pool-providers-v3/vllm-node-31.json', 'sha256': '8dad0cd06485b547f203adc76e723d2d12512fd7946c8267da9cc0f7046a7f0d'}, 'provider.py': {'local': 'scripts/glm53_memcache_pool_provider_v3.py', 'sha256': 'e49811be51a8699bab78e9ad04750e01ffe6726b849f957c7012ce66b430856b'}, 'sdk-preflight.py': {'local': 'scripts/sdk_vllm_pool568_preflight_v2.py', 'sha256': '6f11bca626121ad3bc8e234a28afee43c3f30ad78cd7f1ca766e486d9757dc7f'}, 'entry.py': {'local': 'scripts/vllm_correctness_entry.py', 'sha256': '9595457ff22b2838a60650fd1006ab6ac21d0a8215cd21904bc2ab434491ada7'}, 'preflight.py': {'local': 'scripts/vllm_overlay_preflight_payload_v6.py', 'sha256': 'a9fcdcd159934b39c108643b1d44d74848c34d0b87b9e27fcfd821511ebbce2a'}, 'provider-config/node-43.json': {'local': 'config/standalone-pool-providers-v3/vllm-node-43.json', 'sha256': '5aab178acbda400c44e5911b95247ea37b6d0920c6aad3edb01eb8f8a95c8af9'}, 'provider-config/node-45.json': {'local': 'config/standalone-pool-providers-v3/vllm-node-45.json', 'sha256': '46873c7149e3a33586bee4892ac57ce20d5ba130c41b60ae44b439159f599d0a'}, 'supervisor.py': {'local': 'scripts/glm53_pool_model_supervisor_v1.py', 'sha256': 'd791e7b729d0c639c3109ee6b8b9ba2c5ccba7d537804cbb37f2970f1df6bde7'}, 'supervisor-config/node-30.json': {'local': 'config/vllm-shared-container-v1/node-30.json', 'sha256': '423d5c1098a2e36ff8e5e8f0807e50a3a638b4a3d9834880da21b7db84c27a69'}, 'supervisor-config/node-31.json': {'local': 'config/vllm-shared-container-v1/node-31.json', 'sha256': 'f6561fd975ba130999d9ab7960b9ca15779b41cae6375d9f20e6a9c4fef69726'}, 'supervisor-config/node-43.json': {'local': 'config/vllm-shared-container-v1/node-43.json', 'sha256': '380aad05ab0c5ec5e7b8e0722feb54c38abe35e156a87422a7d3fa6d8fefc21c'}, 'supervisor-config/node-45.json': {'local': 'config/vllm-shared-container-v1/node-45.json', 'sha256': '143ac8f4cd04f09f33c3998071c6c34e254e75bfe14b1571591eabdc6053cd14'}}
COMMON=r'''
import socket,subprocess,json,sys,pathlib,time,hashlib,gzip,urllib.request,re
assert socket.gethostname()==EXPECTED
def cmd(args,timeout=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def owned(cid,name):
 result=cmd(['docker','inspect',cid]);assert result['rc']==0
 data=json.loads(result['stdout'])[0]
 assert data['Id']==cid and data['Name']=='/'+name
 assert data['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
 assert data['Config']['Labels']['project_id']=='glm53' and data['Config']['Labels']['run_id']=='20261017T1100-vllm-correctness-v4'
 return data

def idle_metrics():
 port=30300 if NODE in (30,31) else 30400
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(f'http://172.16.12.{NODE}:{port}/metrics',timeout=5) as response:
  raw=response.read(8388609);assert len(raw)<=8388608,'Fullmetricscapexceeded; no truncatedfence'
 text=raw.decode();gauges=[];ranks=set()
 for line in text.splitlines():
  m=re.match(r'^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$',line)
  if m:
   rank=re.search(r'(?:^|,)engine="(\d+)"(?:,|$)',line.split('{',1)[1].split('}',1)[0]);assert rank is not None,'MissingnativeDPengineid'
   ranks.add(int(rank.group(1)));gauges.append({'name':m.group(1),'value':float(m.group(2)),'line':line})
 expected={30:{0,1},31:{2,3},43:set(range(8)),45:set(range(8,16))}[NODE]
 assert ranks==expected,(ranks,expected)
 assert any(g['name']=='running' for g in gauges) and any(g['name']=='waiting' for g in gauges) and all(g['value']==0 for g in gauges),'Activeormissingidlemetrics; abort'
 return text,gauges
'''


def invoke(node,extra,timeout=220):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code=COMMON+extra
 import re
 for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-vllm-correctness-v4-20261017-{node}'),('META',META),('REMOTE',REMOTE),('STAGED',STAGED),('STAGED_FILES',STAGED_FILES)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],code,timeout=timeout)


FENCE=r'''
data=owned(CID,NAME);assert data['State']['Running']
for relative,item in STAGED_FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/relative).read_bytes()).hexdigest()==item['sha256']
metrics,gauges=idle_metrics()
npu=cmd(['npu-smi','info'],30);assert npu['rc']==0
rows=[]
for line in npu['stdout'].split('Process id',1)[-1].splitlines():
 columns=[item.strip() for item in line.split('|')]
 if len(columns)<5:continue
 card_chip=columns[1].split()
 if len(card_chip)==2 and all(v.isdigit() for v in card_chip) and columns[2].isdigit():rows.append((int(card_chip[0]),int(card_chip[1]),int(columns[2])))
assert rows and len({r[0] for r in rows})+npu['stdout'].count('No running processes')==8
for pid in {row[2] for row in rows}:assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text(),'ForeignNPUprocess; abortbeforestops'
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
(root/'metrics-before.txt').write_text(metrics)
print(json.dumps({'node':NODE,'cid':CID,'idle_gauges':gauges,'metrics_sha256':hashlib.sha256(metrics.encode()).hexdigest(),'state':data['State'],'owned_NPU_rows':rows,'new_staged_source_verified':True}))
'''
STOP=r'''
data=owned(CID,NAME);assert data['State']['Running']
metrics_now,gauges_now=idle_metrics()
root=pathlib.Path(REMOTE)
with (root/'stop-model.once').open('x') as f:f.write(CID)
stopped=cmd(['docker','stop','--time','120',CID],145)
after=owned(CID,NAME);assert not after['State']['Running'] and after['State']['Pid']==0
logs=cmd(['docker','logs',CID],60)
out={'node':NODE,'cid':CID,'stop':stopped,'state':after['State'],'logs':{}}
for stream in ('stdout','stderr'):
 text=logs[stream];path=root/(CID+'.'+stream+'.gz')
 with path.open('xb') as file:
  with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as gz:gz.write(text.encode())
 lines=text.splitlines()
 selected=[l for l in lines if any(x in l for x in ('hybm_data_op_sdma.cpp','CopyG2GAsync','GLM53_','ERROR','Traceback','shutdown_complete'))]
 out['logs'][stream]={'archive':str(path),'bytes':len(text.encode()),'sha256':hashlib.sha256(text.encode()).hexdigest(),
                     'sdma_source_lines':sum('hybm_data_op_sdma.cpp' in l for l in lines),'selected_lines':len(selected),'last_selected':selected[-100:]}
out['npu_after']=cmd(['npu-smi','info'],30)
out['idle_gauges_immediately_before_stop']=gauges_now
print(json.dumps(out));assert out['npu_after']['rc']==0 and out['npu_after']['stdout'].count('No running processes')==8,'NPUresourcesnotempty; Meta remainsrunning'
sys.exit(stopped['rc'])
'''
STOP_META=r'''
name='glm53-vllm-correctness-v4-meta-20261017-30'
data=owned(META,name)
stopped=cmd(['docker','stop','--time','30',META],45)
after=owned(META,name);assert not after['State']['Running'] and after['State']['Pid']==0
logs=cmd(['docker','logs','--tail','100',META],20)
print(json.dumps({'meta_id':META,'stop':stopped,'state':after['State'],'logs':logs,'npu_after':cmd(['npu-smi','info'],30)}));sys.exit(stopped['rc'])
'''


def main():
 idle=b.ROOT/'results/vllm-v4-full-idle-v1.json';assert hashlib.sha256(idle.read_bytes()).hexdigest()=='362fb1e050c1158cd66d2ddd93abf21b235fdde287a500901968af4bed013b1f';assert json.loads(idle.read_text())['pass']
 staged=json.loads((b.ROOT/'runs/20261017T1746-vllm-shared-stage-v2/receipt.json').read_text());assert all(staged['staged'].get(str(n))==0 for n in (30,31,43,45)) and staged['SDK_preflight']['vllm']==0
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'fence':{},'stops':{},'meta_stop':None,'new_model_requests':0}
 try:
  for node in IDS:
   result=invoke(node,FENCE,90);(RUN/f'fence-{node}.json').write_text(json.dumps(result,indent=2))
   receipt['fence'][node]=result['rc'];assert result['rc']==0,'Not all nodes idle; no stops'
  for nodes in ((43,45),(30,31)):
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    fs={pool.submit(invoke,node,STOP,250):node for node in nodes}
    for future in concurrent.futures.as_completed(fs):
     node=fs[future]
     try:result=future.result()
     except Exception as e:result={'error':repr(e)}
     (RUN/f'node-{node}.json').write_text(json.dumps(result,indent=2));receipt['stops'][node]=result.get('rc',result.get('error'))
     print('stop',node,receipt['stops'][node],flush=True)
   assert all(receipt['stops'].get(n)==0 for n in nodes),'Stop failed; keep upstream services alive'
  result=invoke(30,STOP_META,100);(RUN/'meta-stop.json').write_text(json.dumps(result,indent=2));receipt['meta_stop']=result['rc']
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
