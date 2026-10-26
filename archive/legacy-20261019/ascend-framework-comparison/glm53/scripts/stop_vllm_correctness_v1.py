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
RUN=b.ROOT/'runs/20261017T0530-stop-vllm-correctness-v1'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T0530-stop-vllm-correctness-v1'
IDS={30:'a522315f51028f930cb140058884763ca4c6ed0a5a9d369cb837039a9129209a',31:'c6c2542eb34abdc7accb639fa3e687910cb2239201e826e6d4e0d104d7ebd20e',43:'4233c3c06c5daef517ca6c2064bf2f6ed56817a8d2770c5c3ca46cc4266262eb',45:'210f1943bbf32d180ecd66d00848997824917916402309814cec46b9b53b2335'}
META='e729366a1e82e02c70cb6a76f5c8fdf55daf1c57a93c2654b105749030f0a885'
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
 assert data['Config']['Labels']['project_id']=='glm53' and data['Config']['Labels']['run_id']=='20261017T0440-vllm-correctness-v1'
 return data
'''


def invoke(node,extra,timeout=220):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code=COMMON+extra
 import re
 for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-vllm-correctness-v1-20261017-{node}'),('META',META),('REMOTE',REMOTE)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],code,timeout=timeout)


FENCE=r'''
data=owned(CID,NAME);assert data['State']['Running']
port=30300 if NODE in (30,31) else 30400
with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}/metrics',timeout=5) as response:
 metrics=response.read().decode()
gauges=[]
for line in metrics.splitlines():
 m=re.match(r'^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$',line)
 if m:gauges.append({'name':m.group(1),'value':float(m.group(2)),'line':line})
assert any(g['name']=='running' for g in gauges) and any(g['name']=='waiting' for g in gauges),'Missing idle gauges; no stop'
assert all(g['value']==0 for g in gauges),'Active requests; no stop'
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
(root/'metrics-before.txt').write_text(metrics)
print(json.dumps({'node':NODE,'cid':CID,'idle_gauges':gauges,'metrics_sha256':hashlib.sha256(metrics.encode()).hexdigest(),'state':data['State']}))
'''
STOP=r'''
data=owned(CID,NAME);assert data['State']['Running']
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
if NODE==43:
 asset=pathlib.Path('/data/wio/aisbench-offline-43-20261014');metadata={}
 for name in ['venv/pyvenv.cfg','DEPLOYMENT.txt','INSTALL_VERIFIED','benchmark/setup.py','benchmark/ais_bench/benchmark/__init__.py','benchmark/.git/HEAD']:
  path=asset/name
  if path.is_file():metadata[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'text':path.read_text()[:60000]}
 metadata['python_links']=[{'name':str(p),'resolved':str(p.resolve())} for p in (asset/'venv/bin').glob('python*')]
 out['aisbench_metadata']=metadata
print(json.dumps(out));sys.exit(stopped['rc'])
'''
STOP_META=r'''
name='glm53-vllm-correctness-v1-meta-20261017-30'
data=owned(META,name)
stopped=cmd(['docker','stop','--time','30',META],45)
after=owned(META,name);assert not after['State']['Running'] and after['State']['Pid']==0
logs=cmd(['docker','logs','--tail','100',META],20)
print(json.dumps({'meta_id':META,'stop':stopped,'state':after['State'],'logs':logs,'npu_after':cmd(['npu-smi','info'],30)}));sys.exit(stopped['rc'])
'''


def main():
 probe=json.loads((b.ROOT/'runs/20261017T0520-model-pd-probe-v1/receipt.json').read_text())
 assert probe['rc']==0 and probe['client_result']['model_requests_attempted']==4
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
