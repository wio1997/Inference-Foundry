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
RUN=b.ROOT/'runs/20261017T1060-stop-vllm-correctness-v3'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1060-stop-vllm-correctness-v3'
IDS={30: 'b3cf90053554bfeec3e65cfd0a11c69afaf456eb077f3482d6e7b590e63f5219', 31: 'be7b34b50bf325741fdbfa33fdc1070bda627df1082d8c1e67daa63ae197dde8', 43: 'd6cc48ee6b8c993b5a583456c4389213bc57d595054186976c3c79994bfcc294', 45: '8f844997dbb88cbbc8d8dc1f28d170f3bee7199204dd87ad3f944ee98a66181b'}
META='fd358c508ecc349e54d6d41f6918b19e902a4a72831badd9994e2081ed974f00'
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
 assert data['Config']['Labels']['project_id']=='glm53' and data['Config']['Labels']['run_id']=='20261017T0840-vllm-correctness-v3'
 return data
'''


def invoke(node,extra,timeout=220):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code=COMMON+extra
 import re
 for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-vllm-correctness-v3-20261017-{node}'),('META',META),('REMOTE',REMOTE)]:
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
print(json.dumps(out));sys.exit(stopped['rc'])
'''
STOP_META=r'''
name='glm53-vllm-correctness-v3-meta-20261017-30'
data=owned(META,name)
stopped=cmd(['docker','stop','--time','30',META],45)
after=owned(META,name);assert not after['State']['Running'] and after['State']['Pid']==0
logs=cmd(['docker','logs','--tail','100',META],20)
print(json.dumps({'meta_id':META,'stop':stopped,'state':after['State'],'logs':logs,'npu_after':cmd(['npu-smi','info'],30)}));sys.exit(stopped['rc'])
'''


def main():
 staged=json.loads((b.ROOT/'runs/20261017T1030-stage-vllm-graph-v4/receipt.json').read_text())
 assert all(staged['nodes'].get(str(n))==0 for n in (30,31,43,45))
 prior=json.loads((b.ROOT/'runs/20261017T0950-long-semantic-v1/receipt.json').read_text());assert prior['rc']==0
 audit=json.loads((b.ROOT/'runs/20261017T1010-semantic-state-v1/receipt.json').read_text());assert all(x==0 for x in audit['nodes'].values())
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
