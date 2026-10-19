#!/usr/bin/env python3
"""One bounded progress snapshot for current owned long CPU client and D models."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0905-vllm-v3-ready-checkpoint-v1'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T0710-long-diagnostic-v1'
IDS={30: 'b3cf90053554bfeec3e65cfd0a11c69afaf456eb077f3482d6e7b590e63f5219', 31: 'be7b34b50bf325741fdbfa33fdc1070bda627df1082d8c1e67daa63ae197dde8', 43: 'd6cc48ee6b8c993b5a583456c4389213bc57d595054186976c3c79994bfcc294', 45: '8f844997dbb88cbbc8d8dc1f28d170f3bee7199204dd87ad3f944ee98a66181b'}
HOST=r'''
import socket,pathlib,subprocess,json,hashlib,urllib.request,time,re
assert socket.gethostname()==EXPECTED
def cmd(args,timeout=25):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def inspect(cid,name,image):
 r=cmd(['docker','inspect',cid])
 if r['rc']!=0:return {'inspect':r}
 q=json.loads(r['stdout'])[0]
 assert q['Id']==cid and q['Name']=='/'+name and q['Image']==image and q['Config']['Labels']['project_id']=='glm53'
 return {'id':cid,'name':name,'state':q['State'],'logs':cmd(['docker','logs','--tail','100',cid]),'top':cmd(['docker','top',cid,'-eo','pid,ppid,stat,comm,etime'])}
out={'node':NODE,'monotonic_ns':time.monotonic_ns(),'D_model':inspect(CID,NAME,'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'),'npu':cmd(['npu-smi','info'],30)}
try:
 with urllib.request.urlopen(f'http://172.16.12.{NODE}:{30300 if NODE in [30,31] else 30400}/metrics',timeout=5) as response:metrics=response.read().decode()
 out['metrics']={'sha256':hashlib.sha256(metrics.encode()).hexdigest(),'text':metrics}
except Exception as e:out['metrics']={'error':repr(e)}
out['health']={}
for node in [30,31,43,45]:
 port=30300 if node in [30,31] else 30400
 try:
  with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/health',timeout=3) as r:out['health'][node]=r.status
 except Exception as error:out['health'][node]=repr(error)
print(json.dumps(out))
'''


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'nodes':{}}
 import re
 for node in IDS:
  pod,hostname=b.NODES[node]
  m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.'+str(node)
  code=HOST
  for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-vllm-correctness-v3-20261017-{node}'),('REMOTE',REMOTE)]:code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
  result=b.invoke([command],code,timeout=150);(RUN/f'node-{node}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes'][node]=result['rc'];print(node,result['rc'],flush=True)
 (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
