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
RUN=b.ROOT/'runs/20261017T0700-long-progress-v1'
REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T0680-long-functional-v1'
IDS={43:'d6894b55dc4fc874aa6187bb4723b8dec43cdb21bb321b7702d114c6757d142e',45:'3ba6b5d6f580bc18cdb45190b1fcd07950cac2b176a0c4d930d01a492074c3a9'}
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
 with urllib.request.urlopen(f'http://172.16.12.{NODE}:30400/metrics',timeout=5) as response:metrics=response.read().decode()
 out['metrics']={'sha256':hashlib.sha256(metrics.encode()).hexdigest(),'text':metrics}
except Exception as e:out['metrics']={'error':repr(e)}
if NODE==43:
 root=pathlib.Path(REMOTE);out['client_files']={}
 for name in ('client.cid','functional-progress.json','functional-result.json'):
  p=root/name
  if p.is_file():out['client_files'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()}
 if 'client.cid' in out['client_files']:
  cid=out['client_files']['client.cid']['text'].strip();assert re.fullmatch('[0-9a-f]{64}',cid)
  out['client']=inspect(cid,'glm53-long-functional-v1-20261017-43','sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7')
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
  for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-vllm-correctness-v2-20261017-{node}'),('REMOTE',REMOTE)]:code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
  result=b.invoke([command],code,timeout=150);(RUN/f'node-{node}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes'][node]=result['rc'];print(node,result['rc'],flush=True)
 (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
