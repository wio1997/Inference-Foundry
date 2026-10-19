#!/usr/bin/env python3
"""Single read-only owned-container startup checkpoint, no inference."""
import concurrent.futures
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0450-vllm-startup-checkpoint-v1'
IDS={30:'a522315f51028f930cb140058884763ca4c6ed0a5a9d369cb837039a9129209a',31:'c6c2542eb34abdc7accb639fa3e687910cb2239201e826e6d4e0d104d7ebd20e',43:'4233c3c06c5daef517ca6c2064bf2f6ed56817a8d2770c5c3ca46cc4266262eb',45:'210f1943bbf32d180ecd66d00848997824917916402309814cec46b9b53b2335'}
META='e729366a1e82e02c70cb6a76f5c8fdf55daf1c57a93c2654b105749030f0a885'
HOST=r'''
import socket,subprocess,json,sys,pathlib,urllib.request,time,hashlib
assert socket.gethostname()==EXPECTED
def cmd(args,timeout=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def inspect(cid,name):
 raw=cmd(['docker','inspect',cid]);assert raw['rc']==0
 data=json.loads(raw['stdout'])[0]
 assert data['Id']==cid and data['Name']=='/'+name and data['Config']['Labels']['project_id']=='glm53'
 assert data['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
 return {'id':cid,'name':name,'image':data['Image'],'state':data['State'],
         'logs':cmd(['docker','logs','--tail','1600',cid]),
         'top':cmd(['docker','top',cid,'-eo','pid,ppid,stat,comm,etime'])}
out={'node':NODE,'monotonic_ns':time.monotonic_ns(),'model':inspect(CID,NAME),
     'npu':cmd(['npu-smi','info'],30),'load':pathlib.Path('/proc/loadavg').read_text(),
     'meminfo':pathlib.Path('/proc/meminfo').read_text(),
     'listeners':cmd(['ss','-lntp'])}
port=30300 if NODE in (30,31) else 30400
for endpoint in ['/health','/v1/models']:
 try:
  with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}'+endpoint,timeout=2) as response:
   out[endpoint]={'status':response.status,'body':response.read(65536).decode()}
 except Exception as e:out[endpoint]={'error':str(e)}
if NODE==30:out['meta']=inspect(META,'glm53-vllm-correctness-v1-meta-20261017-30')
print(json.dumps(out))
'''


def inspect(node):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code=HOST
 for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-vllm-correctness-v1-20261017-{node}'),('META',META)]:
  code=code.replace(key,repr(value))
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],code,timeout=150)


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(inspect,n):n for n in IDS}
   for f in concurrent.futures.as_completed(futures):
    n=futures[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2));receipt['nodes'][n]=r.get('rc',r.get('error'));print(n,receipt['nodes'][n],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
