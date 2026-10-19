#!/usr/bin/env python3
"""Retain startup failure; gracefully stop only any remaining owned SG processes."""
import concurrent.futures,gzip,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1070-cleanup-sglang-failed-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1070-cleanup-sglang-failed-v1';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
IDS={34:'37c64e8270d9adfd90de047538b5dff35192103371be068f33bea9bb42bbd452',35:'4fed705164406d90f0de89576705f189eb24d76fc394a137299d564b6e461119',36:'5a1414242506fde42d2b85cb68d8422df66d373a0eb9824376c121adf96a1d97',41:'30ec85233f579fd4b741f3f47d1fd184740eb60a59fdbce887f5d31473531322'};META='2165d4226ab3f66dcee685440f810bdb9754ff932eacdbc19a42d2fed4c0d4bc'
COMMON=r'''import socket,subprocess,pathlib,json,gzip,hashlib,sys
assert socket.gethostname()==HOST
def cmd(args,timeout=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def owned(cid,name):
 r=cmd(['docker','inspect',cid]);assert r['rc']==0;q=json.loads(r['stdout'])[0]
 assert q['Id']==cid and q['Name']=='/'+name and q['Image']==IMAGE and q['Config']['Labels']['project_id']=='glm53' and q['Config']['Labels']['run_id']=='20261017T0980-sglang-correctness-v1'
 return q
'''
STOP=r'''
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
before=owned(CID,NAME);stopped=None
if before['State']['Running']:
 stopped=cmd(['docker','stop','--time','120',CID],145);assert stopped['rc']==0,stopped
q=owned(CID,NAME);assert not q['State']['Running'] and q['State']['Pid']==0
logs=cmd(['docker','logs',CID],60);assert logs['rc']==0
out={'node':NODE,'cid':CID,'before_state':before['State'],'stop':stopped,'after_state':q['State'],'logs':{}}
for stream in ['stdout','stderr']:
 data=logs[stream].encode();path=root/(CID+'.'+stream+'.gz')
 with path.open('xb') as file:
  with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as gz:gz.write(data)
 out['logs'][stream]={'archive':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'tail':logs[stream][-12000:]}
out['npu_after']=cmd(['npu-smi','info'],30);assert out['npu_after']['rc']==0 and out['npu_after']['stdout'].count('No running processes')==8
print(json.dumps(out))
'''
STOPMETA=r'''
for cid,name in [(CID,NAME)]:assert not owned(cid,name)['State']['Running']
name='glm53-sglang-correctness-v1-meta-20261017-34';before=owned(META,name);stopped=None
if before['State']['Running']:stopped=cmd(['docker','stop','--time','30',META],45);assert stopped['rc']==0
q=owned(META,name);assert not q['State']['Running'];print(json.dumps({'cid':META,'before_state':before['State'],'stop':stopped,'after_state':q['State'],'logs':cmd(['docker','logs','--tail','100',META])}))
'''
def invoke(node,body,timeout):
 pod,host=b.NODES[node];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.'+str(node)
 code=COMMON+body
 for k,v in [('HOST',host),('CID',IDS[node]),('NAME',f'glm53-sglang-correctness-v1-20261017-{node}'),('NODE',node),('IMAGE',IMAGE),('REMOTE',REMOTE),('META',META)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def main():
 d=json.loads((b.ROOT/'runs/20261017T1040-sglang-startup-failure-v1/receipt.json').read_text());assert all(x==0 for x in d['nodes'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'meta_rc':None,'external_model_requests':0}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   fs={pool.submit(invoke,n,STOP,240):n for n in IDS}
   for future in concurrent.futures.as_completed(fs):
    node=fs[future]
    try:r=future.result()
    except Exception as error:r={'error':repr(error)}
    (RUN/f'node-{node}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][node]=r.get('rc',r.get('error'))
  assert all(x==0 for x in receipt['nodes'].values()),'KeepMetaaliveifanymodelnotstopped'
  r=invoke(34,STOPMETA,90);(RUN/'meta-stop.json').write_text(json.dumps(r,indent=2)+'\n');receipt['meta_rc']=r['rc']
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
