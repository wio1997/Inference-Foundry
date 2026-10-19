#!/usr/bin/env python3
"""Retain startup failure; gracefully stop only any remaining owned SG processes."""
import concurrent.futures,gzip,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1605-stop-sglang-v6-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1605-stop-sglang-v6-v1';IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
IDS={34: '9844ec4193f7bc662ac43a62e2606a72f8299992283f27d2005a8ce9c643f805', 35: 'bf1859f9bad4d03c10e78a647703dcff951d4dc81dd53dd6e941ba0d97e928db', 36: '2af3259c99c29dd00a7e8ee84906c96c4657bd5050a9190ebcc093d49991dbad', 41: '9d3495a179b91716c8685ecb18a8e78001abbad6e7a90b4a64f329f33c319196'}
META="97d6a65bdc24cdac9447f16d52ba41e737ba49f9c8888b0bd6bed436fb3c39fc"
COMMON=r'''import socket,subprocess,pathlib,json,gzip,hashlib,sys
assert socket.gethostname()==HOST
def cmd(args,timeout=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def owned(cid,name):
 r=cmd(['docker','inspect',cid]);assert r['rc']==0;q=json.loads(r['stdout'])[0]
 assert q['Id']==cid and q['Name']=='/'+name and q['Image']==IMAGE and q['Config']['Labels']['project_id']=='glm53' and q['Config']['Labels']['run_id']=='20261017T1562-sglang-correctness-v6-v3'
 return q
'''
STOP=r'''
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False)
before=owned(CID,NAME);stopped=None
if before['State']['Running']:
 import urllib.request
 port=30500 if NODE in [34,35] else 30600
 with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}/metrics',timeout=10) as response:metrics=response.read().decode()
 gauges=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(x==0 for x in gauges),'Active requests; preserve live service'
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
PREFLIGHT=r'''
before=owned(CID,NAME);assert before['State']['Running']
import urllib.request
port=30500 if NODE in [34,35] else 30600
with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}/metrics',timeout=10) as response:raw=response.read(8388609)
assert len(raw)<=8388608
metrics=raw.decode();gauges=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{','sglang:hicache_host_used_tokens{'))];assert gauges and not any(gauges),'Ownmodelorbufferworkstillactive: no stop'
print(json.dumps({'node':NODE,'current_idle_gauges':gauges,'ready_to_stop_own_model':True}))
'''
STOPMETA=r'''
for cid,name in [(CID,NAME)]:assert not owned(cid,name)['State']['Running']
name='glm53-sglang-correctness-v6-meta-20261017-34';before=owned(META,name);stopped=None
if before['State']['Running']:stopped=cmd(['docker','stop','--time','30',META],45);assert stopped['rc']==0
q=owned(META,name);assert not q['State']['Running'];print(json.dumps({'cid':META,'before_state':before['State'],'stop':stopped,'after_state':q['State'],'logs':cmd(['docker','logs','--tail','100',META])}))
'''
def invoke(node,body,timeout):
 pod,host=b.NODES[node];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.'+str(node)
 code=COMMON+body
 for k,v in [('HOST',host),('CID',IDS[node]),('NAME',f'glm53-sglang-correctness-v6-20261017-{node}'),('NODE',node),('IMAGE',IMAGE),('REMOTE',REMOTE),('META',META)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
def main():
 staged=json.loads((b.ROOT/'runs/20261017T1600-standalone-budget-stage-sdk-preflight-v1/receipt.json').read_text());assert all(x==0 for x in staged['staged'].values()) and staged['SDK_preflight']['sglang']==0
 finished=json.loads((b.ROOT/'runs/20261017T1578-sglang-long-semantic-v3/receipt.json').read_text());assert finished['rc']==0
 captured=json.loads((b.ROOT/'runs/20261017T1592-sglang-native-memcache-logs-v1/receipt.json').read_text());assert all(x==0 for x in captured['nodes'].values())
 qualified=json.loads((b.ROOT/'results/sglang-buffer-only-cache-qualification-v1.json').read_text());assert qualified['actual_MC_total_bytes']==512*1024**3 and qualified['native_MC_Python_batch_get_good_end']==80
 for n in IDS:
  c=json.loads((b.ROOT/'runs/20261017T1590-sglang-v6-long-cache-state-v1'/f'node-{n}.json').read_text());assert c['rc']==0;o=json.loads(c['stdout']);metrics=o['/metrics']['body'];gauges=[float(l.rsplit(' ',1)[1]) for l in metrics.splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{','sglang:hicache_host_used_tokens{'))];assert gauges and not any(gauges)
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'meta_rc':None,'external_model_requests':0,'prechecks':{}}
 try:
  for n in IDS:
   r=invoke(n,PREFLIGHT,90);(RUN/f'precheck-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['prechecks'][n]=r['rc'];assert r['rc']==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
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
