#!/usr/bin/env python3
"""One frozen EP32 synthetic correctness run on the two idle owned D nodes."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1290-deepep-alltoall-probe-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1290-deepep-alltoall-probe-v1'
IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
PAYLOAD=(P.parent/'probe_deepep_alltoall_payload_v1.py').read_text();assert hashlib.sha256(PAYLOAD.encode()).hexdigest()=='01483b03c972965abe5c2300d8b35fb099dd9c7e1c32963401267bd1342044ed'
idx=json.loads((b.ROOT/'runs/20261017T1260-deepep-native-contract-v1/code/index.json').read_text())
SHAS={str(Path(x['path']).relative_to('/usr/local/python3.12.13/lib/python3.12/site-packages/deep_ep')):x['sha256'] for x in idx if x['local'] in ['0-__init__.py','1-ep_strategy.py','3-buffer.py','4-device_info.py','9-normal_strategy.py']}
assert len(SHAS)==5

def invoke(node,code,timeout=120):
 pod,host=b.NODES[node];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']==f'172.16.12.{node}'
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)

def precheck(node):
 code='''import pathlib,subprocess,json
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
r=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20);assert r.returncode==0 and r.stdout.strip()==IMAGE
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0 and r.stdout.count('No running processes')==8
s=socket.socket();s.bind(('0.0.0.0',29460));s.close()
r=subprocess.run(['ip','-j','address','show'],capture_output=True,text=True,timeout=20);assert r.returncode==0
ifs=[i['ifname'] for i in json.loads(r.stdout) if any(a.get('local')==IP for a in i.get('addr_info',[]))];assert len(ifs)==1
root.mkdir(parents=True);(root/'payload.py').write_text(PAYLOAD)
print(json.dumps({'interface':ifs[0],'empty_cards':8}))
'''
 for k,v in [('REMOTE',REMOTE),('IMAGE',IMAGE),('IP',f'172.16.12.{node}'),('PAYLOAD',PAYLOAD)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return invoke(node,code)

def execute(node,interface):
 name=f'glm53-deepep-alltoall-probe-v1-20261017-{node}'
 args=['docker','run','-d','--pull=never','--network=host','--shm-size=8g','--memory=128g','--pids-limit=8192','--cpus=128','--label','project_id=glm53','--label','run_id=20261017T1290-deepep-alltoall-probe-v1','--name',name,'-v',REMOTE+':/run/glm53','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
 for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
 env={'HCCL_IF_IP':f'172.16.12.{node}','HCCL_SOCKET_IFNAME':interface,'GLOO_SOCKET_IFNAME':interface,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','DEEP_USE_MODE':'alltoall','DEEP_NORMAL_MODE_USE_INT8_QUANT':'0','GLM53_DEEPEP_SOURCE_SHA':json.dumps(SHAS),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','XDG_CACHE_HOME':'/run/glm53/cache'}
 for k,v in env.items():args+=['-e',k+'='+v]
 args+=['--entrypoint','python3',IMAGE,'-m','torch.distributed.run','--nnodes=2','--nproc-per-node=16','--node-rank='+str(0 if node==36 else 1),'--master-addr=172.16.12.36','--master-port=29460','--max-restarts=0','--monitor-interval=1','/run/glm53/payload.py']
 code='''import pathlib,subprocess,json,gzip,hashlib
root=pathlib.Path(REMOTE)
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0 and r.stdout.count('No running processes')==8
with (root/'execute-once').open('x') as f:f.write(NAME)
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
cid=r.stdout.strip();info=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0];assert info['Name']=='/'+NAME and info['Config']['Labels']['project_id']=='glm53' and info['Image']==IMAGE
out={'node':NODE,'cid':cid,'args':ARGS,'timeout':False}
try:
 r=subprocess.run(['docker','wait',cid],capture_output=True,text=True,timeout=360);out['wait']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
except subprocess.TimeoutExpired:
 out['timeout']=True;r=subprocess.run(['docker','stop','--time=30',cid],capture_output=True,text=True,timeout=45);out['stop']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
info=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0];out['state']=info['State']
r=subprocess.run(['docker','logs',cid],capture_output=True,text=True,timeout=30);out['logs']={}
for stream in ['stdout','stderr']:
 text=getattr(r,stream);raw=text.encode();path=root/(stream+'.gz')
 with path.open('xb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as gz:gz.write(raw)
 out['logs'][stream]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'path':str(path),'tail':text[-30000:]}
out['ranks']={p.name:json.loads(p.read_text()) for p in sorted(root.glob('rank-*.json'))}
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);out['npu_after']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
(root/'result.json').write_text(json.dumps(out,indent=2)+'\\n');print(json.dumps(out))
'''
 for k,v in [('REMOTE',REMOTE),('NAME',name),('NODE',node),('IMAGE',IMAGE),('ARGS',args)]:code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return invoke(node,code,timeout=490)

def main():
 cleanup=json.loads((b.ROOT/'runs/20261017T1270-cleanup-sglang-failed-v3/receipt.json').read_text());assert cleanup['meta_rc']==0 and all(v==0 for v in cleanup['nodes'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'synthetic_collective_only':True,'prechecks':{},'nodes':{}}
 try:
  interfaces={}
  for n in [36,41]:
   r=precheck(n);(RUN/f'precheck-{n}.json').write_text(json.dumps(r,indent=2));receipt['prechecks'][n]=r['rc'];assert r['rc']==0
   interfaces[n]=json.loads(r['stdout'])['interface']
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(execute,n,interfaces[n]):n for n in [36,41]}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][n]=r.get('rc',r.get('error'));print(n,receipt['nodes'][n],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
