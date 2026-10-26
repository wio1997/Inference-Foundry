#!/usr/bin/env python3
"""Read/copy only own native SDK log directories, no SDK/model operations."""
import hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1596-vllm-native-memcache-logs-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1596-vllm-native-memcache-logs-v1'
TARGETS={30: [('2ee8e748d0328653b0e9a2bc50c394dddd103f80572753de7da2f491280da251', 'glm53-vllm-correctness-v4-meta-20261017-30')]}
LIST="import pathlib,json\nout={}\nfor name in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']:\n p=pathlib.Path(name);out[name]=[{'path':str(f),'bytes':f.stat().st_size} for f in p.rglob('*') if f.is_file() and not f.is_symlink()] if p.exists() else None\nprint(json.dumps(out))"
HOST=r'''import socket,subprocess,pathlib,json,hashlib,tarfile
assert socket.gethostname()==HOSTNAME
root=pathlib.Path(REMOTE);root.mkdir(parents=True,exist_ok=False);out=[]
for cid,name in TARGETS:
 r=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=15);assert r.returncode==0;d=json.loads(r.stdout)[0]
 assert d['Id']==cid and d['Name']=='/'+name and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1100-vllm-correctness-v4' and d['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
 r=subprocess.run(['docker','exec',cid,'python3','-c',LIST],capture_output=True,text=True,timeout=30);assert r.returncode==0;listing=json.loads(r.stdout);total=sum(f['bytes'] for fs in listing.values() if fs for f in fs);assert total<=256*1024**2,'NativeSDKlogs exceed256MiB cap; preserve and redesign capture'
 dest=root/cid;dest.mkdir();copies=[]
 for source,fs in listing.items():
  if fs is None:continue
  target=dest/pathlib.Path(source).name
  r=subprocess.run(['docker','cp',cid+':'+source,str(target)],capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr;copies.append(str(target))
 files=[];selected=[]
 for f in sorted(dest.rglob('*')):
  if not f.is_file() or f.is_symlink():continue
  raw=f.read_bytes();files.append({'path':str(f.relative_to(dest)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  text=raw.decode(errors='replace')
  for line in text.splitlines():
   low=line.lower()
   if any(x in low for x in ['metrics summary','error','batch_get','batchget','batchput','batch_put','allocated','capacity','lease_expired','copyg2h','copyh2g']):selected.append({'file':str(f.relative_to(dest)),'line':line[:1800]})
 archive=root/(cid+'.tar.gz')
 with tarfile.open(archive,'x:gz') as tar:tar.add(dest,arcname=cid,recursive=True)
 out.append({'container_id':cid,'name':name,'listing':listing,'copied_log_dirs':copies,'files':files,'snapshot_file_bytes':sum(f['bytes'] for f in files),'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'selected_total':len(selected),'selected_last':selected[-2000:]})
print('GLM53_NATIVE_SDK_LOG_CAPTURE='+json.dumps({'node':NODE,'containers':out,'model_requests':0}),flush=True)
'''
def capture(node):
 pod,host=b.NODES[node];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']==f'172.16.12.{node}'
 code=HOST
 for key,value in [('HOSTNAME',host),('REMOTE',REMOTE),('TARGETS',TARGETS[node]),('LIST',LIST),('NODE',node)]:code=re.sub(r'\b'+key+r'\b',lambda _:repr(value),code)
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=300)
def main():
 prior=json.loads((b.ROOT/'runs/20261017T1578-sglang-long-semantic-v3/receipt.json').read_text());assert prior['rc']==0
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'model_requests':0}
 try:
  for n in TARGETS:
   r=capture(n);(RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][n]=r['rc'];print(n,r['rc'],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
