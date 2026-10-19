#!/usr/bin/env python3
"""Official A3 read-only memory scan + own actual config/SDK configuration support."""
import concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1370-memcache-official-memory-scan-v2';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1370-memcache-official-memory-scan-v2'
SCAN=(b.ROOT/'runs/20261017T1340-official-memcache-docs-review/mf13-memscan.txt').read_text();assert hashlib.sha256(SCAN.encode()).hexdigest()=='7eeffc19e7c65f34968a7eba29388accb92c51a04392c1d65ca8b1be28681354'
IDS={30:'c37c2e631f60ff4a4517d129f0a5341a56ab06c0d448463b466ca3c35bb06786',31:'2f48868b06679ffd0750c5165fcd64840fbc9de43b6d4647e4b0bdbba47555dd',34:'b91ac6ca6fffa3caa5ab08869997b0590f53846db48723991b9189cd7d248aeb',35:'b3cae6b8479be6db5a2bd001eade675330f01b238176292abdca172fec93c529'}
INNER=r'''import pathlib,json,os,hashlib
keys=['ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE','ASCEND_GLOBAL_RESOURCE_CONFIG','ASCEND_LOCAL_COMM_RES','PYTHONHASHSEED','MMC_LOCAL_CONFIG_PATH','MMC_META_CONFIG_PATH','SGLANG_HICACHE_HOST_MEM_BACKEND','DEEP_USE_MODE']
out={'selected_env':{k:os.environ.get(k) for k in keys},'configs':{},'native_meta_lease_option_libraries':[]}
for key in ['MMC_LOCAL_CONFIG_PATH','MMC_META_CONFIG_PATH']:
 path=os.environ.get(key)
 if path:
  p=pathlib.Path(path)
  if p.is_file():
   raw=p.read_bytes();out['configs'][key]={'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
roots=[pathlib.Path('/opt/glm53/sdk13/memcache_hybrid'),pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid')]
for root in roots:
 if not root.is_dir():continue
 for p in root.rglob('*.so*'):
  if not p.is_file() or p.stat().st_size>180000000:continue
  raw=p.read_bytes()
  fields=[field for field in ['ock.mmc.meta.lease_ttl_ms','meta.lease_ttl_ms','ock.mmc.dynamic_config.enable'] if field.encode() in raw]
  if fields:out['native_meta_lease_option_libraries'].append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'supported_option_strings':fields})
print(json.dumps(out))
'''
HOST=r'''import socket,pathlib,subprocess,json,hashlib,time
assert socket.gethostname()==HOSTNAME
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay';root.mkdir(parents=True)
path=root/'mem_scan.py';path.write_text(SCAN);assert hashlib.sha256(path.read_bytes()).hexdigest()=='7eeffc19e7c65f34968a7eba29388accb92c51a04392c1d65ca8b1be28681354'
def cmd(args,timeout=30):
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 except subprocess.TimeoutExpired as e:return {'error':'TIMEOUT','stdout':(e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else e.stdout,'stderr':(e.stderr or b'').decode(errors='replace') if isinstance(e.stderr,bytes) else e.stderr}
r=cmd(['docker','inspect',CID]);assert r['rc']==0;info=json.loads(r['stdout'])[0]
assert info['Id']==CID and info['Name']=='/'+NAME and info['Image']==IMAGE and info['Config']['Labels']['project_id']=='glm53'
out={'node':NODE,'monotonic_ns':time.monotonic_ns(),'model_state':info['State'],'meminfo':pathlib.Path('/proc/meminfo').read_text(),'page_size':cmd(['getconf','PAGESIZE']),'kernel':cmd(['uname','-a']),'selected_container_env':{k:v for e in info['Config']['Env'] for k,_,v in [e.partition('=')] if k in ['ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE','ASCEND_GLOBAL_RESOURCE_CONFIG','ASCEND_LOCAL_COMM_RES','PYTHONHASHSEED','DEEP_USE_MODE']},'model_requests':0,'npu_allocations':0,'system_changes':False}
out['driver_version']=pathlib.Path('/usr/local/Ascend/driver/version.info').read_text()
out['container_config_audit']={'omitted':'Alreadycaptured1360; thisfollowuponlycompletesnativephysicalsnapshot.'}
out['official_memory_scan']={}
for minimum in [1024,2]:
 r=cmd(['python3',str(path),'-m',str(minimum),'-w','8'],300);out['official_memory_scan'][str(minimum)]=r
 (root/f'mem_scan_{minimum}.json').write_text(json.dumps(r,indent=2)+'\n')
out['scan_limitation']='HostPythonmaylackACL; native scanner reportsUNKNOWN then usesdocumentedA3physicaladdressranges. Readonly/proc/kpageflags snapshot, notanallocationpromise. Noacl.init/hugepage/sysctlwrites.'
(root/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''
def capture(node):
 pod,host=b.NODES[node];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']==f'172.16.12.{node}'
 code=HOST;values={'HOSTNAME':host,'REMOTE':REMOTE,'SCAN':SCAN,'CID':IDS[node],'NAME':f'glm53-'+('vllm-correctness-v4' if node in [30,31] else 'sglang-correctness-v4')+f'-20261017-{node}','IMAGE':'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' if node in [30,31] else 'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263','NODE':node,'INNER':INNER}
 for k,v in values.items():code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=720)
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'official_scan_sha256':'7eeffc19e7c65f34968a7eba29388accb92c51a04392c1d65ca8b1be28681354','model_requests':0,'npu_allocations':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   fs={pool.submit(capture,n):n for n in IDS}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][n]=r.get('rc',r.get('error'));print(n,receipt['nodes'][n],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
