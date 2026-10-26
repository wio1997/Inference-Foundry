"""Bounded read-only SDK ptracer filename/tail diagnosis; no SDK imports/calls."""
import ast,concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261017T1887-vllm-C64-SDK-ptracer-snapshot-recovery-v1'
def load(name,path,sha):
 p=ROOT/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('inventory','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
identity=load('identity','scripts/glm53_vllm_shared_model_identity_v3.py','24a93a46cfc19946e4e52cc2d255e92ea552c127b560d12ee1fea7108a18fb89')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
SDK_SOURCE="""import pathlib,json,hashlib,time,stat
root=pathlib.Path('/var/log/memfabric_hybrid');paths=sorted(p for p in root.iterdir() if 'ptracer' in p.name);assert 0<len(paths)<=1024
out={'root':str(root),'time_ns':time.time_ns(),'files':[],'SDK_init_calls':0,'model_requests':0}
for p in paths:
 st=p.lstat();assert stat.S_ISREG(st.st_mode) and not p.is_symlink()
 with p.open('rb') as f:
  header=f.read(512);offset=max(0,st.st_size-8192);f.seek(offset);tail=f.read(8193);assert len(tail)<=8192
 end=p.stat();out['files'].append({'name':p.name,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'inode':st.st_ino,'stable_stat_during_read':(st.st_size,st.st_mtime_ns,st.st_ino)==(end.st_size,end.st_mtime_ns,end.st_ino),'header':header.decode(errors='replace'),'tail_offset':offset,'tail_bytes':len(tail),'tail_sha256':hashlib.sha256(tail).hexdigest(),'tail_diagnostic_only':tail.decode(errors='replace')})
print(json.dumps(out))
"""
HOST="""import pathlib,json,hashlib,gzip,time
source=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1886-vllm-C64-SDK-ptracer-diagnosis-v1/SDK-ptracer-inventory.json')
assert source.is_file() and not source.is_symlink()
raw=source.open('rb').read(32*1024**2+1);assert len(raw)<=32*1024**2
out=json.loads(raw);assert out['project_id']=='glm53' and out['node']==NODE and out['container_id']==CID and out['model_requests']==out['SDK_init_calls']==0
archive=pathlib.Path(ARCHIVE);assert not archive.exists();archive.mkdir(parents=True)
out['diagnostic_snapshot_recovery']={'source':str(source),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'read_monotonic_ns':time.monotonic_ns(),'new_model_requests':0,'new_SDK_init_calls':0}
"""

def code_for(node,settings):
 item=settings['runtime_containers'][str(node)];values={'CID':item['container_id'],'NODE':node,'IDENTITY':identity.HOST,'IDENTITY_ENV':{'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'NODE':node},'SDK_SOURCE':SDK_SOURCE,'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+RID}
 code=re.sub(r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b',lambda m:repr(values[m[0]]),HOST);code+=codec.ENCODER;ast.parse(code);return code

def invoke(node,code):
 pod,hostname=base.NODES[node];r=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert r['rc']==0 and r['stdout']=='172.16.12.'+str(node)
 code='import socket\nassert socket.gethostname()=='+repr(hostname)+'\n'+code
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);response=base.invoke([command],code,timeout=180)
 if response['rc']!=0:return response|{'recovery_failed_preserved_remote_stderr':True}
 return codec.decode(response)
def main():
 p=ROOT/'config/formal-runs-v2/vllm-C64.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='40b233ae758612a57d2a8bec35d90e65ace0a210efe22ce071fd5af7c3829f6d';settings=json.loads(p.read_text())
 prior=json.loads((ROOT/'runs/20261017T1884-vllm-formal-C64-native-v4/receipt.json').read_text());assert len(prior['nodes'])==4 and all(v==0 for v in prior['nodes'].values())
 rpc=json.loads((ROOT/'runs/20261017T1880-vllm-formal-C64-v2/controller-rpc-state.json').read_text());assert rpc['done_received'] and rpc['connection_rc']==0
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(invoke,n,code_for(n,settings)):n for n in [43,45]}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f]
    try:r=f.result()
    except Exception as error:r={'error':repr(error)}
    (run/f'node-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][str(n)]=r.get('rc',r.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
