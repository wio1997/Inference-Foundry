"""One bounded read-only capture of provider child stderr after failed pool1754."""
import concurrent.futures,hashlib,importlib.util,json,re
from pathlib import Path
P=Path(__file__).resolve();source=P.with_name('start_vllm_shared_pool_v2.py');assert hashlib.sha256(source.read_bytes()).hexdigest()=='c0aa59f1232bf7cbb380c0d6c8d9cba8153e724fbb8aa8311f0e9ac8a99a550c'
s=importlib.util.spec_from_file_location('pool',source);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
RUN=base.b.ROOT/'runs/20261017T1755-vllm-shared-pool-failure-v1'
HOST=r'''import pathlib,subprocess,json,hashlib
r=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID and not d['State']['Running'] and d['State']['ExitCode']==1
root=pathlib.Path(REMOTE);files={}
for name in ['provider.stdout.txt','provider.stderr.txt','provider-status.json','supervisor-status.json']:
 p=root/name
 if p.exists():
  raw=p.read_bytes();assert len(raw)<=1024**2;files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'text':raw.decode()}
assert not (root/'model.stdout.txt').exists() and not (root/'model.stderr.txt').exists()
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0 and r.stdout.count('No running processes')==8
print(json.dumps({'project_id':'glm53','node':NODE,'container_id':d['Id'],'state':d['State'],'files':files,'empty_NPU_cards':8,'model_requests':0}))
'''
def main():
 receipt=json.loads((base.RUN/'receipt.json').read_text());assert not receipt['pool_kept_running'] and receipt['cleanup']=={'30':0,'31':0,'43':0,'45':0,'meta':0}
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 out={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(base.invoke,n,base.replace(HOST,{'NAME':base.name(n),'IMAGE':base.IMAGE,'RID':base.RID,'REMOTE':base.REMOTE,'NODE':n}),100):n for n in [30,31,43,45]}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f];r=f.result();(RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2)+'\n');out['nodes'][str(n)]=r['rc']
 finally:(RUN/'receipt.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
