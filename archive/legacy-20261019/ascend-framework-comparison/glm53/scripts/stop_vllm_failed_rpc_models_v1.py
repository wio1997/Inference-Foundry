"""Stop only the four owned failed-frontend shared instances, then Meta last."""
import ast,concurrent.futures,hashlib,importlib.util,json,re
from pathlib import Path
P=Path(__file__).resolve();source=P.with_name('start_vllm_shared_pool_v3.py');assert hashlib.sha256(source.read_bytes()).hexdigest()=='f47203d5735ad6ce81c943a214fcf261d3179e30f566d5b037b1d7605bb17beb'
s=importlib.util.spec_from_file_location('pool',source);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
RUN=base.b.ROOT/'runs/20261017T1764-stop-vllm-rpc-failure-v1';STOP_REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1764-stop-vllm-rpc-failure-v1'
FENCE=r'''import pathlib,subprocess,json,hashlib
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID and d['State']['Running']
root=pathlib.Path(SHARED_REMOTE);supervisor=json.loads((root/'supervisor-status.json').read_text());provider=json.loads((root/'provider-status.json').read_text())
assert supervisor['phase']=='MODEL_EXITED_POOL_RETAINED' and supervisor['model_exit_code']==1 and supervisor['model_pid']==128 and provider['phase']=='READY' and provider['pid']==7
raw=(root/'model.stderr.txt').read_bytes();assert hashlib.sha256(raw).hexdigest()==MODEL_STDERR_SHA and b'zmq.error.ZMQError: No such file or directory for ipc path' in raw and b'/run/glm53/rpc/' in raw
r=subprocess.run(['docker','top',CID,'-eo','pid,args'],capture_output=True,text=True,timeout=20);assert r.returncode==0
provider_host=[]
for line in r.stdout.splitlines()[1:]:
 fields=line.split(None,1)
 if len(fields)!=2 or not fields[0].isdigit():continue
 proc=pathlib.Path('/proc')/fields[0]
 try:nspid=int(next(l.split()[-1] for l in (proc/'status').read_text().splitlines() if l.startswith('NSpid:')))
 except FileNotFoundError:continue
 assert nspid!=128,'An original model process is still alive; no stop allowed'
 if nspid==7:assert CID in (proc/'cgroup').read_text();provider_host.append(int(fields[0]))
assert len(provider_host)==1
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert r.returncode==0
rows=[]
for line in r.stdout.split('Process id',1)[-1].splitlines():
 cols=[v.strip() for v in line.split('|')]
 if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():rows.append((int(cols[1].split()[0]),int(cols[2])))
assert rows and {row[1] for row in rows}==set(provider_host) and len({row[0] for row in rows})+r.stdout.count('No running processes')==8
print(json.dumps({'node':NODE,'container_id':CID,'failed_model_exit':1,'only_owned_provider_NPU_pid':provider_host[0],'model_stderr_sha256':MODEL_STDERR_SHA,'external_model_requests':0}))
'''
# Reuse the reviewed exact-byte log preservation/normal supervisor shutdown body,
# adding an exact CID guard and a new output namespace. Never call the old launcher.
tree=ast.parse(source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='snapshot_stop')
STOP=ast.literal_eval(next(n.value for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets)))
STOP=STOP.replace('root=pathlib.Path(REMOTE)',"root=pathlib.Path(STOP_REMOTE);root.mkdir(parents=True,exist_ok=True)\nwith (root/('meta-stop.once' if META else 'provider-stop.once')).open('x') as f:f.write(CID)")
STOP=STOP.replace("['docker','inspect',NAME]","['docker','inspect',CID]").replace("info=json.loads(r['stdout'])[0];assert", "info=json.loads(r['stdout'])[0];assert info['Id']==CID;assert")
STOP=STOP.replace('p=root/child_file','p=pathlib.Path(SHARED_REMOTE)/child_file')
def render(code,values):
 pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b';text=re.sub(pattern,lambda m:repr(values[m[0]]),code);ast.parse(text);return text

def main():
 evidence=base.b.ROOT/'results/vllm-shared-startup-failure-v1.json';assert hashlib.sha256(evidence.read_bytes()).hexdigest()=='3e6665321a22d0fb13b13d59d3b8217a34c221582460c080d641104ba67c2d9f';failure=json.loads(evidence.read_text())
 cap=json.loads((base.b.ROOT/'results/vllm-shared-pool-capacity-v3.json').read_text());meta=json.loads(cap['meta']['stdout'])
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'fence':{},'stops':{},'meta_stop':None,'model_requests':0}
 def values(n,meta_flag=False):
  item=meta if meta_flag else cap['providers'][str(n)]
  return {'CID':item['container_id'],'NAME':item['name'],'IMAGE':base.IMAGE,'RID':base.RID,'NODE':n,'SHARED_REMOTE':base.REMOTE,'STOP_REMOTE':STOP_REMOTE,'META':meta_flag,'MODEL_STDERR_SHA':failure['nodes'][str(n)]['model_stderr_sha256']}
 def save(name,r):(RUN/name).write_text(json.dumps(r,indent=2)+'\n')
 try:
  for n in [30,31,43,45]:
   r=base.invoke(n,render(FENCE,values(n)),120);save(f'fence-{n}.json',r);receipt['fence'][str(n)]=r['rc'];assert r['rc']==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(base.invoke,n,render(STOP,values(n)),260):n for n in [30,31,43,45]}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f];r=f.result();save(f'node-{n}.json',r);receipt['stops'][str(n)]=r['rc'];assert r['rc']==0
  for n in [30,31,43,45]:
   data=json.loads(json.loads((RUN/f'node-{n}.json').read_text())['stdout']);state=json.loads(data['state_after']['stdout']);assert not state['Running'] and data['npu_after']['stdout'].count('No running processes')==8
  r=base.invoke(30,render(STOP,values(30,True)),260);save('meta-stop.json',r);receipt['meta_stop']=r['rc']
 finally:save('receipt.json',receipt)
if __name__=='__main__':main()
