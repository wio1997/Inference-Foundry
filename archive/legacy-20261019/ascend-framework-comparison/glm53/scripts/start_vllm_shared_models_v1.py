"""Start models once after exact native shared-pool capacity + CLI qualification."""
import ast,concurrent.futures,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve()
def load(name,filename,sha):
 path=P.with_name(filename);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('pool','start_vllm_shared_pool_v3.py','f47203d5735ad6ce81c943a214fcf261d3179e30f566d5b037b1d7605bb17beb')
guard=load('guard','glm53_vllm_shared_container_precheck_v3.py','b63bec7e81c1c545fec81af0d088ef548f8b8eb80023c5244bcb9d8ebc00ace1')
RUN=base.b.ROOT/'runs/20261017T1761-vllm-shared-model-start-v1'
CAPACITY_SHA='f2b21dfad636261e1975c37e214baa7c892de8482d8cb1a41fdcee42563dd6ce'
PRECHECK=r'''
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8');files=[p for p in model.rglob('*') if p.is_file()]
assert len(files)==193 and sum(p.stat().st_size for p in files)==764240829290
assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()=='ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca'
assert hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest()=='19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
ports=([30300,29410,28100] if NODE in [30,31] else [30400,29420,28200])
if NODE==30:ports.append(29580)
for port in ports:
 sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
config=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text())
assert config['run_id']==RID and config['provider_env']=={} and 'PYTHONPATH' not in config['model_env'] and 'LD_LIBRARY_PATH' not in config['model_env']
assert config['model_argv']==['python3','/opt/glm53/bundle/entry.py',str(NODE)]
available=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert available>800*1024**3
assert not (root/'model.stdout.txt').exists() and not (root/'model.stderr.txt').exists()
print(json.dumps({'node':NODE,'container_id':CID,'supervisor':supervisor,'provider_phase':provider['phase'],'npu_before':npu_before,'available_host_bytes':available,'model_not_started':True}))
'''
START=r'''
import time
with (root/'model-start-controller.once').open('x') as file:file.write(CID)
signal_result=command(['docker','kill','--signal=USR1',CID]);assert signal_result.strip()==CID
deadline=time.monotonic()+10
while time.monotonic()<deadline:
 observed=json.loads((root/'supervisor-status.json').read_text())
 if observed.get('model_started') and observed.get('model_pid') and observed['phase']=='MODEL_RUNNING':break
 time.sleep(.1)
assert observed['model_started'] and observed['phase']=='MODEL_RUNNING' and 'model_exit_code' not in observed,observed
print(json.dumps({'node':NODE,'container_id':CID,'supervisor_after':observed,'model_stdout_path':str(root/'model.stdout.txt'),'model_stderr_path':str(root/'model.stderr.txt'),'classification':'MODEL_PROCESS_STARTED_FULL_NATIVE_STARTUP_PENDING','external_model_requests':0}))
'''
def main():
 path=base.b.ROOT/'results/vllm-shared-pool-capacity-v3.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==CAPACITY_SHA
 capacity=json.loads(path.read_text());assert capacity['pass'] and capacity['actual_MC_bytes']==568*1024**3 and not capacity['model_started'] and len(capacity['providers'])==4 and len(capacity['native_preflight'])==4
 for node in [30,31,43,45]:
  assert capacity['native_preflight'][str(node)]['torch_device_count']==16 and capacity['native_preflight'][str(node)]['ACL_count']==[16,0] and capacity['native_preflight'][str(node)]['native_all4_CLI_validated']
  ast.parse(guard.render(node,PRECHECK,capacity,base.FILES,base.STAGED));ast.parse(guard.render(node,START,capacity,base.FILES,base.STAGED))
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'capacity_evidence_sha256':CAPACITY_SHA,'precheck':{},'starts':{},'external_model_requests':0}
 try:
  for node in [30,31,43,45]:
   result=base.invoke(node,guard.render(node,PRECHECK,capacity,base.FILES,base.STAGED),150);(RUN/f'precheck-{node}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['precheck'][str(node)]=result['rc'];assert result['rc']==0,result
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(base.invoke,node,guard.render(node,START,capacity,base.FILES,base.STAGED),120):node for node in [30,31,43,45]}
   for f in concurrent.futures.as_completed(fs):
    node=fs[f]
    try:result=f.result()
    except Exception as error:result={'error':repr(error)}
    (RUN/f'start-{node}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['starts'][str(node)]=result.get('rc',result.get('error'))
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 assert len(receipt['starts'])==4 and all(value==0 for value in receipt['starts'].values()),receipt
if __name__=='__main__':main()
