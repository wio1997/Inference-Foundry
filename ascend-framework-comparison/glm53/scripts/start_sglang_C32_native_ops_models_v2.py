"""C32-only L04 restore: reuse unchanged SDK payload qualification; signal four parents once, no microbenchmark repeat."""
import ast, concurrent.futures, hashlib, importlib.util, json
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1]
def load(name,file,sha):
 p=P.parent/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('metadata_pool','start_sglang_C32_native_ops_pool_v2.py','9132c5625e831b51eb7a3f190eaf5dc4d328534ee9887e7502416dddecaf3417')
guard=load('metadata_guard','glm53_C32_ops_container_precheck_v1.py','d5957465d5bdc4b48acec931acd9d649cbb1ab0c9a32c51f045395b7ca386a33')
RUN=ROOT/'runs/20261017T2068-sglang-C32-native-ops-model-start-v1'
PAYLOAD=ROOT/'scripts/glm53_sdk_host_readback_fixture_v1.py'
assert hashlib.sha256(PAYLOAD.read_bytes()).hexdigest()=='f5bf88adf229d4524bc4949bf993a621e0a37e0fc80dc261c21afd51610f5aae'
PRECHECK=r'''
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8');files=[p for p in model.rglob('*') if p.is_file()]
assert len(files)==193 and sum(p.stat().st_size for p in files)==764240829290
assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()=='ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca'
assert hashlib.sha256((model/'tokenizer.json').read_bytes()).hexdigest()=='19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
for port in ([29430,30500,29898,29590] if NODE in [34,35] else [29440,30600]):
 sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
config=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text())
assert config['run_id']==RID and 'PYTHONPATH' not in config['model_env'] and config['provider_env']=={}
assert '${PYTHONPATH:-}' in config['model_argv'][2]
assert config['model_env'].get('GLM53_DEEPEP_METADATA_BATCH')==('1' if NODE in [36,41] else None)
assert config['model_env'].get('GLM53_NODE')==(str(NODE) if NODE in [36,41] else None)
assert config['model_env'].get('GLM53_BUFFER_DEDUP_VALIDATE') is None
available=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert available>800*1024**3
assert not (root/'model.stdout.txt').exists() and not (root/'model.stderr.txt').exists()
print(json.dumps({'node':NODE,'container_id':CID,'supervisor':supervisor,'provider':provider,'npu_before':npu_before,'available_host_bytes':available,'model_not_started':True}))
'''
READBACK=r'''
r=subprocess.run(['docker','exec',CID,'python3','-c',PAYLOAD_SOURCE],capture_output=True,text=True,timeout=180)
assert r.returncode==0,(r.stdout,r.stderr)
data=json.loads(r.stdout.split('GLM53_SDK_HOST_READBACK=')[-1]);assert data['pass'] and data['close_rc']==0
after=check_npu();assert not json.loads((root/'supervisor-status.json').read_text())['model_started']
print(json.dumps({'container_id':CID,'payload_rc':r.returncode,'payload':data,'stderr':r.stderr,'npu_before':npu_before,'npu_after':after,'model_requests':0}))
'''
START=r'''
import time
with (root/'model-start-controller.once').open('x') as f:f.write(CID)
result=command(['docker','kill','--signal=USR1',CID]);assert result.strip()==CID
deadline=time.monotonic()+10
while time.monotonic()<deadline:
 observed=json.loads((root/'supervisor-status.json').read_text())
 if observed.get('model_started') and observed.get('model_pid') and observed['phase']=='MODEL_RUNNING':break
 time.sleep(.1)
assert observed['model_started'] and observed['phase']=='MODEL_RUNNING' and 'model_exit_code' not in observed
print(json.dumps({'node':NODE,'container_id':CID,'supervisor_after':observed,'classification':'MODEL_PARENT_STARTED_NATIVE_FULL_STARTUP_PENDING','external_model_requests':0}))
'''
def main():
 capacity=json.loads((ROOT/'results/sglang-C32-native-ops-pool-capacity-v1.json').read_text());assert capacity['pass'] and capacity['actual_MC_bytes']==568*1024**3 and not capacity['model_started']
 for name,sha in capacity['source_evidence_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'capacity_sha256':hashlib.sha256((ROOT/'results/sglang-C32-native-ops-pool-capacity-v1.json').read_bytes()).hexdigest(),'precheck':{},'starts':{},'external_model_requests':0}
 def save(name,x):(RUN/name).write_text(json.dumps(x,indent=2)+'\n');return x
 def code(n,action):
  result=guard.render(n,action,capacity,base.FILES,base.STAGED);ast.parse(result);return result
 try:
  for n in [34,35,36,41]:
   result=save(f'precheck-{n}.json',base.invoke(n,code(n,PRECHECK),150));receipt['precheck'][str(n)]=result['rc'];assert result['rc']==0
  previous_path=ROOT/'runs/20261017T2030-sglang-buffer-dedup-model-start-v1/SDK-readback-34.json'
  previous=json.loads(previous_path.read_text());assert previous['rc']==0
  verified=json.loads(previous['stdout'])['payload'];assert verified['pass'] and verified['setup_rc']==verified['init_rc']==verified['close_rc']==0 and verified['protocol']=='device_sdma' and verified['dram_size']=='0GB' and verified['init_bm'] is True
  assert len(verified['cases'])==4 and all(row['put_rc']==row['remove_rc']==0 and len(row['reads'])==2 and all(read['sha256']==row['expected_sha256'] for read in row['reads']) for row in verified['cases'])
  receipt['SDK_payload_probe_not_repeated']=True
  receipt['previous_identical_SDK_protocol_configuration_payload_evidence']={'path':str(previous_path.relative_to(ROOT)),'sha256':hashlib.sha256(previous_path.read_bytes()).hexdigest(),'scope':'NativeSDK/source/protocol/maxdramclientconfigurationunchanged; newactualprefixpayloadgatesremainmandatoryforC32measurements.'}
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures={pool.submit(base.invoke,n,code(n,START),120):n for n in [34,35,36,41]}
   for future in concurrent.futures.as_completed(futures):
    n=futures[future]
    try:result=future.result()
    except Exception as error:result={'error':repr(error)}
    save(f'start-{n}.json',result);receipt['starts'][str(n)]=result.get('rc',result.get('error'));print('start',n,receipt['starts'][str(n)],flush=True)
 finally:save('receipt.json',receipt)
 assert len(receipt['starts'])==4 and all(x==0 for x in receipt['starts'].values())
if __name__=='__main__':main()
