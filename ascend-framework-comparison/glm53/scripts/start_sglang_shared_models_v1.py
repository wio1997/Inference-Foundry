"""Start native models once in the qualified shared containers; no external inference."""
import ast, concurrent.futures, hashlib, importlib.util, json
from pathlib import Path

P = Path(__file__).resolve()
dependency = P.parent/'run_sdk_host_readback_fixture_v1.py'
assert hashlib.sha256(dependency.read_bytes()).hexdigest() == '669ab4a74df2fb9942a41319dba4fb370304abf02b8b76243cb186172417c7ba'
spec = importlib.util.spec_from_file_location('qualified_shared', dependency)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
RUN = m.base.b.ROOT/'runs/20261017T1724-sglang-shared-model-start-v1'
FIXTURE_SHA = 'b45f3e2b2bb4a2ed450d6dff889e9b64e1af919e7b37cc08f6d2303bcd5d5474'
PRECHECK = r'''
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8')
files=[p for p in model.rglob('*') if p.is_file()]
assert len(files)==193 and sum(p.stat().st_size for p in files)==764240829290
assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()=='ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca'
for port in ([29430,30500,29898,29590] if NODE in [34,35] else [29440,30600]):
 sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
config=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text())
assert config['run_id']==RID and 'PYTHONPATH' not in config['model_env']
assert '${PYTHONPATH:-}' in config['model_argv'][2] and config['provider_env']=={}
available=int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
assert available>800*1024**3
assert not (root/'model.stdout.txt').exists() and not (root/'model.stderr.txt').exists()
print(json.dumps({'node':NODE,'container_id':CID,'supervisor':supervisor,'provider':provider,'npu_before':npu_before,'available_host_bytes':available,'model_not_started':True}))
'''
START = r'''
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

def code(node, action):
    return m.guard.render(node, action, m.capacity, m.base.FILES, m.base.STAGED)

def main():
    evidence = m.base.b.ROOT/'results/sdk-host-readback-qualification-v1.json'
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == FIXTURE_SHA
    data = json.loads(evidence.read_text()); assert data['pass'] and data['model_requests']==0
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
    receipt = {'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'precheck':{},'starts':{},'external_model_requests':0}
    try:
        for node in [34,35,36,41]:
            result=m.base.invoke(node,code(node,PRECHECK),timeout=150)
            (RUN/f'precheck-{node}.json').write_text(json.dumps(result,indent=2)+'\n')
            receipt['precheck'][str(node)]=result['rc'];assert result['rc']==0,result
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(m.base.invoke,node,code(node,START),120):node for node in [34,35,36,41]}
            for future in concurrent.futures.as_completed(futures):
                node=futures[future]
                try:result=future.result()
                except Exception as error:result={'error':repr(error)}
                (RUN/f'start-{node}.json').write_text(json.dumps(result,indent=2)+'\n')
                receipt['starts'][str(node)]=result.get('rc',result.get('error'))
    finally:
        (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert len(receipt['starts'])==4 and all(value==0 for value in receipt['starts'].values()),receipt

if __name__=='__main__':main()
