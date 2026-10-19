#!/usr/bin/env python3
"""Read-only shutdown diagnosis after1705; never retry a stop."""
import concurrent.futures,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();dep=P.parent/'stop_sglang_empty_pool_v1.py'
assert hashlib.sha256(dep.read_bytes()).hexdigest()=='cab27807a7ee441ec7c5ae2c3462d7b268fb8b4a2e85312eef3e57147f04b4ed'
s=importlib.util.spec_from_file_location('stop_helpers',dep);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RUN=m.m.b.ROOT/'runs/20261017T1707-sglang-pool-shutdown-audit-v1'
HOST=r'''import subprocess,json,pathlib,gzip,hashlib
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1645-sglang-four-node-pool-start-v1'
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1705-stop-empty-sglang-pool-v1')
out={'node':NODE,'state':d['State'],'provider_status':json.loads(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1645-sglang-four-node-pool-start-v1/provider-status.json').read_text()),'logs':{}}
for stream in ['stdout','stderr']:
 p=root/(stream+'.gz')
 if p.is_file():
  raw=gzip.decompress(p.read_bytes());out['logs'][stream]={'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'tail':raw[-25000:].decode(errors='replace')}
r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);out['npu']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
out['meminfo']=pathlib.Path('/proc/meminfo').read_text()
print(json.dumps(out))
'''
def capture(n):
    p=m.POOL['providers'][str(n)];return m.m.invoke(n,m.m.replace(HOST,{'CID':p['container_id'],'NAME':p['name'],'NODE':n,'IMAGE':m.m.IMAGE}),timeout=100)
def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'model_requests':0}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fs={ex.submit(capture,n):n for n in [34,35,36,41]}
            for f in concurrent.futures.as_completed(fs):
                n=fs[f]
                try:r=f.result()
                except Exception as e:r={'error':repr(e)}
                (RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][n]=r.get('rc',r.get('error'))
    finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
