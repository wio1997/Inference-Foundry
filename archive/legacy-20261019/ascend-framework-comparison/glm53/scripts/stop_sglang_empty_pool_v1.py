#!/usr/bin/env python3
"""Replace only the empty owned four-container pool; providers before Meta."""
import concurrent.futures,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();dep=P.parent/'start_sglang_four_node_pool_v1.py'
assert hashlib.sha256(dep.read_bytes()).hexdigest()=='8fc7c6e9f387cf08c35bbc3c21899f9f5965ffe64db7767db4ed7d7fedd452c2'
s=importlib.util.spec_from_file_location('pool_helpers',dep);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RUN=m.b.ROOT/'runs/20261017T1705-stop-empty-sglang-pool-v1';REMOTE='/data/wio/inference-foundry/glm53/runs/20261017T1705-stop-empty-sglang-pool-v1'
cp=m.b.ROOT/'results/sglang-four-node-pool-capacity-v1.json';assert hashlib.sha256(cp.read_bytes()).hexdigest()=='7e91f93a151b88b6d5312108ab7935ec9cf744980535769584a39fe788a9636c';POOL=json.loads(cp.read_text())
FAILED=json.loads((m.b.ROOT/'results/sglang-v7-startup-failure-v1.json').read_text())['nodes']
HOST=r'''import subprocess,json,pathlib,gzip,hashlib
def cmd(a,t=30):
 r=subprocess.run(a,capture_output=True,text=True,timeout=t);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def inspect(cid,name,run):
 r=cmd(['docker','inspect',cid]);assert r['rc']==0;d=json.loads(r['stdout'])[0]
 assert d['Id']==cid and d['Name']=='/'+name and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==run
 return d
provider=inspect(CID,NAME,'20261017T1645-sglang-four-node-pool-start-v1');model=inspect(MODEL,'glm53-sglang-correctness-v7-20261017-'+str(NODE),'20261017T1655-sglang-correctness-v7-v1')
assert provider['State']['Running'] and not model['State']['Running'];pid=provider['State']['Pid']
npu=cmd(['npu-smi','info']);rows=[]
for line in npu['stdout'].split('Process id',1)[-1].splitlines():
 c=[x.strip() for x in line.split('|')]
 if len(c)>=5:
  pair=c[1].split()
  if len(pair)==2 and all(x.isdigit() for x in pair) and c[2].isdigit():rows.append((int(pair[0]),int(c[2])))
assert npu['rc']==0 and {r[1] for r in rows}<={pid} and len({r[0] for r in rows})+npu['stdout'].count('No running processes')==8
root=pathlib.Path(REMOTE)
if ACTION=='precheck':
 assert not root.exists();root.mkdir(parents=True);print(json.dumps({'node':NODE,'provider_pid':pid,'model_stopped':True,'only_owned_provider_NPU':True}))
else:
 assert root.is_dir()
 stop=cmd(['docker','stop','--time','60',CID],90);state=cmd(['docker','inspect',CID,'--format','{{json .State}}'])
 logs=cmd(['docker','logs',CID],60);archives={}
 for stream in ['stdout','stderr']:
  raw=logs[stream].encode();p=root/(stream+'.gz');p.write_bytes(gzip.compress(raw));archives[stream]={'path':str(p),'raw_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':len(raw)}
 status=json.loads(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1645-sglang-four-node-pool-start-v1/provider-status.json').read_text());after=cmd(['npu-smi','info'])
 result={'node':NODE,'stop':stop,'state_after':state,'provider_status':status,'logs':archives,'npu_after':after}
 assert stop['rc']==0 and not json.loads(state['stdout'])['Running'] and status['phase']=='CLOSE_COMPLETE' and status['events'][-1]['rc']==0 and after['stdout'].count('No running processes')==8
 print(json.dumps(result))
'''
META_AUDIT=r'''import subprocess,json,pathlib,hashlib
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1645-sglang-four-node-pool-start-v1' and d['State']['Running']
code="import pathlib,json,hashlib;p=pathlib.Path('/var/log/memcache_hybrid/logs/mmc-meta-audit.log');raw=p.read_bytes();lines=[x for x in raw.decode().splitlines() if 'Metrics summary:' in x];print(json.dumps({'sha256':hashlib.sha256(raw).hexdigest(),'last_summary':lines[-1]}))"
r=subprocess.run(['docker','exec',CID,'python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0;r=json.loads(r.stdout)
fields=dict(x.split('=',1) for x in r['last_summary'].split() if '=' in x)
assert int(fields['keys'])==0 and fields['dram_used']=='0/609885356032','Pool must be empty; do not clear existing application data'
print(json.dumps(r))
'''
def save(name,data):(RUN/name).write_text(json.dumps(data,indent=2)+'\n')
def provider(n,action):
    c=POOL['providers'][str(n)];code=m.replace(HOST,{'CID':c['container_id'],'NAME':c['name'],'MODEL':FAILED[str(n)]['container_id'],'NODE':n,'IMAGE':m.IMAGE,'REMOTE':REMOTE,'ACTION':action});return m.invoke(n,code,timeout=240)
def meta_audit():
    return m.invoke(34,m.replace(META_AUDIT,{'CID':POOL['meta']['container_id'],'NAME':POOL['meta']['name'],'IMAGE':m.IMAGE}),timeout=80)
def meta_stop():
    code=r'''import subprocess,json,pathlib,gzip,hashlib
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1645-sglang-four-node-pool-start-v1'
r=subprocess.run(['docker','stop','--time','30',CID],capture_output=True,text=True,timeout=60);assert r.returncode==0
state=subprocess.run(['docker','inspect',CID,'--format','{{json .State}}'],capture_output=True,text=True,timeout=20);assert state.returncode==0;state=json.loads(state.stdout);assert not state['Running']
logs=subprocess.run(['docker','logs',CID],capture_output=True,text=True,timeout=60);archives={}
for stream,text in [('stdout',logs.stdout),('stderr',logs.stderr)]:
 raw=text.encode();p=pathlib.Path(REMOTE)/('meta-'+stream+'.gz');assert not p.exists();p.write_bytes(gzip.compress(raw));archives[stream]={'path':str(p),'raw_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':len(raw)}
print(json.dumps({'container_id':CID,'stop_rc':r.returncode,'state_after':state,'logs':archives}))
'''
    return m.invoke(34,m.replace(code,{'CID':POOL['meta']['container_id'],'NAME':POOL['meta']['name'],'IMAGE':m.IMAGE,'REMOTE':REMOTE}),timeout=170)
def main():
    stage=json.loads((m.b.ROOT/'runs/20261017T1700-sglang-shared-stage-v4/receipt.json').read_text());assert stage['SDK_preflight']['sglang']==0 and all(x==0 for x in stage['staged'].values())
    fixture=json.loads((m.b.ROOT/'runs/20261017T1695-pool-supervisor-cpu-fixture-v1/receipt.json').read_text());assert fixture['rc']==0
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'precheck':{},'nodes':{},'model_requests':0,'meta_stop_rc':None}
    try:
        r=meta_audit();save('meta-empty-precheck.json',r);assert r['rc']==0
        for n in [34,35,36,41]:
            r=provider(n,'precheck');save(f'precheck-{n}.json',r);receipt['precheck'][n]=r['rc'];assert r['rc']==0
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futures={ex.submit(provider,n,'stop'):n for n in [34,35,36,41]}
            for f in concurrent.futures.as_completed(futures):
                n=futures[f]
                try:r=f.result()
                except Exception as e:r={'error':repr(e)}
                save(f'node-{n}.json',r);receipt['nodes'][n]=r.get('rc',r.get('error'))
        assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values()),'Preserve Meta if any pool shutdown incomplete'
        r=meta_stop();save('meta-stop.json',r);receipt['meta_stop_rc']=r['rc'];assert r['rc']==0
    finally:save('receipt.json',receipt)
if __name__=='__main__':main()
