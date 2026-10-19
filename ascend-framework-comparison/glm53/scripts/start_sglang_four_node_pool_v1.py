#!/usr/bin/env python3
"""Once-only four-node SDK pool startup; retain ready owned pool, close on failure."""
import concurrent.futures, hashlib, importlib.util, json, re, shlex
from pathlib import Path

P=Path(__file__).resolve(); prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RID='20261017T1645-sglang-four-node-pool-start-v1'; RUN=b.ROOT/'runs'/RID
REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
STAGED='/data/wio/inference-foundry/glm53/overlays/sglang-sdma-v8-20261017'
IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
MAN=b.ROOT/'config/standalone-budget-stage-v2.json'
assert hashlib.sha256(MAN.read_bytes()).hexdigest()=='721371d6904e224c8e00d3a36e9ec773d212c66bf66daa1f8c83571aa14723bf'
BUNDLE=next(x for x in json.loads(MAN.read_text())['bundles'] if x['framework']=='sglang')
FILES={k:v for k,v in BUNDLE['files'].items() if k in ['provider.py','config/meta.conf','provider-config/node-34.json','provider-config/node-35.json','provider-config/node-36.json','provider-config/node-41.json']}
assert len(FILES)==6
for v in FILES.values():assert hashlib.sha256((b.ROOT/v['local']).read_bytes()).hexdigest()==v['sha256']
OLD={34:'9844ec4193f7bc662ac43a62e2606a72f8299992283f27d2005a8ce9c643f805',35:'bf1859f9bad4d03c10e78a647703dcff951d4dc81dd53dd6e941ba0d97e928db',36:'2af3259c99c29dd00a7e8ee84906c96c4657bd5050a9190ebcc093d49991dbad',41:'9d3495a179b91716c8685ecb18a8e78001abbad6e7a90b4a64f329f33c319196'}

def replace(code,values):
    for k,v in values.items():code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
    return code

def invoke(n,code,timeout=120):
    pod,host=b.NODES[n];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
    assert r['rc']==0 and r['stdout']==f'172.16.12.{n}'
    code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
    return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)

def name(n,meta=False):return 'glm53-sglang-four-node-pool-v1-'+('meta-' if meta else '')+f'20261017-{n}'

def precheck(n):
    code='''import pathlib,subprocess,json,hashlib
def cmd(a):
 r=subprocess.run(a,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr;return r.stdout
assert cmd(['docker','image','inspect',IMAGE,'--format','{{.Id}}']).strip()==IMAGE
old=json.loads(cmd(['docker','inspect',OLD]))[0]
assert old['Id']==OLD and old['Image']==IMAGE and old['Config']['Labels']['project_id']=='glm53' and not old['State']['Running']
npu=cmd(['npu-smi','info']);assert npu.count('No running processes')==8
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
for k,v in FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/k).read_bytes()).hexdigest()==v['sha256']
mem=pathlib.Path('/proc/meminfo').read_text();available=int(next(l.split()[1] for l in mem.splitlines() if l.startswith('MemAvailable:')))*1024
assert available>1100*1024**3
if NODE==34:
 for port in [29780,29781,29782]:
  sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
root.mkdir(parents=True)
print(json.dumps({'node':NODE,'empty_NPUs':8,'mem_available_bytes':available,'candidate_bytes':{34:136,35:112,36:160,41:160}[NODE]*1024**3,'scan_does_not_establish_capacity':True}))
'''
    return invoke(n,replace(code,{'IMAGE':IMAGE,'OLD':OLD[n],'REMOTE':REMOTE,'FILES':FILES,'STAGED':STAGED,'NODE':n}))

def start(n,meta=False):
    nm=name(n,meta)
    args=['docker','run','-d','--pull=never','--network=host','--pids-limit=1024','--memory='+('8g' if meta else '800g'),'--cpus=4','--ulimit','memlock=-1','--label','project_id=glm53','--label','run_id='+RID,'--name',nm,'--log-opt','max-size=128m','--log-opt','max-file=8','-v',STAGED+':/opt/glm53/staged:ro','-v',REMOTE+':/run/glm53','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONHASHSEED=0','-e','ACL_OP_INIT_MODE=1','-e','ASCEND_ENABLE_USE_FABRIC_MEM=1','-e','MMC_META_CONFIG_PATH=/opt/glm53/staged/config/meta.conf']
    if meta:
        boot="exec python3 -c 'import memcache_hybrid as mc,sys;from importlib.metadata import version;assert version(\"memcache_hybrid\")==\"1.3.0\";sys.exit(mc.MetaService.main())'"
    else:
        for dev in ['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
        args+=['-e','ASCEND_RT_VISIBLE_DEVICES=0']
        boot='exec python3 /opt/glm53/staged/provider.py --config /opt/glm53/staged/provider-config/node-'+str(n)+'.json --sha256 '+FILES[f'provider-config/node-{n}.json']['sha256']
    args+=['--entrypoint','/bin/bash',IMAGE,'-c','set -eu; '+boot]
    code='''import pathlib,subprocess,json,time
root=pathlib.Path(REMOTE)
with (root/('start-meta.once' if META else 'start-provider.once')).open('x') as f:f.write(NAME)
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=40)
result={'node':NODE,'name':NAME,'run_rc':r.returncode,'container_id':r.stdout.strip(),'stderr':r.stderr}
(root/('meta-start.json' if META else 'provider-start.json')).write_text(json.dumps(result,indent=2)+'\\n')
assert r.returncode==0,r.stderr
deadline=time.monotonic()+(35 if META else 150);ready=False
while time.monotonic()<deadline:
 if META:
  try:
   with socket.create_connection(('172.16.12.34',29780),timeout=1):ready=True;break
  except OSError:pass
 else:
  p=root/'provider-status.json'
  if p.is_file():
   status=json.loads(p.read_text());result['provider_status']=status
   if status['phase']=='READY':ready=True;break
   if any(e['event']=='FAILED' for e in status['events']):break
 state=subprocess.run(['docker','inspect',NAME,'--format','{{json .State}}'],capture_output=True,text=True,timeout=10)
 if state.returncode==0 and not json.loads(state.stdout)['Running']:break
 time.sleep(.5)
result['ready']=ready
print(json.dumps(result))
'''
    return invoke(n,replace(code,{'REMOTE':REMOTE,'META':meta,'NAME':nm,'NODE':n,'ARGS':args}),timeout=240)

def snapshot_stop(n,meta=False):
    code='''import pathlib,subprocess,json,hashlib,gzip,re
root=pathlib.Path(REMOTE)
def cmd(a,t=30):
 r=subprocess.run(a,capture_output=True,text=True,timeout=t);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
r=cmd(['docker','inspect',NAME]);out={'node':NODE,'name':NAME,'inspect_rc':r['rc']}
if r['rc']==0:
 info=json.loads(r['stdout'])[0];assert info['Name']=='/'+NAME and info['Image']==IMAGE and info['Config']['Labels']['project_id']=='glm53' and info['Config']['Labels']['run_id']==RID
 cid=info['Id'];out['container_id']=cid;out['state_before']=info['State']
 # Capture own native logs before closing and retain stopped container for diagnostics.
 for directory in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']:
  out['copy_'+directory]=cmd(['docker','cp',cid+':'+directory,str(root/('meta-' if META else 'provider-'))+directory.rsplit('/',1)[1]],60)
 logs=cmd(['docker','logs',cid],60)
 for key in ['stdout','stderr']:
  raw=logs[key].encode();p=root/(('meta-' if META else 'provider-')+key+'.gz');p.write_bytes(gzip.compress(raw));out[key+'_archive']={'path':str(p),'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest()};out[key+'_tail']=raw[-20000:].decode(errors='replace')
 out['native_selected_before_close']=[]
 for p in root.glob(('meta-' if META else 'provider-')+'memcache_hybrid/**/*'):
  if p.is_file() and p.stat().st_size<20000000:
   rows=[line[:1000] for line in p.read_text(errors='replace').splitlines() if re.search('total.?size|capacity|mount|register|lease.?ttl|dram|HalMemCreate',line,re.I)]
   if rows:out['native_selected_before_close'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'matches':rows[-100:]})
 out['stop']=cmd(['docker','stop','--time','60',cid],90)
 out['state_after']=cmd(['docker','inspect',cid,'--format','{{json .State}}'])
 if not META:out['npu_after']=cmd(['npu-smi','info'])
print(json.dumps(out))
'''
    return invoke(n,replace(code,{'REMOTE':REMOTE,'NAME':name(n,meta),'IMAGE':IMAGE,'RID':RID,'META':meta,'NODE':n}),timeout=260)

def save(filename,value):
    (RUN/filename).write_text(json.dumps(value,indent=2)+'\n');return value

def live_meta_snapshot():
    code='''import pathlib,subprocess,json,hashlib,re
r=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=20);assert r.returncode==0
d=json.loads(r.stdout)[0];assert d['Image']==IMAGE and d['Name']=='/'+NAME and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID
dst=pathlib.Path(REMOTE)/'live-meta-memcache-logs';r=subprocess.run(['docker','cp',d['Id']+':/var/log/memcache_hybrid',str(dst)],capture_output=True,text=True,timeout=60)
out={'copy_rc':r.returncode,'stderr':r.stderr,'files':[]}
for p in dst.rglob('*'):
 if p.is_file() and p.stat().st_size<20000000:
  raw=p.read_bytes();rows=[x[:1000] for x in raw.decode(errors='replace').splitlines() if re.search('total.?size|capacity|mount|register|lease.?ttl|dram',x,re.I)]
  out['files'].append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'matches':rows[-100:]})
print(json.dumps(out))
'''
    return invoke(34,replace(code,{'NAME':name(34,True),'IMAGE':IMAGE,'RID':RID,'REMOTE':REMOTE}),timeout=100)

def main():
    assert json.loads((b.ROOT/'runs/20261017T1610-sglang-standalone-native-preflight-v1/receipt.json').read_text())['preflight_rc']==0
    assert all(v==0 for v in json.loads((b.ROOT/'runs/20261017T1616-sglang-D-empty-memory-scan-v1/receipt.json').read_text())['nodes'].values())
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'candidate_total_bytes':568*1024**3,'classification':'SDK_ONLY_POOL_START_NOT_ACCEPTED_BASELINE','precheck':{},'providers':{},'cleanup':{}}
    stage=json.loads((b.ROOT/'runs/20261017T1640-sglang-four-node-pool-stage-v2/receipt.json').read_text())
    assert stage['SDK_preflight']['sglang']==0 and all(stage['staged'][str(n)]==0 for n in [34,35,36,41])
    failed=json.loads((b.ROOT/'results/sglang-standalone-allocation-probe-v1.json').read_text());assert failed['Meta_stopped_last']
    may_exist=False; keep=False
    try:
        for n in [34,35,36,41]:
            r=save(f'precheck-{n}.json',precheck(n));receipt['precheck'][n]=r['rc'];assert r['rc']==0
        may_exist=True
        r=save('meta-start.json',start(34,True));assert r['rc']==0 and json.loads(r['stdout'])['ready']
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fs={pool.submit(start,n):n for n in [34,35,36,41]}
            for f in concurrent.futures.as_completed(fs):
                n=fs[f]
                try:r=f.result()
                except Exception as e:r={'error':repr(e)}
                save(f'provider-{n}.json',r);receipt['providers'][n]=r;print('provider',n,r.get('rc',r.get('error')),flush=True)
        keep=len(receipt['providers'])==4 and all(r.get('rc')==0 and json.loads(r.get('stdout','{}')).get('ready') for r in receipt['providers'].values())
        receipt['pool_kept_running']=keep
        save('live-meta-snapshot.json',live_meta_snapshot())
        receipt['actual_capacity_qualification']='PENDING_NATIVE_META_CAPACITY_AUDIT; SDKinit0 isnotactualcapacityproof'

    finally:
        if may_exist and not keep:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs={pool.submit(snapshot_stop,n):n for n in [34,35,36,41]}
                for f in concurrent.futures.as_completed(fs):
                    n=fs[f]
                    try:r=f.result()
                    except Exception as e:r={'error':repr(e)}
                    save(f'provider-stop-{n}.json',r);receipt['cleanup'][n]=r.get('rc',r.get('error'))
            try:r=snapshot_stop(34,True)
            except Exception as e:r={'error':repr(e)}
            save('meta-stop.json',r);receipt['cleanup']['meta']=r.get('rc',r.get('error'))
        save('receipt.json',receipt)
if __name__=='__main__':main()
