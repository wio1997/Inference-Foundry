#!/usr/bin/env python3
"""Once-only four-node SDK pool startup; retain ready owned pool, close on failure."""
import concurrent.futures, hashlib, importlib.util, json, re, shlex
from pathlib import Path

P=Path(__file__).resolve(); prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RID='20261017T1750-vllm-shared-pool-v1'; RUN=b.ROOT/'runs'/RID
REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
STAGED='/data/wio/inference-foundry/glm53/overlays/vllm-sdma-v8-20261017'
IMAGE='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
MAN=b.ROOT/'config/standalone-budget-stage-v8.json'
assert hashlib.sha256(MAN.read_bytes()).hexdigest()=='8aaf1560dc3b63145bda7b228eccb8f39940d7571c01c57713a7ff943fb889fe'
BUNDLE=next(x for x in json.loads(MAN.read_text())['bundles'] if x['framework']=='vllm')
FILES=BUNDLE['files']
assert {'supervisor.py','entry.py','provider.py','config/meta.conf','config/local.conf',*[f'supervisor-config/node-{n}.json' for n in [30,31,43,45]]}<=set(FILES)
for v in FILES.values():assert hashlib.sha256((b.ROOT/v['local']).read_bytes()).hexdigest()==v['sha256']
OLD={30: 'c37c2e631f60ff4a4517d129f0a5341a56ab06c0d448463b466ca3c35bb06786', 31: '2f48868b06679ffd0750c5165fcd64840fbc9de43b6d4647e4b0bdbba47555dd', 43: '101a44f8ff8913bc5d103fb28ac3c09754005652ef06f805195187dd17db8a3c', 45: '5abe1c23a1a68e0e0e033048fe4d57864347213604b49ff89ea1c294fc861e12'}
SOURCE=json.loads((b.ROOT/'config/overlays/vllm-sdma-v5/manifest.json').read_text())

def replace(code,values):
    for k,v in values.items():code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
    return code

def invoke(n,code,timeout=120):
    pod,host=b.NODES[n];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
    assert r['rc']==0 and r['stdout']==f'172.16.12.{n}'
    code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
    return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)

def name(n,meta=False):return 'glm53-vllm-shared-v1-'+('meta-' if meta else '')+f'20261017-{n}'

def precheck(n):
    code='''import pathlib,subprocess,json,hashlib
def cmd(a):
 r=subprocess.run(a,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr;return r.stdout
assert cmd(['docker','image','inspect',IMAGE,'--format','{{.Id}}']).strip()==IMAGE
old=json.loads(cmd(['docker','inspect',OLD]))[0]
assert old['Id']==OLD and old['Image']==IMAGE and old['Config']['Labels']['project_id']=='glm53' and old['Config']['Labels']['run_id']=='20261017T1100-vllm-correctness-v4' and not old['State']['Running']
npu=cmd(['npu-smi','info']);assert npu.count('No running processes')==8
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
for k,v in FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/k).read_bytes()).hexdigest()==v['sha256']
sdk=pathlib.Path('/data/wio/inference-foundry/glm53/overlays/sdk13-20261017')
assert (sdk/'archive.sha256').read_text().strip()=='f858a88e60e33f2c954bd9920b56b96193e2be53749fa16e83316d461cb731b3'
sdk_manifest=json.loads((sdk/'_glm53_overlay_manifest.json').read_text());assert sdk_manifest['source_image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263' and len(sdk_manifest['files'])==108
sizes=0
for entry in sdk_manifest['files']:
 rel=pathlib.PurePosixPath(entry['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in sdk_manifest['packages']
 raw=(sdk/str(rel)).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256'];sizes+=len(raw)
assert sizes==48262231
mem=pathlib.Path('/proc/meminfo').read_text();available=int(next(l.split()[1] for l in mem.splitlines() if l.startswith('MemAvailable:')))*1024
assert available>1100*1024**3
if NODE==30:
 for port in [29680,29681,29682]:
  sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8');weights=[p for p in model.rglob('*') if p.is_file()];assert len(weights)==193 and sum(p.stat().st_size for p in weights)==764240829290
assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()=='ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca'
network=json.loads(cmd(['ip','-j','address','show']));interfaces=[row['ifname'] for row in network if any(a.get('local')=='172.16.12.'+str(NODE) for a in row.get('addr_info',[]))];assert len(interfaces)==1
supervisor=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text());assert supervisor['control_interface']==interfaces[0]
root.mkdir(parents=True)
print(json.dumps({'node':NODE,'empty_NPUs':8,'mem_available_bytes':available,'candidate_bytes':{30:136,31:112,43:160,45:160}[NODE]*1024**3,'scan_does_not_establish_capacity':True}))
'''
    return invoke(n,replace(code,{'IMAGE':IMAGE,'OLD':OLD[n],'REMOTE':REMOTE,'FILES':FILES,'STAGED':STAGED,'NODE':n}))

def start(n,meta=False):
    nm=name(n,meta)
    args=['docker','run','-d','--pull=never','--network=host','--pids-limit='+('2048' if meta else '32768'),'--memory='+('8g' if meta else '1700g'),'--ulimit','memlock=-1','--label','project_id=glm53','--label','run_id='+RID,'--name',nm,'--log-opt','max-size=128m','--log-opt','max-file=8','-v',STAGED+':/opt/glm53/bundle:ro','-v',STAGED+'/source:/opt/glm53/source:ro','-v',STAGED+'/config:/opt/glm53/config:ro','-v','/data/wio/inference-foundry/glm53/overlays/sdk13-20261017:/opt/glm53/sdk13:ro','-v',REMOTE+':/run/glm53','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONHASHSEED=0','-e','ACL_OP_INIT_MODE=1','-e','ASCEND_ENABLE_USE_FABRIC_MEM=1','-e','MMC_META_CONFIG_PATH=/opt/glm53/config/meta.conf']
    if meta:
        args+=['--cpus=4']
        boot="exec python3 -c 'import memcache_hybrid as mc,sys;from importlib.metadata import version;assert version(\"memcache_hybrid\")==\"1.3.0\";sys.exit(mc.MetaService.main())'"
    else:
        args+=['--shm-size=64g','--ulimit','nofile=655360:655360','-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro']
        for e in SOURCE['files']:args+=['-v',STAGED+'/source/'+e['local']+':'+e['container_path']+':ro']
        for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
        # Model-only MMC_LOCAL_CONFIG_PATH/PD-store/HCCL env belongs to the
        # model child. The contributor must not inherit inference dram0 config.
        args+=['-e','ASCEND_VISIBLE_DEVICES=0-15','-e','ASCEND_RT_VISIBLE_DEVICES='+','.join(map(str,range(16)))]
        boot='exec python3 /opt/glm53/bundle/supervisor.py --config /opt/glm53/bundle/supervisor-config/node-'+str(n)+'.json --sha256 '+FILES[f'supervisor-config/node-{n}.json']['sha256']
    args+=['--entrypoint','/bin/bash',IMAGE,'-c','set -eu; export PYTHONPATH=/opt/glm53/sdk13:/opt/glm53/source/python:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; '+boot]

    code='''import pathlib,subprocess,json,time
root=pathlib.Path(REMOTE)
with (root/('start-meta.once' if META else 'start-provider.once')).open('x') as f:f.write(NAME)
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=40)
result={'node':NODE,'name':NAME,'run_rc':r.returncode,'container_id':r.stdout.strip(),'stderr':r.stderr}
(root/('meta-start.json' if META else 'provider-start.json')).write_text(json.dumps(result,indent=2)+'\\n')
assert r.returncode==0,r.stderr
deadline=time.monotonic()+(35 if META else 180);ready=False
while time.monotonic()<deadline:
 if META:
  try:
   with socket.create_connection(('172.16.12.30',29680),timeout=1):ready=True;break
  except OSError:pass
 else:
  p=root/'provider-status.json'
  if p.is_file():
   status=json.loads(p.read_text());result['provider_status']=status
   sp=root/'supervisor-status.json'
   if sp.is_file():
    supervisor=json.loads(sp.read_text());result['supervisor_status']=supervisor
    if supervisor['phase']=='POOL_READY_WAITING_MODEL_SIGNAL' and status['phase']=='READY':
     assert supervisor['model_started'] is False;ready=True;break
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

def native_preflight(n):
    inner=r"""import json,os,pathlib
import torch,torch_npu,acl
count=torch.npu.device_count();ac=acl.rt.get_device_count()
assert count==16 and torch.npu.is_available() and list(ac)==[16,0],(count,ac)
print('GLM53_SHARED_HARDWARE='+json.dumps({'node':NODE,'torch_device_count':count,'ACL_count':ac,'soc_name':acl.get_soc_name(),'acl_module':acl.__file__,'PYTHONPATH':os.environ.get('PYTHONPATH'),'model_requests':0,'SDK_init_calls':0}),flush=True)
path=pathlib.Path('/opt/glm53/bundle/preflight.py')
exec(compile(path.read_bytes(),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
"""
    inner=replace(inner,{'NODE':n})
    code=r"""import pathlib,subprocess,json
r=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID and d['State']['Running'] and not d['HostConfig']['Privileged']
assert len(d['HostConfig']['Devices'])==19
root=pathlib.Path(REMOTE);state=json.loads((root/'supervisor-status.json').read_text());assert state['phase']=='POOL_READY_WAITING_MODEL_SIGNAL' and state['model_started'] is False
model_env=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text())['model_env']
assert 'PYTHONPATH' not in model_env and 'LD_LIBRARY_PATH' not in model_env
bootstrap='set -eu; export PYTHONPATH=/opt/glm53/sdk13:/opt/glm53/source/python:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; exec python3 -c "$1"'
exec_args=['docker','exec',d['Id'],'env']+[k+'='+v for k,v in model_env.items()]+['/bin/bash','-c',bootstrap,'glm53-native-preflight',INNER]
r=subprocess.run(exec_args,capture_output=True,text=True,timeout=240)
after=json.loads((root/'supervisor-status.json').read_text());assert after['model_started'] is False
print(json.dumps({'node':NODE,'container_id':d['Id'],'query_rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'supervisor_before':state,'supervisor_after':after}))
raise SystemExit(r.returncode)
"""
    return invoke(n,replace(code,{'NAME':name(n),'IMAGE':IMAGE,'RID':RID,'REMOTE':REMOTE,'INNER':inner,'NODE':n,'STAGED':STAGED}),timeout=300)

def live_meta_snapshot():
    code='''import pathlib,subprocess,json,hashlib,re
r=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=20);assert r.returncode==0
d=json.loads(r.stdout)[0];assert d['Image']==IMAGE and d['Name']=='/'+NAME and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID
dst=pathlib.Path(REMOTE)/'live-meta-memcache-logs';r=subprocess.run(['docker','cp',d['Id']+':/var/log/memcache_hybrid',str(dst)],capture_output=True,text=True,timeout=60)
out={'copy_rc':r.returncode,'stderr':r.stderr,'files':[]}
for p in dst.rglob('*'):
 if p.is_file() and p.stat().st_size<20000000:
  raw=p.read_bytes();rows=[x[:1000] for x in raw.decode(errors='replace').splitlines() if re.search('total.?size|capacity|mount|register|lease.?ttl|dram|evict_threshold',x,re.I)]
  out['files'].append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'matches':rows[-100:]})
print(json.dumps(out))
'''
    return invoke(30,replace(code,{'NAME':name(30,True),'IMAGE':IMAGE,'RID':RID,'REMOTE':REMOTE}),timeout=100)

def main():
    stage=json.loads((b.ROOT/'runs/20261017T1746-vllm-shared-stage-v2/receipt.json').read_text())
    assert stage['manifest_sha256']=='8aaf1560dc3b63145bda7b228eccb8f39940d7571c01c57713a7ff943fb889fe' and stage['SDK_preflight']['vllm']==0 and len(stage['staged'])==4 and all(stage['staged'][str(n)]==0 for n in [30,31,43,45])
    stopped=json.loads((b.ROOT/'runs/20261017T1745-stop-vllm-v4-for-shared-v1/receipt.json').read_text());assert stopped['meta_stop']==0 and len(stopped['stops'])==4 and all(v==0 for v in stopped['stops'].values())
    scan=json.loads((b.ROOT/'runs/20261017T1751-vllm-empty-memory-scan-v1/receipt.json').read_text());assert len(scan['nodes'])==4 and all(v==0 for v in scan['nodes'].values())
    signal_path=b.ROOT/'results/sdk-postinit-signal-qualification-v1.json';assert hashlib.sha256(signal_path.read_bytes()).hexdigest()=='cfe77966f53bf16811c9157d0a1165de4073f20938718dfb37f458cadf65dcf1'
    signal_qualified=json.loads(signal_path.read_text());assert signal_qualified['pass'] and signal_qualified['actor_exit']==0 and signal_qualified['Meta_exit']==0
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'candidate_total_bytes':568*1024**3,'classification':'SHARED_CONTAINER_POOL_ONLY_MODELS_NOT_STARTED','precheck':{},'providers':{},'native_preflight':{},'cleanup':{}}
    may_exist=False; keep=False
    try:
        for n in [30,31,43,45]:
            r=save(f'precheck-{n}.json',precheck(n));receipt['precheck'][n]=r['rc'];assert r['rc']==0
        may_exist=True
        r=save('meta-start.json',start(30,True));assert r['rc']==0 and json.loads(r['stdout'])['ready']
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fs={pool.submit(start,n):n for n in [30,31,43,45]}
            for f in concurrent.futures.as_completed(fs):
                n=fs[f]
                try:r=f.result()
                except Exception as e:r={'error':repr(e)}
                save(f'provider-{n}.json',r);receipt['providers'][n]=r;print('provider',n,r.get('rc',r.get('error')),flush=True)
        keep=len(receipt['providers'])==4 and all(r.get('rc')==0 and json.loads(r.get('stdout','{}')).get('ready') for r in receipt['providers'].values())
        if keep:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs={pool.submit(native_preflight,n):n for n in [30,31,43,45]}
                for f in concurrent.futures.as_completed(fs):
                    n=fs[f]
                    try:r=f.result()
                    except Exception as e:r={'error':repr(e)}
                    save(f'native-preflight-{n}.json',r);receipt['native_preflight'][n]=r.get('rc',r.get('error'))
            keep=len(receipt['native_preflight'])==4 and all(x==0 for x in receipt['native_preflight'].values())
        receipt['pool_kept_running']=keep
        save('live-meta-snapshot.json',live_meta_snapshot())
        receipt['actual_capacity_qualification']='PENDING_NATIVE_META_CAPACITY_AUDIT; SDKinit0 isnotactualcapacityproof'

    finally:
        if may_exist and not keep:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs={pool.submit(snapshot_stop,n):n for n in [30,31,43,45]}
                for f in concurrent.futures.as_completed(fs):
                    n=fs[f]
                    try:r=f.result()
                    except Exception as e:r={'error':repr(e)}
                    save(f'provider-stop-{n}.json',r);receipt['cleanup'][n]=r.get('rc',r.get('error'))
            try:r=snapshot_stop(30,True)
            except Exception as e:r={'error':repr(e)}
            save('meta-stop.json',r);receipt['cleanup']['meta']=r.get('rc',r.get('error'))
        save('receipt.json',receipt)
    return 0 if keep else 1
if __name__=='__main__':raise SystemExit(main())
