#!/usr/bin/env python3
"""C32-only one-time restore of accepted L04 source/config; new owned pool, no repeated SDK microbenchmarks."""
import concurrent.futures, hashlib, importlib.util, json, re, shlex
from pathlib import Path

P=Path(__file__).resolve(); prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RID='20261018T-L10-sglang-C32-target-graph-fullmesh-v3'; RUN=b.ROOT/'runs'/RID
REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
STAGED='/data/wio/inference-foundry/glm53/overlays/sglang-C32-target-graph-fullmesh-v3-20261018'
IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
MAN=b.ROOT/'config/sglang-C32-target-graph-fullmesh-stage-v3.json'
assert hashlib.sha256(MAN.read_bytes()).hexdigest()=='4a00cb3433b0bc96ee92de899d371a3c3174027e3ceb8ee338ccffa493fe0291'
BUNDLE=next(x for x in json.loads(MAN.read_text())['bundles'] if x['framework']=='sglang')
FILES=BUNDLE['files']
assert {'supervisor.py','entry.py','provider.py','config/meta.conf','config/local.conf',*[f'supervisor-config/node-{n}.json' for n in [34,35,36,41]]}<=set(FILES)
for v in FILES.values():assert hashlib.sha256((b.ROOT/v['local']).read_bytes()).hexdigest()==v['sha256']
OLD={34: '10b2bb84cb4b0f7cb1e59fc80a9a1bfb5a100614cc2ef4315f02324cefbfea03', 35: 'bd94a86cc3db256f518cb8247aaaefd5014b237e027cc67a372a69508e8d505f', 36: '3e3c4ac720ea1e9445a53dc29ec762fc25237d24b62032f56d688ccada627543', 41: '3e5a78a9f6e1089b8b9a47c4c6ca7e05cf706fd0695f81751d469e06b3d78fc8'}
SOURCE=json.loads((b.ROOT/'config/overlays/sglang-sdma-v1/manifest.json').read_text())

def replace(code,values):
    for k,v in values.items():code=re.sub(r'\b'+k+r'\b',lambda _:repr(v),code)
    return code

def invoke(n,code,timeout=120):
    pod,host=b.NODES[n];r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
    assert r['rc']==0 and r['stdout']==f'172.16.12.{n}'
    code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code
    return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)

def name(n,meta=False):return 'glm53-sglang-C32-target-graph-fullmesh-v3-'+('meta-' if meta else '')+f'20261018-{n}'

def precheck(n):
    code='''import pathlib,subprocess,json,hashlib
def cmd(a):
 r=subprocess.run(a,capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr;return r.stdout
assert cmd(['docker','image','inspect',IMAGE,'--format','{{.Id}}']).strip()==IMAGE
old=json.loads(cmd(['docker','inspect',OLD]))[0]
assert old['Id']==OLD and old['Image']==IMAGE and old['Config']['Labels']['project_id']=='glm53' and old['Config']['Labels']['run_id']=='20261018T-L10-sglang-C32-target-graph-fullmesh-v2' and not old['State']['Running']
npu=cmd(['npu-smi','info']);assert npu.count('No running processes')==8
root=pathlib.Path(REMOTE);assert not root.exists(),'No overwrite/replay'
for k,v in FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/k).read_bytes()).hexdigest()==v['sha256']
mem=pathlib.Path('/proc/meminfo').read_text();available=int(next(l.split()[1] for l in mem.splitlines() if l.startswith('MemAvailable:')))*1024
assert available>1100*1024**3
if NODE==34:
 for port in [29780,29781,29782]:
  sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
model=pathlib.Path('/data/wio/models/GLM-5.3-w8a8');weights=[p for p in model.rglob('*') if p.is_file()];assert len(weights)==193 and sum(p.stat().st_size for p in weights)==764240829290
assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest()=='ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca'
network=json.loads(cmd(['ip','-j','address','show']));interfaces=[row['ifname'] for row in network if any(a.get('local')=='172.16.12.'+str(NODE) for a in row.get('addr_info',[]))];assert len(interfaces)==1
supervisor=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text());assert supervisor['control_interface']==interfaces[0]
root.mkdir(parents=True)
print(json.dumps({'node':NODE,'empty_NPUs':8,'mem_available_bytes':available,'candidate_bytes':{34:136,35:112,36:160,41:160}[NODE]*1024**3,'scan_does_not_establish_capacity':True}))
'''
    return invoke(n,replace(code,{'IMAGE':IMAGE,'OLD':OLD[n],'REMOTE':REMOTE,'FILES':FILES,'STAGED':STAGED,'NODE':n}))

def start(n,meta=False):
    nm=name(n,meta)
    args=['docker','run','-d','--pull=never','--network=host','--pids-limit='+('2048' if meta else '32768'),'--memory='+('8g' if meta else '1700g'),'--ulimit','memlock=-1','--label','project_id=glm53','--label','run_id='+RID,'--name',nm,'--log-opt','max-size=128m','--log-opt','max-file=8','-v',STAGED+':/opt/glm53/source:ro','-v',REMOTE+':/run/glm53','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONHASHSEED=0','-e','ACL_OP_INIT_MODE=1','-e','ASCEND_ENABLE_USE_FABRIC_MEM=1','-e','MMC_META_CONFIG_PATH=/opt/glm53/source/config/meta.conf']
    if meta:
        args+=['--cpus=4']
        boot="exec python3 -c 'import memcache_hybrid as mc,sys;from importlib.metadata import version;assert version(\"memcache_hybrid\")==\"1.3.0\";sys.exit(mc.MetaService.main())'"
    else:
        args+=['--shm-size=64g','--ulimit','nofile=655360:655360','-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro']
        for e in SOURCE['files']:args+=['-v',STAGED+'/'+e['local']+':'+e['container_path']+':ro']
        for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',dev]
        # Model-only MMC_LOCAL_CONFIG_PATH/PD-store/HCCL env belongs to the
        # model child. The contributor must not inherit inference dram0 config.
        args+=['-e','ASCEND_VISIBLE_DEVICES=0-15','-e','ASCEND_RT_VISIBLE_DEVICES='+','.join(map(str,range(16)))]
        boot='exec python3 /opt/glm53/source/supervisor.py --config /opt/glm53/source/supervisor-config/node-'+str(n)+'.json --sha256 '+FILES[f'supervisor-config/node-{n}.json']['sha256']
    args+=['--entrypoint','/bin/bash',IMAGE,'-c','set -eu; '+boot]

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
   with socket.create_connection(('172.16.12.34',29780),timeout=1):ready=True;break
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
   if rows:out['native_selected_before_close'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'matches':rows[-100:],'critical_lines_complete':[x for x in raw.decode(errors='replace').splitlines() if re.search(r'Mount bm on loc|Unmount|UnMount|ock\\.mmc\\.meta\\.lease_ttl_ms|ock\\.mmc\\.evict_threshold_',x)]})
 out['stop']=cmd(['docker','stop','--time','240',cid],270)
 out['state_after']=cmd(['docker','inspect',cid,'--format','{{json .State}}'])
 if not META:out['npu_after']=cmd(['npu-smi','info'])
print(json.dumps(out))
'''
    return invoke(n,replace(code,{'REMOTE':REMOTE,'NAME':name(n,meta),'IMAGE':IMAGE,'RID':RID,'META':meta,'NODE':n}),timeout=380)

def save(filename,value):
    (RUN/filename).write_text(json.dumps(value,indent=2)+'\n');return value

def native_preflight(n):
    inner=r'''import pathlib,json,hashlib,os
import sys
assert sys.path[1]=='/opt/glm53/source/python',sys.path
import torch,torch_npu
count=torch.npu.device_count();assert count==16 and torch.npu.is_available(),count
import acl
acl_count=acl.rt.get_device_count();assert count==16 and torch.npu.is_available() and list(acl_count)==[16,0],(count,acl_count)
root=pathlib.Path('/opt/glm53/source');manifest=json.loads((root/'manifest.json').read_text())
for e in manifest['files']:assert hashlib.sha256(pathlib.Path(e['container_path']).read_bytes()).hexdigest()==e['patched_sha256']
for e in manifest['helpers']:assert hashlib.sha256((root/e['local']).read_bytes()).hexdigest()==e['sha256']
import glm53_sglang_sdma_audit as guard
guard.prepare_sdk()
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
assert AscendTransferEngine._get_transfer_protocol()=='sdma' and AscendTransferEngine._resolve_trans_op_type('sdma').value==1
from sglang.srt.environ import envs
assert envs.SGLANG_MOONCAKE_SEND_AUX_TCP.get() is False
from sglang.srt.server_args import prepare_server_args
from sglang.srt.arg_groups.overrides import resolved_view
config=json.loads((root/('config/node-'+str(NODE)+'.json')).read_text())
for key,value in config['environment_intent'].items():assert os.environ[key]==value
args=prepare_server_args(config['launch_args']);args.resolve_once();args.check_server_args();cfg=resolved_view(args)
assert cfg.tp_size==32 and cfg.ep_size==32 and cfg.dp_size==(4 if NODE in [34,35] else 16)
assert cfg.speculative_num_steps==5 and cfg.speculative_num_draft_tokens==6 and cfg.kv_cache_dtype=='bfloat16'
assert cfg.hicache_size==(9 if NODE in [34,35] else 8) and cfg.hicache_host_memory_mode==('buffer_only' if NODE in [34,35] else 'cache')
if NODE in [34,35]:
 backend=json.loads(config['launch_args'][config['launch_args'].index('--hicache-storage-backend-extra-config')+1]);assert backend['max_dram_size']=='160GB' and backend['dram_size']=='0GB' and backend['init_bm'] is True
assert cfg.moe_a2a_backend=='deepep'
if NODE in [34,35]:assert cfg.deepep_mode=='normal' and os.environ['DEEP_USE_MODE']=='alltoall'
else:
 assert cfg.deepep_mode=='low_latency' and os.environ['DEEP_USE_MODE']=='ops'
 assert os.environ['HCCL_BUFFSIZE']=='256' and os.environ['TASK_QUEUE_ENABLE']=='0' and os.environ['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK']=='32'
 assert cfg.cuda_graph_config.decode.backend=='full' and cfg.cuda_graph_config.decode.max_bs==4 and list(cfg.cuda_graph_config.decode.bs)==[1,2,3,4]
 assert cfg.cuda_graph_config.prefill.backend=='disabled'
 import deep_ep.strategies.low_latency_strategy as native_ops
 assert native_ops.OpsLowLatencyCommStrategy._glm53_C32_target_graph

import deep_ep.strategies.normal_strategy as strategy
assert getattr(strategy.AlltoAllNormalCommStrategy,'_glm53_metadata_batch_installed',False)==(NODE in [36,41])
assert (os.environ.get('GLM53_DEEPEP_METADATA_BATCH')=='1')==(NODE in [36,41])
if NODE in [36,41]:assert os.environ['GLM53_NODE']==str(NODE)
import sglang.srt.mem_cache.buffer_mode.pipeline as cache_pipeline
expected_cache_patch=NODE in [34,35] and False
assert bool(getattr(cache_pipeline.BufferModePipeline.enqueue_backup_intent,'_glm53_dedup_validation',False))==expected_cache_patch
assert (os.environ.get('GLM53_BUFFER_DEDUP_VALIDATE')=='1')==expected_cache_patch
if expected_cache_patch:assert os.environ['GLM53_NODE']==str(NODE)

print('GLM53_SHARED_PREFLIGHT='+json.dumps({'node':NODE,'torch_device_count':count,'ACL_count':acl_count,'soc_name':acl.get_soc_name(),'tp':cfg.tp_size,'dp':cfg.dp_size,'ep':cfg.ep_size,'MTP_steps':cfg.speculative_num_steps,'hicache_size_decimal_GB':cfg.hicache_size,'hicache_host_memory_mode':cfg.hicache_host_memory_mode,'acl_module':acl.__file__,'pythonpath':os.environ.get('PYTHONPATH'),'native_source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'engine_created':False,'decode_graph_backend':cfg.cuda_graph_config.decode.backend,'decode_graph_bs':cfg.cuda_graph_config.decode.bs,'decode_graph_max_bs':cfg.cuda_graph_config.decode.max_bs,'HCCL_BUFFSIZE':os.environ.get('HCCL_BUFFSIZE'),'dispatch_capacity':os.environ.get('SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK')}))
'''
    inner=replace(inner,{'NODE':n})
    code=r'''import pathlib,subprocess,json
r=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID and d['State']['Running'] and not d['HostConfig']['Privileged']
root=pathlib.Path(REMOTE);state=json.loads((root/'supervisor-status.json').read_text());assert state['phase']=='POOL_READY_WAITING_MODEL_SIGNAL' and state['model_started'] is False
model_env=json.loads((pathlib.Path(STAGED)/('supervisor-config/node-'+str(NODE)+'.json')).read_text())['model_env']
assert 'PYTHONPATH' not in model_env
bootstrap='set -eu; export PYTHONPATH=/opt/glm53/source/python:${PYTHONPATH:-}; exec python3 -c "$1"'
exec_args=['docker','exec',d['Id'],'env']+[k+'='+v for k,v in model_env.items()]+['/bin/bash','-c',bootstrap,'glm53-preflight-env',INNER]
r=subprocess.run(exec_args,capture_output=True,text=True,timeout=180)
print(json.dumps({'node':NODE,'container_id':d['Id'],'query_rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'supervisor_before':state,'supervisor_after':json.loads((root/'supervisor-status.json').read_text())}))
raise SystemExit(r.returncode)
'''
    return invoke(n,replace(code,{'NAME':name(n),'IMAGE':IMAGE,'RID':RID,'REMOTE':REMOTE,'INNER':inner,'NODE':n,'STAGED':STAGED}),timeout=240)

def live_meta_snapshot():
    code='''import pathlib,subprocess,json,hashlib,re
r=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=20);assert r.returncode==0
d=json.loads(r.stdout)[0];assert d['Image']==IMAGE and d['Name']=='/'+NAME and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID
dst=pathlib.Path(REMOTE)/'live-meta-memcache-logs';r=subprocess.run(['docker','cp',d['Id']+':/var/log/memcache_hybrid',str(dst)],capture_output=True,text=True,timeout=60)
out={'copy_rc':r.returncode,'stderr':r.stderr,'files':[]}
for p in dst.rglob('*'):
 if p.is_file() and p.stat().st_size<20000000:
  raw=p.read_bytes();rows=[x[:1000] for x in raw.decode(errors='replace').splitlines() if re.search('total.?size|capacity|mount|register|lease.?ttl|dram|evict_threshold',x,re.I)]
  out['files'].append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'matches':rows[-100:],'critical_lines_complete':[x for x in raw.decode(errors='replace').splitlines() if re.search(r'Mount bm on loc|Unmount|UnMount|ock\\.mmc\\.meta\\.lease_ttl_ms|ock\\.mmc\\.evict_threshold_',x)]})
print(json.dumps(out))
'''
    return invoke(34,replace(code,{'NAME':name(34,True),'IMAGE':IMAGE,'RID':RID,'REMOTE':REMOTE}),timeout=100)

def main():
    stage=json.loads((b.ROOT/'runs/20261018T-L10-target-graph-stage-v3/receipt.json').read_text());assert stage['pass'] and len(stage['stages'])==4 and all(x==0 for x in stage['stages'].values()) and stage['fixed_image_CPU_qualification']['pass']
    stopped=json.loads((b.ROOT/'runs/20261018T-L10-stop-entry-failed-v2-for-corrected-v3/receipt.json').read_text());assert len(stopped['nodes'])==4 and all(x==0 for x in stopped['nodes'].values()) and stopped['meta_rc']==0
    for node in [34,35,36,41]:
        closed=json.loads(json.loads((b.ROOT/('runs/20261018T-L10-stop-entry-failed-v2-for-corrected-v3/node-'+str(node)+'.json')).read_text())['stdout']);assert closed['resource_shutdown_complete'] and closed['provider_closed0'] and closed['all_NPUs_empty']
    qualified=json.loads((b.ROOT/'results/sglang-routing-metadata-CPU-qualification-v1.json').read_text());assert qualified['pass']
    signal_path=b.ROOT/'results/sdk-postinit-signal-qualification-v1.json'
    assert hashlib.sha256(signal_path.read_bytes()).hexdigest()=='cfe77966f53bf16811c9157d0a1165de4073f20938718dfb37f458cadf65dcf1'
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'candidate_total_bytes':568*1024**3,'classification':'SHARED_CONTAINER_POOL_ONLY_MODELS_NOT_STARTED','precheck':{},'providers':{},'native_preflight':{},'cleanup':{}}
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
        if keep:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fs={pool.submit(native_preflight,n):n for n in [34,35,36,41]}
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
