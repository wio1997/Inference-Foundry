#!/usr/bin/env python3
"""Bounded real SDK1.3 PID1 signal test, 1GiB only; zero model calls."""
import ast,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();dep=P.parent/'probe_sglang_standalone_allocation_v1.py'
assert hashlib.sha256(dep.read_bytes()).hexdigest()=='8338fb6c8666d9a2ccdcfe0183a3dd14eb095fa8dc79e3ecbacbb7740405ea77'
s=importlib.util.spec_from_file_location('sdk_helpers',dep);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RID='20261017T1712-sdk-postinit-signal-probe-v1';RUN=m.b.ROOT/'runs'/RID;REMOTE='/data/wio/inference-foundry/glm53/runs/'+RID
PAYLOAD=r'''import pathlib,json,time,signal,threading,importlib.metadata,os
import memcache_hybrid as sdk
assert importlib.metadata.version('memcache_hybrid')=='1.3.0'
root=pathlib.Path('/run/glm53');events=[];stop=threading.Event();caught=[]
def emit(phase,**fields):
 events.append({'phase':phase,'monotonic_ns':time.monotonic_ns(),**fields});temp=root/'status.tmp';temp.write_text(json.dumps({'phase':phase,'pid':os.getpid(),'events':events},indent=2));temp.replace(root/'status.json');print('GLM53_POSTINIT_SIGNAL='+json.dumps(events[-1]),flush=True)
def handler(signum,_):caught.append(signum);stop.set()
for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,handler)
cfg=sdk.LocalConfig();cfg.meta_service_url='tcp://172.16.12.34:34880';cfg.config_store_url='tcp://172.16.12.34:34881';cfg.world_size=16;cfg.protocol='device_sdma';cfg.dram_size='1GB';cfg.max_dram_size='1GB';cfg.hbm_size='0GB';cfg.max_hbm_size='0GB';cfg.log_level='info'
store=sdk.DistributedObjectStore()
try:
 rc=store.setup(cfg);emit('SETUP',rc=rc);assert rc==0
 rc=store.init(0,init_bm=True);emit('INIT',rc=rc);assert rc==0
 # ResourceTracker installs native handlers when Store is constructed.
 for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,handler)
 emit('READY',handlers_restored_after_init=True)
 assert stop.wait(90),'Controller SIGTERM did not reach the restored Python handler'
 emit('APPLICATION_SIGNAL_CAUGHT',signals=caught);assert signal.SIGTERM in caught
finally:
 emit('CLOSE_START');rc=store.close();emit('CLOSE_COMPLETE',rc=rc);assert rc==0
'''
HOST=r'''import pathlib,subprocess,json,socket,time,gzip,hashlib
def cmd(a,t=30):
 r=subprocess.run(a,capture_output=True,text=True,timeout=t);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
root=pathlib.Path(REMOTE);assert not root.exists();root.mkdir(parents=True)
assert cmd(['docker','image','inspect',IMAGE,'--format','{{.Id}}'])['stdout'].strip()==IMAGE
before=cmd(['npu-smi','info']);assert before['rc']==0 and before['stdout'].count('No running processes')==8
for port in [34880,34881,34882]:
 sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
(root/'payload.py').write_text(PAYLOAD)
(root/'meta.conf').write_text('ock.mmc.meta_service_url = tcp://172.16.12.34:34880\nock.mmc.meta_service.config_store_url = tcp://172.16.12.34:34881\nock.mmc.meta_service.metrics_url = http://172.16.12.34:34882\nock.mmc.log_level = info\nock.mmc.meta.lease_ttl_ms = 60000\nock.mmc.evict_threshold_high = 99\nock.mmc.evict_threshold_low = 98\n')
out={'model_requests':0,'SDK_dram_bytes':1073741824,'npu_before':before,'started':{},'cleanup':{}}
def verify(name):
 r=cmd(['docker','inspect',name]);assert r['rc']==0;d=json.loads(r['stdout'])[0];assert d['Name']=='/'+name and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==RID;return d
try:
 for name,args in [(META,META_ARGS),(ACTOR,ACTOR_ARGS)]:
  r=cmd(args,40);out['started'][name]=r;assert r['rc']==0
  if name==META:
   deadline=time.monotonic()+30
   while True:
    try:
     with socket.create_connection(('172.16.12.34',34880),timeout=1):break
    except OSError:
     assert time.monotonic()<deadline;time.sleep(.2)
 deadline=time.monotonic()+60
 while True:
  p=root/'status.json'
  if p.is_file() and json.loads(p.read_text())['phase']=='READY':break
  assert verify(ACTOR)['State']['Running'] and time.monotonic()<deadline;time.sleep(.2)
 out['signal']=cmd(['docker','kill','--signal=TERM',verify(ACTOR)['Id']]);assert out['signal']['rc']==0
 deadline=time.monotonic()+30
 while verify(ACTOR)['State']['Running'] and time.monotonic()<deadline:time.sleep(.2)
 out['actor_after']=verify(ACTOR)['State'];out['application_status']=json.loads((root/'status.json').read_text())
 out['pass']=bool(not out['actor_after']['Running'] and out['actor_after']['ExitCode']==0 and out['application_status']['phase']=='CLOSE_COMPLETE' and out['application_status']['events'][-1]['rc']==0 and any(e['phase']=='APPLICATION_SIGNAL_CAUGHT' for e in out['application_status']['events']))
finally:
 for name in [ACTOR,META]:
  exists=cmd(['docker','inspect',name])
  if exists['rc']!=0:continue
  d=verify(name)
  if d['State']['Running']:out['cleanup'][name]=cmd(['docker','stop','--time','10',d['Id']],25)
  logs=cmd(['docker','logs',d['Id']],40);out[name+'_state']=verify(name)['State'];out[name+'_logs']={}
  for stream in ['stdout','stderr']:
   raw=logs[stream].encode();p=root/(name+'-'+stream+'.gz');p.write_bytes(gzip.compress(raw));out[name+'_logs'][stream]={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'tail':raw[-18000:].decode(errors='replace')}
 out['npu_after']=cmd(['npu-smi','info'])
 (root/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
assert out.get('pass') and out['npu_after']['stdout'].count('No running processes')==8
'''
def main():
    done=json.loads((m.b.ROOT/'runs/20261017T1708-finish-sglang-meta-stop-v1/receipt.json').read_text());assert done['meta_stop_rc']==0
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    meta='glm53-sdk-postinit-signal-v1-meta-20261017-34';actor='glm53-sdk-postinit-signal-v1-actor-20261017-34'
    common=['docker','run','-d','--pull=never','--network=host','--pids-limit=512','--memory=4g','--cpus=2','--label','project_id=glm53','--label','run_id='+RID,'-v',REMOTE+':/run/glm53','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1']
    meta_args=common+['--name',meta,'-e','MMC_META_CONFIG_PATH=/run/glm53/meta.conf','--entrypoint','python3',m.IMAGE,'-c','import memcache_hybrid as mc,sys;sys.exit(mc.MetaService.main())']
    actor_args=common+['--name',actor,'--ulimit','memlock=-1','-e','ASCEND_RT_VISIBLE_DEVICES=0','-e','ACL_OP_INIT_MODE=1','-e','ASCEND_ENABLE_USE_FABRIC_MEM=1']
    for dev in ['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:actor_args+=['--device',dev]
    actor_args+=['--entrypoint','python3',m.IMAGE,'/run/glm53/payload.py']
    code=m.replace(HOST,{'REMOTE':REMOTE,'IMAGE':m.IMAGE,'RID':RID,'META':meta,'ACTOR':actor,'META_ARGS':meta_args,'ACTOR_ARGS':actor_args,'PAYLOAD':PAYLOAD});ast.parse(code)
    r=m.invoke(34,code,timeout=280);(RUN/'node-34.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0,'SDK_allocated_candidate_bytes':1073741824},indent=2)+'\n')
if __name__=='__main__':main()
