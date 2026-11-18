"""Refresh idle four-core AISBench lease before a new separately gated campaign."""
from pathlib import Path
import datetime,fcntl,json,os,signal,socket,subprocess,time,shutil
ROOT=Path('<TASK_ROOT>');R=ROOT/'client_glm53_g576'
IMAGE='sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76';OLD_CID='0a2d37de7eb21cf1e07b04ab2c1dee7a18d15a149bccf2a6cdfbce92bc3067a9';NAME='yxy-glm53-aisbench-g576';CID=None;CLAIMED=False

def save(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def state(phase,**kw):
 d={'phase':phase,'pid':os.getpid(),'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw};save(R/'state.json',d);print(json.dumps(d),flush=True)
def run(argv,timeout=30):return subprocess.check_output(argv,cwd=R,text=True,stderr=subprocess.STDOUT,timeout=timeout)
def stop(*_):raise InterruptedError('Owned CPU lease stop')
assert socket.gethostname()=='<ROLE_NODE>' and Path(__file__).resolve().parent==R
os.chdir(R);coord=Path('<COORD_ROOT>');lock=(coord/'<ROLE_NODE>-cpu20-44-68-92.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owner=coord/'<ROLE_NODE>-cpu20-44-68-92.owner.json';previous=json.loads(owner.read_text()) if owner.exists() else {};assert not previous or previous['state']=='released';generation=int(previous.get('generation',0))+1
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
state('preflight')
try:
 mounts={p:shutil.disk_usage(p).free for p in ['/data','/data1']};assert max(mounts,key=mounts.get)=='/data1';save(R/'storage_preflight.json',mounts)
 old=json.loads(run(['docker','inspect',OLD_CID]))[0];assert old['Id']==OLD_CID and old['Image']==IMAGE and old['Config']['Labels']['codex.task']==ROOT.name and old['Config']['Labels']['codex.run']=='client_vllm_g42'
 assert old['State']['Running'] and old['Config']['Cmd']==['72000'] and old['Config']['Entrypoint']==['sleep']
 top=run(['docker','top',OLD_CID,'-eo','pid,args']);rows=top.splitlines()[1:];assert len(rows)==1 and rows[0].split(None,1)[1]=='sleep 72000','Existing CPU work must finish before refresh'
 failed=json.loads((ROOT/'client_vllm_g42/runs/sglang/SG_G316_C16_256K_g404/state.json').read_text());assert failed['phase']=='failed_requires_review'
 h=json.loads((ROOT/'client_vllm_g42/SG_G316_256K_first_formal_g514_handle.json').read_text());p=Path('/proc',str(h['pid']),'cmdline');assert not p.exists() or b'run_aisbench_cold_hot.py' not in p.read_bytes()
 q=subprocess.run(['docker','inspect',NAME],cwd=R,capture_output=True,text=True,timeout=20);assert q.returncode and 'No such' in q.stderr
 save(R/'old_idle_snapshot.json',{'old_id':OLD_CID,'top':top,'image':IMAGE,'old_started_at':old['State']['StartedAt'],'old_mounts':old['Mounts'],'scope':'Oldfailedfirstanddatasetartifacts retained; idle CPU sleep only'})
 claim={'state':'claimed','pid':os.getpid(),'generation':generation,'task_id':ROOT.name,'run_directory':str(R)};save(owner,claim);CLAIMED=True
 run(['docker','stop','--time','5',OLD_CID],20);assert not json.loads(run(['docker','inspect',OLD_CID]))[0]['State']['Running']
 for part in ['home','tmp','shm']:(R/part).mkdir(exist_ok=True)
 args=['docker','run','-d','--name',NAME,'--label','codex.task='+ROOT.name,'--label','codex.run='+R.name,'--label','codex.controller_pid='+str(os.getpid()),'--network','host','--log-driver','none','--read-only','--cpuset-cpus','20,44,68,92','--cpus','4','--memory','32g','-v',str(ROOT)+':/task:ro','-v',str(R)+':/task/'+R.name,'-v',str(ROOT/'client_vllm_g42/datasets')+':/prior_datasets:ro','-v','/data/models/GLM-5.3-Flash-w8a8:/models/glm53:ro']
 for part,dest in [('home','/root'),('tmp','/tmp'),('shm','/dev/shm')]:args+=['-v',str(R/part)+':'+dest]
 for k,v in {'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','TMPDIR':'/task/'+R.name+'/tmp','HF_HOME':'/task/'+R.name+'/home/hf'}.items():args+=['-e',k+'='+v]
 args+=['-w','/task/'+R.name,'--entrypoint','sleep',IMAGE,'72000'];save(R/'container_command.json',args);CID=run(args,60).strip();a=json.loads(run(['docker','inspect',CID]))[0]
 assert a['Id']==CID and a['Image']==IMAGE and a['State']['Running'] and not a['HostConfig']['Devices'] and not a['HostConfig']['Privileged'] and a['HostConfig']['Memory']==34359738368 and a['HostConfig']['CpusetCpus']=='20,44,68,92'
 save(R/'create_receipt.json',{'container_id':CID,'name':NAME,'image':IMAGE,'started_at':a['State']['StartedAt'],'NPU_devices':[],'cpus':4,'cpuset':'20,44,68,92','memory_bytes':34359738368,'mounts':a['Mounts'],'old_client_retired_idle':OLD_CID,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPUleaseonly; newnonceassets/usagecontract/campaigngates notprepared; no AISBench/modelrequests'})
 state('CPU_ready_assets_and_campaign_UNVERIFIED',container_id=CID,lifetime_s=72000,no_model_requests=True)
 deadline=time.monotonic()+71900
 while not (R/'stop.request').exists():
  assert time.monotonic()<deadline,'CPUleasebound';assert json.loads(run(['docker','inspect',CID]))[0]['State']['Running'];time.sleep(15)
except BaseException as exc:
 state('failed_requires_review',error=repr(exc));raise
finally:
 if CID is None:
  check=subprocess.run(['docker','inspect',NAME],cwd=R,capture_output=True,text=True,timeout=20)
  if check.returncode==0:
   item=json.loads(check.stdout)[0]
   assert item['Config']['Labels'].get('codex.controller_pid')==str(os.getpid()) and item['Config']['Labels'].get('codex.run')==R.name,'Unknown CPU container identity: ownership retained'
   CID=item['Id']
  else:assert 'No such' in check.stderr,'Docker identity cannot be reconciled; ownership retained'
 if CID:
  a=json.loads(run(['docker','inspect',CID]))[0];assert a['Config']['Labels']['codex.controller_pid']==str(os.getpid()) and a['Config']['Labels']['codex.run']==R.name
  if a['State']['Running']:run(['docker','stop','--time','10',CID],30)
  assert not json.loads(run(['docker','inspect',CID]))[0]['State']['Running']
 if CLAIMED:
  claim=json.loads(owner.read_text());assert claim['pid']==os.getpid() and claim['generation']==generation;claim['state']='released';save(owner,claim);state('CPU_stopped_released')
 lock.close()
