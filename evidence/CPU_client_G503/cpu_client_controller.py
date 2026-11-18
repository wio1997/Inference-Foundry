"""Owned four-core CPU client lease; no NPU devices or serving endpoints."""
from pathlib import Path
import datetime,fcntl,json,os,signal,socket,subprocess,time

ROOT=Path('/PRIVATE_TASK_ROOT')
RUN=ROOT/'client_ds41_g503'
OLD=Path('/PRIVATE_TASK_ROOT')
IMAGE='sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
NAME='yxy-ds41-aisbench-g503'
CID=None

def atomic(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)

def command(argv,timeout=30):
    return subprocess.check_output(argv,cwd=RUN,text=True,stderr=subprocess.STDOUT,timeout=timeout)

def state(phase,**kw):
    atomic(RUN/'controller_state.json',dict(phase=phase,pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))

def main():
    global CID
    assert socket.gethostname()=='<ROLE_HOSTNAME>' and Path(__file__).resolve().parent==RUN
    coord=Path('/root/.codex/coord');lock=(coord/'<ROLE_HOSTNAME>-cpu184-187.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=coord/'<ROLE_HOSTNAME>-cpu184-187.owner.json'
    prior=json.loads(owner.read_text()) if owner.exists() else {}
    assert not prior or prior['state']=='released','Existing CPU owner must be reconciled'
    generation=int(prior.get('generation',0))+1
    claim=dict(state='claimed',pid=os.getpid(),generation=generation,task_id=ROOT.name,run_directory=str(RUN))
    atomic(owner,claim)
    def stop(*_):raise InterruptedError('Owned CPU client controller stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        state('claimed',cpuset='184-187',cpus=4,memory_bytes=34359738368,NPU_devices=[],lifetime_s=72000)
        exists=subprocess.run(['docker','inspect',NAME],cwd=RUN,capture_output=True,text=True,timeout=20)
        assert exists.returncode and 'No such' in exists.stderr,'Existing new client must be reconciled'
        old=json.loads(command(['docker','inspect','yxy-ds41-aisbench-g83']))[0]
        assert old['Id']=='341aff6cccf17b09718cf4c787b264aee85117a4694a63c299684823d9309fe6' and old['Image']==IMAGE
        assert old['Config']['Labels']['codex.task']==ROOT.name and old['Config']['Labels']['codex.run']=='client_ds41_g83'
        if old['State']['Running']:
            top=command(['docker','top',old['Id'],'-eo','pid,args']).splitlines()[1:]
            assert len(top)==1 and top[0].split(None,1)[1]=='sleep 72000','Old client has active work; preserve it'
            command(['docker','stop','--time','5',old['Id']],20)
        assert not json.loads(command(['docker','inspect',old['Id']]))[0]['State']['Running']
        for part in ['home','tmp','shm']:(RUN/part).mkdir(exist_ok=True)
        args=['docker','run','-d','--name',NAME,'--label','codex.task='+ROOT.name,'--label','codex.run='+RUN.name,'--label','codex.controller_pid='+str(os.getpid()),'--network','host','--log-driver','none','--read-only','--cpuset-cpus','184-187','--cpus','4','--memory','32g','-v',str(ROOT)+':/task:ro','-v',str(RUN)+':/task/'+RUN.name,'-v',str(OLD/'client_ds41_g83/datasets')+':/task/'+RUN.name+'/datasets:ro','-v',str(OLD/'models/DeepSeek-V4.1-Flash-1017-w4a8')+':/models/ds41:ro']
        for part,dest in [('home','/root'),('tmp','/tmp'),('shm','/dev/shm')]:args+=['-v',str(RUN/part)+':'+dest]
        for key,value in {'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','TMPDIR':'/task/'+RUN.name+'/tmp','HF_HOME':'/task/'+RUN.name+'/home/hf'}.items():args+=['-e',key+'='+value]
        args+=['-w','/task/'+RUN.name,'--entrypoint','sleep',IMAGE,'72000']
        CID=command(args,60).strip()
        info=json.loads(command(['docker','inspect',CID]))[0]
        assert info['Id']==CID and info['Image']==IMAGE and info['State']['Running']
        assert not info['HostConfig']['Devices'] and not info['HostConfig']['Privileged'] and info['HostConfig']['Memory']==34359738368
        atomic(RUN/'create_receipt.json',dict(container=NAME,container_id=CID,image_id=IMAGE,cpuset='184-187',NPU_devices=[],mounts=info['Mounts'],old_idle_client_retired=old['Id'],at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with (RUN/'asset_revalidation.log').open('x') as log:
            proc=subprocess.Popen(['docker','exec',NAME,'python3','-u','/task/'+RUN.name+'/validate_existing_assets.py'],cwd=RUN,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state('asset_revalidation_running',container_id=CID,asset_job_pid=proc.pid)
            assert proc.wait(timeout=1860)==0,'CPU asset validation failed; no model requests sent'
        state('ready_for_separately_gated_benchmark',container_id=CID,asset_state=json.loads((RUN/'assets_state.json').read_text()))
        deadline=time.monotonic()+70000
        while not (RUN/'stop.request').exists():
            assert time.monotonic()<deadline,'CPU client lifecycle bound'
            assert json.loads(command(['docker','inspect',CID]))[0]['State']['Running'],'CPU client exited'
            time.sleep(15)
    except BaseException as exc:
        state('terminal',error=repr(exc));raise
    finally:
        if CID is None:
            result=subprocess.run(['docker','inspect',NAME],cwd=RUN,capture_output=True,text=True,timeout=20)
            if result.returncode==0:
                item=json.loads(result.stdout)[0]
                assert item['Config']['Labels'].get('codex.controller_pid')==str(os.getpid()),'Unknown client identity: retain CPU owner record'
                CID=item['Id']
        if CID:
            info=json.loads(command(['docker','inspect',CID]))[0]
            assert info['Config']['Labels']['codex.controller_pid']==str(os.getpid()) and info['Config']['Labels']['codex.run']==RUN.name
            if info['State']['Running']:command(['docker','stop','--time','10',CID],30)
            assert not json.loads(command(['docker','inspect',CID]))[0]['State']['Running']
        current=json.loads(owner.read_text());assert current['pid']==os.getpid() and current['generation']==generation
        current['state']='released';atomic(owner,current);lock.close()

if __name__=='__main__':main()
