"""One NPU container, distinct native pool/model processes; model stops first."""
import argparse,hashlib,json,os,signal,subprocess,threading,time
from pathlib import Path

def supervise(config):
    assert config['project_id']=='glm53' and config['framework'] in ['sglang','vllm']
    root=Path(config['root']);assert str(root).startswith('/run/glm53') and root.is_dir()
    status=root/'supervisor-status.json';assert not status.exists()
    stop=threading.Event();start=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    signal.signal(signal.SIGINT,lambda *_:stop.set())
    signal.signal(signal.SIGUSR1,lambda *_:start.set())
    state={'project_id':'glm53','run_id':config['run_id'],'pid':os.getpid(),'phase':'PREPARING','events':[],'model_started':False}
    processes={};streams=[];timeout_used=False
    def emit(event,**fields):
        state.update(fields);state['phase']=event;state['events'].append({'event':event,'monotonic_ns':time.monotonic_ns(),**fields})
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(status)
        print('GLM53_SUPERVISOR='+json.dumps(state['events'][-1]),flush=True)
    def launch(role):
        for suffix in ['stdout','stderr']:streams.append((root/f'{role}.{suffix}.txt').open('x'))
        process=subprocess.Popen(config[f'{role}_argv'],stdout=streams[-2],stderr=streams[-1],env=os.environ|config.get(f'{role}_env',{}),start_new_session=True)
        processes[role]=process;emit(role.upper()+'_STARTED',**{role+'_pid':process.pid});return process
    def close(role,seconds):
        nonlocal timeout_used
        process=processes.get(role)
        if process is None:return
        def live_group_members():
            found=[]
            for path in Path('/proc').glob('[0-9]*/stat'):
                try:
                    fields=path.read_text().rpartition(')')[2].split()
                    if len(fields)>2 and int(fields[2])==process.pid and fields[0]!='Z':found.append(int(path.parent.name))
                except (OSError,ValueError):pass
            return found
        # Native parents may exit while their workers still own the process group.
        if process.poll() is None or live_group_members():
            emit(role.upper()+'_STOP_REQUESTED')
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            deadline=time.monotonic()+seconds
            while live_group_members() and time.monotonic()<deadline:time.sleep(.05)
            if live_group_members():
                timeout_used=True;emit(role.upper()+'_STOP_TIMEOUT')
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            process.wait(timeout=10)
        emit(role.upper()+'_CLOSED',**{role+'_exit_code':process.returncode})
    try:
        provider=launch('provider');deadline=time.monotonic()+config.get('provider_init_timeout_seconds',180)
        while not stop.is_set():
            if provider.poll() is not None:
                emit('PROVIDER_INIT_FAILED',provider_exit_code=provider.returncode);return 1
            path=root/'provider-status.json'
            if path.is_file():
                observed=json.loads(path.read_text())
                if observed.get('phase')=='READY':
                    assert observed['pid']==provider.pid
                    emit('POOL_READY_WAITING_MODEL_SIGNAL',provider_pid=provider.pid);break
            if time.monotonic()>deadline:raise TimeoutError('Native provider initialization deadline exceeded')
            stop.wait(.2)
        while not stop.is_set():
            if provider.poll() is not None:
                emit('PROVIDER_EXITED_UNEXPECTEDLY',provider_exit_code=provider.returncode);return 1
            if start.is_set() and not state['model_started']:
                state['model_started']=True;launch('model');emit('MODEL_RUNNING',model_started=True)
            model=processes.get('model')
            if model is not None and model.poll() is not None and 'model_exit_code' not in state:
                emit('MODEL_EXITED_POOL_RETAINED',model_exit_code=model.returncode)
            stop.wait(.2)
        emit('SHUTDOWN_REQUESTED')
    except BaseException as error:
        emit('SUPERVISOR_ERROR',error=repr(error));raise
    finally:
        # Keep the SDK contributor alive until native model shutdown has drained.
        close('model',config.get('model_stop_seconds',120))
        close('provider',config.get('provider_stop_seconds',60))
        emit('STOPPED',timeout_used=timeout_used)
        for stream in streams:stream.close()
    return 1 if timeout_used else 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
    raw=Path(a.config).read_bytes();assert hashlib.sha256(raw).hexdigest()==a.sha256
    raise SystemExit(supervise(json.loads(raw)))
if __name__=='__main__':main()
