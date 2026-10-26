"""Linux child/signal integration fixture; fake processes, no NPU/SDK/model."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/run/glm53')
FAKE=r'''import json,os,pathlib,signal,subprocess,sys,time
role,mode,directory=sys.argv[1:];root=pathlib.Path(directory)
def event(name):
 fd=os.open(root/'order.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
 os.write(fd,(json.dumps({'event':name,'pid':os.getpid()})+'\n').encode());os.close(fd)
def stop(*_):
 time.sleep(.05);event(role+'_STOP');raise SystemExit(0)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
event(role+'_START')
if role=='provider':
 if mode=='init_failure':raise SystemExit(17)
 temp=root/'provider-status.tmp';temp.write_text(json.dumps({'phase':'READY','pid':os.getpid()}));temp.replace(root/'provider-status.json')
if role=='model' and mode=='orphan_worker':
 subprocess.Popen([sys.executable,__file__,'grandchild',mode,directory])
 deadline=time.monotonic()+3
 while 'grandchild_START' not in (root/'order.jsonl').read_text():
  assert time.monotonic()<deadline;time.sleep(.01)
 event('model_PARENT_EXIT');raise SystemExit(0)
while True:signal.pause()
'''

def wait(predicate,timeout=8):
    deadline=time.monotonic()+timeout
    while not predicate():
        if time.monotonic()>deadline:raise TimeoutError('Fixture state did not arrive')
        time.sleep(.02)

def main():
    fake=ROOT/'fake-child.py';fake.write_text(FAKE);cases=[]
    for mode in ['normal','orphan_worker','init_failure']:
        root=ROOT/mode;root.mkdir()
        cfg={'project_id':'glm53','framework':'sglang','run_id':'fake-'+mode,'root':str(root),'provider_argv':[sys.executable,str(fake),'provider',mode,str(root)],'model_argv':[sys.executable,str(fake),'model',mode,str(root)],'provider_init_timeout_seconds':5,'model_stop_seconds':2,'provider_stop_seconds':2}
        path=root/'config.json';raw=(json.dumps(cfg)+'\n').encode();path.write_bytes(raw)
        with (root/'supervisor.stdout').open('w') as out,(root/'supervisor.stderr').open('w') as err:
            process=subprocess.Popen([sys.executable,str(ROOT/'glm53_pool_model_supervisor_v1.py'),'--config',str(path),'--sha256',hashlib.sha256(raw).hexdigest()],stdout=out,stderr=err,start_new_session=True)
            try:
                if mode=='init_failure':
                    assert process.wait(timeout=8)==1
                    status=json.loads((root/'supervisor-status.json').read_text())
                    assert not status['model_started'] and any(e['event']=='PROVIDER_INIT_FAILED' for e in status['events'])
                    cases.append({'mode':mode,'no_model_start':True,'provider_exit':status['provider_exit_code']});continue
                wait(lambda:(root/'supervisor-status.json').is_file() and json.loads((root/'supervisor-status.json').read_text())['phase']=='POOL_READY_WAITING_MODEL_SIGNAL')
                assert 'model_START' not in (root/'order.jsonl').read_text()
                os.kill(process.pid,signal.SIGUSR1);wait(lambda:'model_START' in (root/'order.jsonl').read_text());os.kill(process.pid,signal.SIGUSR1)
                if mode=='orphan_worker':wait(lambda:'model_exit_code' in json.loads((root/'supervisor-status.json').read_text()))
                time.sleep(.05);assert (root/'order.jsonl').read_text().count('"model_START"')==1
                os.kill(process.pid,signal.SIGTERM);assert process.wait(timeout=8)==0
                order=[json.loads(line)['event'] for line in (root/'order.jsonl').read_text().splitlines()]
                terminal='grandchild_STOP' if mode=='orphan_worker' else 'model_STOP'
                assert order.index(terminal)<order.index('provider_STOP'),order
                status=json.loads((root/'supervisor-status.json').read_text());assert not status['timeout_used'] and status['provider_exit_code']==0 and status['model_exit_code']==0
                cases.append({'mode':mode,'order':order,'model_before_provider':True,'no_duplicate_model_start':True,'timeout_used':False})
            finally:
                if process.poll() is None:
                    os.kill(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=8)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
    result={'project_id':'glm53','pass':True,'model_requests':0,'SDK_init_calls':0,'NPU_devices':False,'cases':cases,'scope':'Linux subprocess/process-group lifecycle only; no native SDK/model qualification.'}
    (ROOT/'fixture-result.json').write_text(json.dumps(result,indent=2)+'\n');print('GLM53_SUPERVISOR_FIXTURE='+json.dumps(result),flush=True)
if __name__=='__main__':main()
