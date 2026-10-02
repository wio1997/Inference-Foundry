"""Single owner for GLM task resources; immutable stages, no unknown-stage replay."""
import argparse, fcntl, hashlib, json, os, signal, socket, subprocess, sys, time
from pathlib import Path
from phase_runner import atomic_json, process_identity, same_process, run_phase, PhaseExecutionError, utc
LOCK_ROOT=Path('/data/tiankuan/wio/glm52-pd')
def work(path):
    path=Path(path).resolve(); root=path.parent; raw=path.read_bytes(); spec=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    locks=[]
    for name in ('.controller.lock','.formal-test.lock'):
        lock=(LOCK_ROOT/name).open('a+'); locks.append(lock)
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return 82
    state_path=root/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'run_id':spec['run_id'],'spec_sha256':digest,'completed_stages':[],'active_stage':None,'status':'starting'}
    if state['spec_sha256']!=digest: raise ValueError('immutable spec changed')
    if state['status'] in ('completed','failed','cancelled','needs_reconciliation'): return 83
    # Reject malformed/falsely pinned new specs before publishing a running owner.
    for stage in spec['stages']:
        if not stage['id'].isalnum():raise ValueError('invalid stage id')
        for source in stage.get('sources',[]):
            if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:raise ValueError('source identity changed')
    state['owner']={'host':socket.gethostname(),**process_identity(os.getpid())}
    state['status']='running'; last=0; cancelled=[]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda signum,_frame:cancelled.append(signum))
    def beat(force=False):
        nonlocal last
        if force or time.monotonic()-last>=5:
            state['heartbeat_at']=utc(); atomic_json(state_path,state)
            atomic_json(LOCK_ROOT/'controller-owner.json',{'run_id':spec['run_id'],'state_path':str(state_path),'owner':state['owner'],'status':state['status'],'heartbeat_at':state['heartbeat_at']});last=time.monotonic()
    beat(True)
    if state['active_stage']:
        phase=root/(state['active_stage']+'.phase.json'); rec=json.loads(phase.read_text()) if phase.exists() else None
        stage=next(s for s in spec['stages'] if s['id']==state['active_stage'])
        if rec and rec['run_id']==spec['run_id'] and rec['argv']==stage['argv'] and rec['status']=='succeeded' and rec['exit_code']==0:
            state['completed_stages'].append(stage['id']);state['active_stage']=None
        else:
            state['status']='needs_reconciliation';state['unresolved_child']=rec.get('child') if rec else None;beat(True);return 84
    for stage in spec['stages']:
        if stage['id'] in state['completed_stages']:continue
        if cancelled:state['status']='cancelled';beat(True);return 130
        try:
            for source in stage.get('sources',[]):
                if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:raise ValueError('source identity changed')
        except (OSError,ValueError) as error:
            state['status']='failed';state['failure_phase']=stage['id'];state['failure_kind']='source_validation';state['error']=str(error);beat(True);return 1
        state['active_stage']=stage['id'];beat(True)
        try:
            run_phase(stage['argv'],phase=stage['id'],run_id=spec['run_id'],cwd=stage.get('cwd',str(root)),record_path=root/(stage['id']+'.phase.json'),log_path=root/(stage['id']+'.log'),timeout_s=stage.get('timeout_s'),heartbeat=beat,cancel_requested=lambda:bool(cancelled),echo=False)
        except PhaseExecutionError as e:
            state['status']='cancelled' if e.record['cancelled'] else 'failed';state['failure_phase']=stage['id'];beat(True);return 1
        state['completed_stages'].append(stage['id']);state['active_stage']=None;beat(True)
    state['status']='completed';state['finished_at']=utc();beat(True);return 0

def start(path):
    root=Path(path).resolve().parent
    with (root/'controller.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'work',str(Path(path).resolve())],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(100):
        state_path=root/'state.json'
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state.get('owner',{}).get('pid')==p.pid:print(json.dumps(state));return 0
        if p.poll() is not None:return p.returncode
        time.sleep(.05)
    return 85
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','work']);p.add_argument('spec');a=p.parse_args();sys.exit({'start':start,'work':work}[a.action](a.spec))
