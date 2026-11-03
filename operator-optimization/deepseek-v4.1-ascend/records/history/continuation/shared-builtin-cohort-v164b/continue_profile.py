"""Complete the unstarted formal pass under the user's broadened container authorization.

Reuses the valid reference. Preserves the first terminal controller and its
failure. The original acquisition deadline is not reset.
"""
import sys,json,os,time,subprocess,traceback,signal
from pathlib import Path
ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT))
import controller as c

def main():
    old=json.loads((ROOT/'STATUS.json').read_text())
    assert old.get('finished_unix') and old['restore']=='OWN_ORIGINAL_SERVICE_HEALTH_200'
    assert old['baseline_case_valid'] and old['profile_attempts']==0
    prep=json.loads((ROOT/'control/PREPARED.json').read_text());diag=prep['diagnostic_container_id']
    status=ROOT/'CORRECTION-STATUS.json';assert not status.exists(),'Poll same corrected handle;never resubmit'
    fd=os.open(status,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    state={'state':'GUARD_UNSTARTED_FORMAL_PASS','pid':os.getpid(),'proc_start_ticks':c.proc_start(os.getpid()),
        'started_unix':time.time(),'original_acquisition_deadline_unix':old['started_unix']+2700,
        'reference_reused':True,'reference_attempts':0,'profile_attempts':0,
        'first_terminal_controller':str(ROOT/'STATUS.json'),'overall_complete':False}
    def save():
        q=ROOT/'CORRECTION-STATUS.tmp';q.write_text(json.dumps(state,ensure_ascii=False,indent=2));q.replace(status)
    stopped=False;started=False;child=None
    def guard():
        state['output_bytes']=c.disk_bytes(ROOT);save()
        assert state['output_bytes']<=16*1024**3,'Directory stop threshold exceeded'
        assert time.time()<state['original_acquisition_deadline_unix'],'Original acquisition deadline reached'
    def interrupted(*a):raise RuntimeError('Owned corrected controller interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save()
    try:
        assert time.time()+600<state['original_acquisition_deadline_unix'],'Insufficient original window remaining'
        assert json.loads((ROOT/'control/APPROVAL.json').read_text())['container_process_authorization_broadened']
        for row in prep['execution_pins']:
            import hashlib
            b=Path(row['path']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],row['path']
        assert c.inspect(c.MAIN)['State']['Running'] and not c.inspect(diag)['State']['Running']
        c.idle_metrics(c.request(30075,'/metrics')[1].decode());pids,text=c.busy();c.owns_processes(pids,c.owned(c.MAIN))
        stopped=True;c.command(['docker','stop','--time','60',c.MAIN],90)
        end=time.monotonic()+60
        while c.busy()[0]:
            assert time.monotonic()<end,'NPU pool not idle';time.sleep(1)
        started=True;c.command(['docker','start',diag],60);state['state']='CORRECTED_DIAGNOSTIC_LOADING';save()
        while True:
            guard();assert c.inspect(diag)['State']['Running'],'Corrected diagnostic exited during load'
            try:
                if c.request(30076,'/health')[0]==200:break
            except Exception:pass
            time.sleep(3)
        c.idle_metrics(c.request(30076,'/metrics')[1].decode());pids,text=c.busy();c.owns_processes(pids,c.owned(diag))
        env=dict(os.environ);env.update(WIO_PHASE='profile',WIO_OUTPUT=str(ROOT/'outputs/profile'),
            WIO_FROZEN_ROOT=str(ROOT/'runtime/benchmark-20261028/workload/frozen'),WIO_API_URL='http://127.0.0.1:30076')
        state.update(state='RUNNING_ONE_FORMAL_PROFILE_PASS',profile_attempts=1);save()
        with (ROOT/'profile-stdout.log').open('xb') as out,(ROOT/'profile-stderr.log').open('xb') as err:
            child=subprocess.Popen(['python3','-I','-S',str(ROOT/'kit/client.py')],env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            state['client_pid']=child.pid;save();end=time.monotonic()+900
            while child.poll() is None:
                guard();assert time.monotonic()<end,'Full formal client deadline reached';time.sleep(1)
            rc=child.returncode;child=None;assert rc==0,rc
        case=json.loads((ROOT/'outputs/profile/window-status.json').read_text());assert case['valid']
        state['profile_case_valid']=True;state['profile_windows']=case.get('windows',[])
        (ROOT/'control/FINALIZE_METADATA').write_text('Full formal client returned.\n')
        assert c.request(30076,'/stop_profile',{},90)[0]==200
        state['state']='FORMAL_PROFILE_FINISHED_RAW_REVIEW_PENDING';save()
    except BaseException:state.update(state='CORRECTED_PASS_FAILED',error=traceback.format_exc());save()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        try:
            if started and c.inspect(diag)['State']['Running']:c.command(['docker','stop','--time','10',diag],45)
        except BaseException:state['diagnostic_stop_error']=traceback.format_exc()
        try:
            if stopped:
                end=time.monotonic()+60
                while c.busy()[0] and time.monotonic()<end:time.sleep(1)
                pids,text=c.busy()
                if pids:state.update(restore='DEFERRED_OCCUPIED_NPU',restore_busy_pids=sorted(pids))
                else:
                    c.command(['python3',str(c.MIGRATION/'environment_control171.py'),'start'],150)
                    state['restore']='LOADING_OWN_ORIGINAL_SERVICE';save();end=time.monotonic()+2400
                    while True:
                        assert c.inspect(c.MAIN)['State']['Running']
                        try:
                            if c.request(30075,'/health')[0]==200:break
                        except Exception:pass
                        assert time.monotonic()<end,'Restore deadline reached';time.sleep(3)
                    state['restore']='OWN_ORIGINAL_SERVICE_HEALTH_200'
            else:state['restore']='NOT_REQUIRED_ORIGINAL_NOT_STOPPED'
        except BaseException:state['restore_error']=traceback.format_exc()
        state['finished_unix']=time.time();save();print(json.dumps(state),flush=True)
if __name__=='__main__':main()
