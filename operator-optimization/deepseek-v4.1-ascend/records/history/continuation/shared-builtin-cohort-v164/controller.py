"""One explicitly approved complete171 C8 reference/profile cohort and restore.

Default --describe performs no server/container/model mutations.
"""
import argparse,hashlib,json,os,re,signal,socket,stat,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cohort_policy import approved,owns_processes,restore_allowed,no_existing_submission,idle_metrics

MAIN='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb'
IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
MIGRATION=Path('/data/tiankuan/wio/migration-171-20261102')
def command(argv,timeout=30):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError({'argv_prefix':argv[:3],'rc':p.returncode,'stdout':p.stdout[-1600:],'stderr':p.stderr[-1600:]})
    return p.stdout
def inspect(cid):
    d=json.loads(command(['docker','inspect',cid]))[0]
    if d['Id']!=cid or d['Image']!=IMAGE:raise RuntimeError('Container/image identity changed')
    return d
def request(port,path,body=None,timeout=5):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    raw=None if body is None else json.dumps(body).encode()
    with opener.open(urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=raw,
        headers={'Content-Type':'application/json'}),timeout=timeout) as r:return r.status,r.read()
def busy():
    text=command(['npu-smi','info']);pids=set()
    for line in text.splitlines():
        f=[x.strip() for x in line.split('|')]
        if len(f)>4 and re.fullmatch(r'\d+\s+\d+',f[1]) and f[2].isdigit():pids.add(int(f[2]))
    return pids,text
def owned(cid):return {int(x.strip()) for x in command(['docker','top',cid,'-eo','pid']).splitlines() if x.strip().isdigit()}
def disk_bytes(root):
    total=0
    for here,dirs,files in os.walk(root,followlinks=False):
        for name in files:
            try:s=(Path(here)/name).lstat()
            except FileNotFoundError:continue
            if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total
def proc_start(pid):
    try:return Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None
def execute(root):
    if socket.gethostname()!='bm-jn-zs-zone1-910C-64G-10-171' or os.geteuid()!=0:raise RuntimeError('Wrong host/account')
    control=root/'control';plan_bytes=(control/'PLAN.json').read_bytes();prepared_bytes=(control/'PREPARED.json').read_bytes()
    plan=approved(plan_bytes,prepared_bytes,json.loads((control/'APPROVAL.json').read_text()))
    prepared=json.loads(prepared_bytes);diag=prepared['diagnostic_container_id'];status=root/'STATUS.json'
    no_existing_submission(status)
    # O_EXCL is the one-submission guard, before any workload or container action.
    fd=os.open(status,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    state={'state':'GUARD','pid':os.getpid(),'proc_start_ticks':proc_start(os.getpid()),'run_id':plan['run_id'],
        'approval_id':plan['approval_slot_id'],'main_container':MAIN,'diagnostic_container':diag,
        'reference_attempts':0,'profile_attempts':0,'started_unix':time.time(),
        'overall_complete':False,'new_historical_native_promotions':0}
    def save():
        p=root/'STATUS.tmp';p.write_text(json.dumps(state,ensure_ascii=False,indent=2));p.replace(status)
    main_stopped=False;diag_started=False;child=None;stage_deadline=time.monotonic()+plan['limits']['acquisition_seconds']
    def monitor():
        current=disk_bytes(root);state['observed_output_bytes']=current
        if current>plan['limits']['directory_stop_threshold_bytes']:
            raise RuntimeError('Diagnostic directory exceeded stop threshold;polling overshoot retained,not hidden')
        if time.monotonic()>stage_deadline:raise RuntimeError('Acquisition deadline reached')
        save()
    def run_client(phase,port,timeout):
        nonlocal child
        env=dict(os.environ);env.update(WIO_PHASE=phase,WIO_OUTPUT=str(root/'outputs'/phase),
            WIO_FROZEN_ROOT=str(root/'runtime/benchmark-20261028/workload/frozen'),WIO_API_URL='http://127.0.0.1:'+str(port))
        state['state']='RUNNING_'+phase.upper();state[('reference' if phase=='baseline' else 'profile')+'_attempts']+=1;save()
        with (root/(phase+'-stdout.log')).open('xb') as out,(root/(phase+'-stderr.log')).open('xb') as err:
            child=subprocess.Popen(['python3','-I','-S',str(root/'kit/client.py')],env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
            state['client_pid']=child.pid;state['client_start_ticks']=proc_start(child.pid);save();deadline=time.monotonic()+timeout
            while child.poll() is None:
                monitor()
                if time.monotonic()>deadline:raise RuntimeError('Full fixed-case client deadline reached')
                time.sleep(1)
            rc=child.returncode;child=None
        if rc:raise RuntimeError('One full-case client returned '+str(rc)+';no automatic retry')
        result=json.loads((root/'outputs'/phase/'window-status.json').read_text())
        if not result.get('valid'):raise RuntimeError('Frozen full-case validation failed')
        state[phase+'_case_valid']=True;state[phase+'_windows']=result.get('windows',[]);save()
    save()
    def interrupted(signum,frame):raise RuntimeError('Owned controller interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for row in prepared['execution_pins']:
            p=Path(row['path']);b=p.read_bytes()
            if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:raise RuntimeError('Execution input changed: '+str(p))
        main=inspect(MAIN)
        if not main['State']['Running'] or main['Config']['Labels'].get('wio.migration')!='166-to-171-20261102':raise RuntimeError('Original own service not running')
        diagnostic=inspect(diag)
        if diagnostic['State']['Status']!='created' or diagnostic['Config']['Labels'].get('wio.shared-cohort')!=plan['run_id']:
            raise RuntimeError('Prepared diagnostic container is not untouched CREATED')
        pids,text=busy();owns_processes(pids,owned(MAIN));(root/'npu-before.txt').write_text(text)
        idle_metrics(request(30075,'/metrics')[1].decode())
        state['main_start_before']=main['State']['StartedAt'];save()
        run_client('baseline',30075,plan['limits']['client_seconds'])
        idle_metrics(request(30075,'/metrics')[1].decode())
        pids,text=busy();owns_processes(pids,owned(MAIN));(root/'npu-before-stop.txt').write_text(text)
        # Set the restoration obligation before asking Docker to stop.
        main_stopped=True;state['state']='STOPPING_OWN_ORIGINAL_SERVICE';save()
        command(['docker','stop','--time','60',MAIN],90)
        idle_deadline=time.monotonic()+60
        while True:
            pids,text=busy()
            if not pids:break
            if time.monotonic()>idle_deadline:raise RuntimeError('No idle16-device window;diagnostic not started')
            monitor();time.sleep(1)
        (root/'npu-idle.txt').write_text(text)
        diag_started=True;state['state']='STARTING_ISOLATED_DIAGNOSTIC';save()
        command(['docker','start',diag],60)
        ready_deadline=time.monotonic()+plan['limits']['diagnostic_load_seconds']
        while True:
            monitor();d=inspect(diag)
            if not d['State']['Running']:raise RuntimeError('Diagnostic service exited during load')
            try:
                if request(plan['diagnostic_port'],'/health')[0]==200:break
            except Exception:pass
            if time.monotonic()>ready_deadline:raise RuntimeError('Diagnostic loading deadline reached')
            time.sleep(3)
        pids,text=busy();owns_processes(pids,owned(diag));(root/'npu-diagnostic-ready.txt').write_text(text)
        run_client('profile',plan['diagnostic_port'],plan['limits']['client_seconds'])
        idle_metrics(request(plan['diagnostic_port'],'/metrics')[1].decode())
        (control/'FINALIZE_METADATA').write_text('Full client returned;close only the supplementary host logs.\n')
        code,body=request(plan['diagnostic_port'],'/stop_profile',{},90)
        if code!=200:raise RuntimeError('Metadata finalization RPC failed')
        monitor();state['state']='FIXED_CASE_COHORT_FINISHED_RAW_EVIDENCE_REVIEW_PENDING';save()
    except BaseException:
        state.update(state='COHORT_FAILED_OR_GUARD_STOPPED',error=traceback.format_exc());save()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        if diag_started:
            try:
                current=inspect(diag)
                if current['State']['Running']:
                    state['diagnostic_stop']=command(['docker','stop','--time','10',diag],45).strip()
                diag_started=False
            except BaseException:state['diagnostic_stop_error']=traceback.format_exc()
        try:
            if main_stopped:
                clear_deadline=time.monotonic()+60
                while True:
                    pids,text=busy()
                    if not pids or time.monotonic()>clear_deadline:break
                    time.sleep(1)
                (root/'npu-before-restore.txt').write_text(text)
                if not restore_allowed(pids):
                    state.update(restore='DEFERRED_OCCUPIED_OR_UNCLEARED_NPU',restore_busy_pids=sorted(pids))
                else:
                    command(['python3',str(MIGRATION/'environment_control171.py'),'start'],150)
                    state['restore']='LOADING_OWN_ORIGINAL_SERVICE';save()
                    deadline=time.monotonic()+plan['limits']['restore_load_seconds']
                    while True:
                        if not inspect(MAIN)['State']['Running']:raise RuntimeError('Original service exited during restore')
                        try:
                            if request(30075,'/health')[0]==200:break
                        except Exception:pass
                        if time.monotonic()>deadline:raise RuntimeError('Original restore deadline reached')
                        state['restore_updated_unix']=time.time();save();time.sleep(3)
                    state['restore']='OWN_ORIGINAL_SERVICE_HEALTH_200'
                    pids,text=busy();owns_processes(pids,owned(MAIN));(root/'npu-after-restore.txt').write_text(text)
            else:
                state['restore']='NOT_REQUIRED_ORIGINAL_SERVICE_NOT_STOPPED'
        except BaseException:
            state['restore_error']=traceback.format_exc()
        state['finished_unix']=time.time()
        try:state['diagnostic_container_state']=inspect(diag)['State']['Status']
        except BaseException:state['diagnostic_container_state']='UNKNOWN_INSPECT_FAILED'
        state['full_operator_goal_complete']=False
        save();print(json.dumps(state),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=a.root.resolve()
    if root.parent!=MIGRATION or not root.name.startswith('builtin-cohort-v164-'):raise RuntimeError('Owned diagnostic root required')
    if not a.execute:
        plan=json.loads((root/'control/PLAN.json').read_text())
        print(json.dumps({'status':'PREPARED_NOT_EXECUTED','plan':plan,'authorization_present':(root/'control/APPROVAL.json').exists()}));return
    execute(root)
if __name__=='__main__':main()
