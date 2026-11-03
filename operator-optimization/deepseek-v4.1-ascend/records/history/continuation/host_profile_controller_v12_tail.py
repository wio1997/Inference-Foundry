"""Appended to the previously reviewed finite diagnostic controller helpers."""
import fcntl
import shutil
from pathlib import Path

DRAFT_ROOT=Path(ROOT+REL)/'host-profile-draft-v12'
BASELINE_RELEASE_REFERENCE=Path('/data/tiankuan/wio/migration-166-20261030/npu-after-stop.txt')

def json_write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def hbm_rows(raw):
    out={};board=None
    for row in raw.splitlines():
        a=[x.strip() for x in row.split('|')]
        if len(a)>3 and re.fullmatch(r'\d+\s+Ascend910',a[1]):board=int(a[1].split()[0])
        if len(a)>3 and board is not None and re.fullmatch(r'\d+\s+\d+',a[1]) and ':' in a[2]:
            pairs=re.findall(r'(\d+)\s*/\s*(\d+)',row)
            if not pairs:raise RuntimeError('Chip HBM field missing')
            chip=int(a[1].split()[0]);used,total=map(int,pairs[-1]);out[(board,chip)]=used
    if len(out)!=16:raise RuntimeError('Could not parse complete16-chip HBM table')
    return out

def wait_released(timeout=120):
    ref=BASELINE_RELEASE_REFERENCE.read_text();refp=set()
    for line in ref.splitlines():
        a=[x.strip() for x in line.split('|')]
        if len(a)>4 and re.fullmatch(r'\d+\s+\d+',a[1]) and a[2].isdigit():refp.add(int(a[2]))
    if refp:raise RuntimeError('Idle reference contains device processes')
    refh=hbm_rows(ref);deadline=time.monotonic()+timeout;stable=0;last=None
    while time.monotonic()<deadline:
        pids,raw=npu_snapshot();cur=hbm_rows(raw)
        good=not pids and all(cur[k]<=refh[k]+1024 for k in refh)
        stable=stable+1 if good else 0
        last={'pids':sorted(pids),'reported_hbm_MiB':{str(k):v for k,v in cur.items()},'policy':'two empty PID snapshots; per-chip HBM<=own previously observed idle+1024MiB; operational policy only'}
        if stable>=2:return last
        if pids:raise RuntimeError('Other NPU processes appeared: '+repr(sorted(pids)))
        time.sleep(2)
    raise RuntimeError('HBM release policy not met: '+repr(last))

def prepare():
    d=baseline_check();stage=inspect(STAGE)
    if stage['Id']!='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d' or stage['Image']!=IMAGE or not stage['State']['Running']:raise RuntimeError('Stage identity changed')
    if not d['State']['Running'] or serve_hash(BASE)!=SERVE_SHA:raise RuntimeError('Baseline unavailable or serve script changed')
    if dict(x.split('=',1) for x in d['Config']['Env']).get('CANN_VERSION')!='9.2.0':raise RuntimeError('CANN differs')
    if any('instrumentation' in x or 'sitecustomize' in x for x in d['Config']['Env']):raise RuntimeError('Baseline has unexpected diagnostic environment')
    guard={'health':health(BASE,True),'ownership':owner_guard(BASE),'baseline':{'id':d['Id'],'image':d['Image'],'state':d['State']}}
    pre=json.loads((DRAFT_ROOT/'preflight.json').read_text());plan=json.loads((DRAFT_ROOT/'PLAN.json').read_text())
    if pre['status']!='PASS_LAZY_IMPORT_NO_TORCH_NPU':raise RuntimeError('Diagnostic lazy preflight missing')
    for name,sha in plan['code_hashes'].items():
        if hashlib.sha256((DRAFT_ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Prepared diagnostic source changed')
    root=Path(HOST_RUN);root.mkdir(exist_ok=False);dst=root/'runtime';dst.mkdir()
    for n in ('scripts','config','cache','home'):
        src=Path(ROOT)/'runtime'/n
        if src.exists():shutil.copytree(src,dst/n,symlinks=True)
        else:(dst/n).mkdir()
    for n in ('tmp','logs','official-traces','host-profile','instrumentation'):(dst/n).mkdir()
    for n in ('sitecustomize.py','wio_host_profile.py'):shutil.copy2(DRAFT_ROOT/n,dst/'instrumentation'/n)
    client_source=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official_profile_client.py')
    if hashlib.sha256(client_source.read_bytes()).hexdigest()!='42a6d9e54ea0a922843e8ab37f4096e52c54fb4dbcdd23b77bb60d69293f3131':raise RuntimeError('Frozen client source changed')
    shutil.copy2(client_source,dst/'official_profile_client.py')
    if hashlib.sha256((dst/'scripts/serve.sh').read_bytes()).hexdigest()!=SERVE_SHA:raise RuntimeError('Copied serve changed')
    lines=(dst/'scripts/serve.sh').read_text().splitlines()
    positions=[i for i,line in enumerate(lines) if line.startswith('exec /usr/local/python3.12.13/bin/vllm serve ')]
    if len(positions)!=1:raise RuntimeError('Exact frozen serve command not found')
    lines[positions[0]]+=' --profiler-config '+shlex.quote(json.dumps(PROFILE,separators=(',',':')))
    lines[positions[0]:positions[0]]=[
      'export PYTHONPATH='+shlex.quote(APP+'/instrumentation')+':"${PYTHONPATH:-}"',
      'export WIO_HOST_PROFILE_ROOT='+shlex.quote(APP+'/host-profile'),
      'export TMPDIR=/tmp']
    (dst/'scripts/serve-host-profile.sh').write_text('\n'.join(lines)+'\n')
    cfg=d['HostConfig'];args=['docker','create','--name',DIAG,'--label','wio.task='+RUN,'--runtime',cfg['Runtime'],'--privileged','--read-only','--network','bridge','--ipc','private','--shm-size',str(cfg['ShmSize']),'--workdir',APP,'--entrypoint','bash']
    for env in d['Config']['Env']:args+=['--env',env]
    args+=['--env','VLLM_SERVER_DEV_MODE=1','--env','VLLM_CUSTOM_SCOPES_FOR_PROFILING=1','--env','TMPDIR=/tmp']
    for u in cfg.get('Ulimits') or []:args+=['--ulimit',f"{u['Name']}={u['Soft']}:{u['Hard']}"]
    for cap in cfg.get('CapAdd') or []:args+=['--cap-add',cap]
    for opt in cfg.get('SecurityOpt') or []:args+=['--security-opt',opt]
    for m in d['Mounts']:
        if m['Type']!='bind':raise RuntimeError('Unexpected non-bind mount')
        if m['Destination']=='/tmp':continue
        src=str(dst) if m['Destination']==APP else m['Source']
        if m['RW'] and m['Destination']!=APP:raise RuntimeError('Unexpected writable baseline bind')
        args+=['--mount','type=bind,src='+src+',dst='+m['Destination']+(',readonly' if not m['RW'] else '')]
    args+=['--mount','type=bind,src='+str(dst/'tmp')+',dst=/tmp','--mount','type=bind,src='+ROOT+'/runtime/benchmark-20261028,dst=/bench-inputs,readonly']
    for dev in cfg.get('Devices') or []:args+=['--device',dev['PathOnHost']+':'+dev['PathInContainer']+':'+dev['CgroupPermissions']]
    args += [IMAGE,'-c','exec bash '+APP+'/scripts/serve-host-profile.sh >'+APP+'/logs/service.log 2>&1']
    diag_id=cmd(args,timeout=90).stdout.strip();created=inspect(DIAG)
    if created['Id']!=diag_id or created['Image']!=IMAGE:raise RuntimeError('Diagnostic identity failed')
    json_write(root/'diagnostic-identity.json',{'name':DIAG,'id':diag_id,'label':RUN,'baseline_id':BASE_ID,'baseline_prestate':d['State'],'image':IMAGE})
    json_write(root/'docker-create-argv.json',args)
    json_write(root/'preflight.json',{'guard':guard,'plan':plan,'profile_config':PROFILE,'source_serve_sha':SERVE_SHA,
      'differences':['isolated mounted runtime/cache/home/tmp/logs','same-call scalar metadata and CPU range markers in temporary import hook','with_stack=True and record_shapes=True;max_iterations=3 per window;two client-controlled windows','frontend profiling enabled;actual function visibility verified after export','instrumented profiler-off baseline and separate historical uninstrumented reference;no optimization'],
      'trace_budget_bytes':TRACE_BUDGET,'controller_deadline_seconds':MAX_SECONDS,'whole_module_coverage_complete':False})
    return diag_id

def capture_size():
    roots=[Path(HOST_RUN)/'runtime/official-traces',Path(HOST_RUN)/'runtime/host-profile']
    return sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())

def client(phase):
    free_guard()
    if phase=='profile':
        json_write(Path(HOST_RUN)/'runtime/host-profile/ARMED.json',{'phase':'fixed-measure1-profile','epoch_ns':time.time_ns(),'note':'Client seeds precede profile start;only exact profiler marker/sidecar association identifies captured calls.'})
    proc=subprocess.Popen(['docker','exec','-e','WIO_PHASE='+phase,'-e','WIO_OUTPUT='+APP+'/client-'+phase,DIAG,'timeout','--kill-after=10s','600s','python3',APP+'/official_profile_client.py'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    deadline=time.monotonic()+625
    try:
        while proc.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Finite diagnostic client timeout')
            free_guard()
            if capture_size()>TRACE_BUDGET:raise RuntimeError('Finite diagnostic output budget exceeded')
            time.sleep(2)
        out,err=proc.communicate(timeout=10);stage_write('client-'+phase+'-process.json',{'exit_code':proc.returncode,'stdout':out,'stderr':err})
        if proc.returncode:raise RuntimeError(phase+' client failed '+err[-1500:])
        result=json.loads((Path(HOST_RUN)/'runtime'/('client-'+phase)/'window-status.json').read_text())
        if not result['valid']:raise RuntimeError('Frozen workload counter validation failed')
    finally:
        if proc.poll() is None:proc.terminate()

def restore(diag_id):
    stop_diag(diag_id);d=baseline_check()
    if not d['State']['Running']:
        try:release=wait_released(timeout=120)
        except Exception as e:status('RESTORE_DEFERRED_DEVICE_NOT_RELEASED',error=repr(e));return False
        json_write(Path(HOST_RUN)/'release-before-restore.json',release)
        cmd(['docker','start',BASE_ID],timeout=120)
    result=wait_health(BASE,timeout=2400);guard=owner_guard(BASE)
    if serve_hash(BASE)!=SERVE_SHA:raise RuntimeError('Restored serve SHA mismatch')
    json_write(Path(HOST_RUN)/'restore-result.json',{'status':'PASS_ORIGINAL_BASELINE_RESTORED','id':BASE_ID,'image':IMAGE,'health':result,'all_npu_pids_owned':True,'npu_pids':guard['npu_pids'],'prefix_cache_cold_reseed_required':True,'target_source_or_serve_modified':False})
    status('RESTORED',health=result,ownership=guard,serve_sha=SERVE_SHA,baseline_id=BASE_ID,image=IMAGE)
    return True

def final_inventory():
    root=Path(HOST_RUN)/'runtime';files=[]
    for directory in ['official-traces','host-profile']:
        for p in sorted((root/directory).rglob('*')):
            if p.is_file():files.append({'path':str(p),'relative_path':str(p.relative_to(root)),'bytes':p.stat().st_size})
    json_write(Path(HOST_RUN)/'capture-index.json',{'files':files,'total_bytes':sum(p['bytes'] for p in files),'complete_semantic_coverage_proved':False,'no_speedup_claim':True})

def main():
    lock=open(Path(ROOT+REL)/'.device-diagnostic.lock','a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    diag_id=prepare();parent=os.getpid();raw=Path('/proc',str(parent),'stat').read_text();parent_start=raw[raw.rfind(')')+2:].split()[19];child=os.fork()
    if child==0:watchdog(parent,parent_start,diag_id)
    def requested_stop(signum,frame):raise RuntimeError('Finite diagnostic interrupted/ownership monitor stop: '+str(signum))
    signal.signal(signal.SIGUSR1,requested_stop);signal.signal(signal.SIGTERM,requested_stop);signal.signal(signal.SIGINT,requested_stop)
    samples=[];stop=threading.Event();sampler=threading.Thread(target=sample_loop,args=(stop,samples),daemon=True);failed=None
    try:
        status('PREPARED',diag_id=diag_id,watchdog_pid=child)
        baseline_check();health(BASE,True);owner_guard(BASE)
        cmd(['docker','stop','--time','45',BASE_ID],timeout=100)
        json_write(Path(HOST_RUN)/'release-before-diagnostic.json',wait_released(timeout=120))
        cmd(['docker','start',diag_id],timeout=120);sampler.start();status('DIAGNOSTIC_LOADING',diag_id=diag_id,watchdog_pid=child)
        wait_health(DIAG,timeout=2400);owner_guard(DIAG);route_check();stage_write('runtime-versions.json',runtime_versions())
        status('WARMUP_FIXED_CASE_RUNNING');client('warmup')
        status('PROFILER_OFF_BASELINE_RUNNING');client('baseline')
        reset_prefix_cache()
        status('STACK_SHAPE_PROFILE_RUNNING');client('profile');health(DIAG,True);validate_comparison()
        status('EXPORT_WAIT');wait_export();final_inventory()
        status('PROFILE_COLLECTED_PENDING_ANALYSIS')
    except BaseException as e:
        failed=repr(e);stage_write('controller-error.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
    finally:
        stop.set()
        if sampler.is_alive():sampler.join(timeout=40)
        stage_write('npu-smi-samples.json',samples)
        restored=restore(diag_id)
        if not restored:raise RuntimeError('Original baseline restore deferred;see explicit state')
        json_write(Path(HOST_RUN)/'controller-finished.json',{'state':'RESTORED_AFTER_ERROR' if failed else 'CAPTURED_AND_RESTORED','capture_error':failed,'watchdog_pid':child,'baseline_id':BASE_ID,'overall_goal_complete':False})
        lock.close()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        try:stage_write('top-level-error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        except BaseException:pass
        raise
