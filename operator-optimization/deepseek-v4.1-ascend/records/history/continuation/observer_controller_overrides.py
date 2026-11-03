# These definitions replace the legacy profiler preparation/client/main.
# Active path collects tensor values only, without a profiler.
import fcntl
import shutil
import tarfile
from pathlib import Path

OBSERVER_BUDGET=256*1024**2
DRAFT_ROOT=Path(ROOT+REL)/'observer-draft'
BASELINE_RELEASE_REFERENCE=Path('/data/tiankuan/wio/migration-166-20261030/npu-after-stop.txt')

def json_write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def hbm_rows(raw):
    out={};board=None
    for row in raw.splitlines():
        a=[x.strip() for x in row.split('|')]
        if len(a)>3 and re.fullmatch(r'\d+\s+Ascend910',a[1]):board=int(a[1].split()[0])
        if len(a)>5 and board is not None and re.fullmatch(r'\d+\s+\d+',a[1]) and re.fullmatch(r'\d+\s*/\s*\d+',a[-2]):
            chip=int(a[1].split()[0]);used,total=map(int,a[-2].split('/'));out[(board,chip)]=used
    if len(out)!=16:raise RuntimeError('Could not parse complete 16-chip HBM table')
    return out

def wait_released(timeout=120):
    ref=BASELINE_RELEASE_REFERENCE.read_text();refp=set()
    for line in ref.splitlines():
        a=[x.strip() for x in line.split('|')]
        if len(a)>4 and re.fullmatch(r'\d+\s+\d+',a[1]) and a[2].isdigit():refp.add(int(a[2]))
    if refp:raise RuntimeError('Empirical idle reference contains device processes')
    refh=hbm_rows(ref);deadline=time.monotonic()+timeout;stable=0;last=None
    while time.monotonic()<deadline:
        pids,raw=npu_snapshot();cur=hbm_rows(raw)
        # Policy envelope, not a mathematical proof of every allocation owner.
        good=not pids and all(cur[k]<=refh[k]+1024 for k in refh)
        stable=stable+1 if good else 0;last={'pids':sorted(pids),'reported_hbm_MiB':{str(k):v for k,v in cur.items()},'policy':'two empty PID snapshots; per-chip HBM<=own previously observed idle+1024MiB; not universal ownership proof'}
        if stable>=2:return last
        if pids:raise RuntimeError('Other NPU processes appeared after stopping own instance: '+repr(sorted(pids)))
        time.sleep(2)
    raise RuntimeError('HBM release policy not met: '+repr(last))

def prepare():
    d=baseline_check();stage=inspect(STAGE)
    if stage['Id']!='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d' or stage['Image']!=IMAGE or not stage['State']['Running']:raise RuntimeError('Stage identity changed')
    if not d['State']['Running'] or serve_hash(BASE)!=SERVE_SHA:raise RuntimeError('Baseline unavailable or serve script changed')
    if dict(x.split('=',1) for x in d['Config']['Env']).get('CANN_VERSION')!='9.2.0':raise RuntimeError('CANN differs')
    guard={'health':health(BASE,True),'ownership':owner_guard(BASE),'baseline':{'id':d['Id'],'image':d['Image'],'state':d['State']}}
    reference=json.loads((Path(ROOT+REL)/'reference-latest.json').read_text())
    if reference['state']!='PASS_FIXED_C8_REFERENCE':raise RuntimeError('Full-case reference missing')
    draft=json.loads((DRAFT_ROOT/'PLAN.json').read_text());pre=json.loads((DRAFT_ROOT/'import-preflight.json').read_text())
    if pre['status']!='PASS_LAZY_IMPORT_ONLY':raise RuntimeError('Observer lazy import preflight missing')
    for name,digest in draft['code_hashes'].items():
        if hashlib.sha256((DRAFT_ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Observer source changed '+name)
    root=Path(HOST_RUN);root.mkdir(exist_ok=False);dst=root/'runtime';dst.mkdir()
    for n in ('scripts','config','cache','home'):
        src=Path(ROOT)/'runtime'/n
        if src.exists():shutil.copytree(src,dst/n,symlinks=True)
        else:(dst/n).mkdir()
    for n in ('tmp','logs','official-traces','observer','instrumentation'):(dst/n).mkdir()
    for n in ('sitecustomize.py','wio_moe_observer.py'):shutil.copy2(DRAFT_ROOT/n,dst/'instrumentation'/n)
    client_source=Path('/data/tiankuan/wio/migration-166-20261030/source169-project/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official_profile_client.py')
    if hashlib.sha256(client_source.read_bytes()).hexdigest()!='42a6d9e54ea0a922843e8ab37f4096e52c54fb4dbcdd23b77bb60d69293f3131':raise RuntimeError('Frozen client source mismatch')
    shutil.copy2(client_source,dst/'official_profile_client.py')
    if hashlib.sha256((dst/'scripts/serve.sh').read_bytes()).hexdigest()!=SERVE_SHA:raise RuntimeError('Copied serve changed')
    # Original serve.sh remains byte-identical. Only the isolated process gets
    # the diagnostic import path, output path and private TMPDIR bind.
    script='#!/usr/bin/env bash\nset -euo pipefail\nexport PYTHONPATH='+shlex.quote(APP+'/instrumentation')+':"${PYTHONPATH:-}"\nexport WIO_OBSERVER_ROOT='+shlex.quote(APP+'/observer')+'\nexec bash '+shlex.quote(APP+'/scripts/serve.sh')+'\n'
    (dst/'scripts/serve-observer.sh').write_text(script)
    cfg=d['HostConfig'];args=['docker','create','--name',DIAG,'--label','wio.task='+RUN,'--runtime',cfg['Runtime'],'--privileged','--read-only','--network','bridge','--ipc','private','--shm-size',str(cfg['ShmSize']),'--workdir',APP,'--entrypoint','bash']
    for env in d['Config']['Env']:args+=['--env',env]
    for u in cfg.get('Ulimits') or []:args+=['--ulimit',f"{u['Name']}={u['Soft']}:{u['Hard']}"]
    for cap in cfg.get('CapAdd') or []:args+=['--cap-add',cap]
    for opt in cfg.get('SecurityOpt') or []:args+=['--security-opt',opt]
    for m in d['Mounts']:
        if m['Type']!='bind':raise RuntimeError('Unexpected non-bind mount')
        if m['Destination']=='/tmp':continue
        src=str(dst) if m['Destination']==APP else m['Source']
        if m['RW'] and m['Destination']!=APP:raise RuntimeError('Unexpected writable bind')
        args+=['--mount','type=bind,src='+src+',dst='+m['Destination']+(',readonly' if not m['RW'] else '')]
    args+=['--mount','type=bind,src='+str(dst/'tmp')+',dst=/tmp','--mount','type=bind,src='+ROOT+'/runtime/benchmark-20261028,dst=/bench-inputs,readonly']
    for dev in cfg.get('Devices') or []:args+=['--device',dev['PathOnHost']+':'+dev['PathInContainer']+':'+dev['CgroupPermissions']]
    args += [IMAGE,'-c','exec bash '+APP+'/scripts/serve-observer.sh >'+APP+'/logs/service.log 2>&1']
    diag_id=cmd(args,timeout=90).stdout.strip();created=inspect(DIAG)
    if created['Id']!=diag_id or created['Image']!=IMAGE:raise RuntimeError('Diagnostic identity failed')
    json_write(root/'diagnostic-identity.json',{'name':DIAG,'id':diag_id,'label':RUN,'baseline_id':BASE_ID,'baseline_prestate':d['State'],'image':IMAGE})
    json_write(root/'docker-create-argv.json',args)
    json_write(root/'preflight.json',{'guard':guard,'reference':reference,'observer_plan':draft,'source_serve_sha':SERVE_SHA,'differences':['isolated mounted runtime/cache/home/tmp/logs','same-call observer adds ordinal update and device-to-host snapshots; timing perturbed','no profiler; no optimization or returned tensor mutation'],'budget_observer_bytes':OBSERVER_BUDGET,'max_eager_samples_per_layer_shape':3,'all_module_coverage_remains_open':True})
    return diag_id

def observer_size():
    return sum(p.stat().st_size for p in (Path(HOST_RUN)/'runtime/observer').rglob('*') if p.is_file())

def client(phase):
    free_guard();proc=subprocess.Popen(['docker','exec','-e','WIO_PHASE='+phase,'-e','WIO_OUTPUT='+APP+'/client-'+phase,DIAG,'timeout','--kill-after=10s','600s','python3',APP+'/official_profile_client.py'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    deadline=time.monotonic()+625
    try:
        while proc.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Finite observer client timeout')
            free_guard()
            if observer_size()>OBSERVER_BUDGET:raise RuntimeError('Observer disk budget exceeded')
            time.sleep(2)
        out,err=proc.communicate(timeout=10);stage_write('client-'+phase+'-process.json',{'exit_code':proc.returncode,'stdout':out,'stderr':err})
        if proc.returncode:raise RuntimeError(phase+' client failed '+err[-1500:])
        result=json.loads((Path(HOST_RUN)/'runtime'/('client-'+phase)/'window-status.json').read_text())
        if not result['valid']:raise RuntimeError('Frozen workload counter validation failed')
    finally:
        if proc.poll() is None:proc.terminate()

def arm_after_warmup():
    health(DIAG,True);root=Path(HOST_RUN)/'runtime/observer';time.sleep(2);files=sorted(root.rglob('*.pt'))
    if not files:raise RuntimeError('No warmup/captured tensor files; observer path not executed')
    if observer_size()>OBSERVER_BUDGET:raise RuntimeError('Observer warmup budget exceeded')
    with tarfile.open(Path(HOST_RUN)/'observer-warmup-snapshots.tar.gz','w:gz',compresslevel=1) as tf:
        for p in files:tf.add(p,arcname=str(p.relative_to(root)),recursive=False)
    # Captured file paths must remain valid. Preserve directories and metadata,
    # remove only old tensor snapshots before actual measured-phase requests.
    for p in files:p.unlink()
    json_write(root/'ARMED.json',{'phase':'fixed-measure1','armed_epoch_ns':time.time_ns(),'warmup_tensor_files_archived_and_removed':len(files),'capture_metadata_retained':True})
    stage_write('arm-record.json',json.loads((root/'ARMED.json').read_text()))

def inventory():
    root=Path(HOST_RUN)/'runtime/observer';previous=None;stable=0
    for _ in range(10):
        files=sorted(root.rglob('*.pt'));marker=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files]
        stable=stable+1 if marker==previous else 0;previous=marker
        if stable>=2:break
        time.sleep(2)
    if not files or stable<2:raise RuntimeError('Post-case snapshots missing/not stable')
    rows=[];rankdirs=set();layers=set()
    for p in files:
        stem=re.sub(r'_device_\d+_\d+$','',p.stem);meta=p.with_name(stem+'.json')
        if not meta.is_file():raise RuntimeError('Tensor has no binding metadata '+str(p))
        m=json.loads(meta.read_text());rankdirs.add(m['ep_rank']);layers.add((m['role'],m['layer_name']))
        rows.append({'tensor':str(p),'tensor_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'metadata':str(meta),'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),'ep_rank':m['ep_rank'],'role':m['role'],'layer_name':m['layer_name'],'capturing':m['capturing'],'M':m['M'],'K':m['K'],'E_local':m['E_local']})
    json_write(Path(HOST_RUN)/'tensor-index.json',rows)
    result={'status':'COLLECTED_VALUES_NOT_YET_SEMANTICALLY_VALIDATED','tensor_files':len(rows),'ep_ranks':sorted(rankdirs),'rank_count':len(rankdirs),'layer_surfaces':[list(x) for x in sorted(layers)],'observer_bytes':observer_size(),'all_case_counters_valid':True,'needs_next_review':['read CPU tensor values and observer ordinals','match all16 ranks by actual group/layer/ordinal, validate routing-count conservation','bind actual custom/CANN branch, capacity, mask and expert placement','capture-path overwrite and sampling gaps; not complete whole-module coverage'],'model_speedup_claimed':False,'overall_goal_complete':False}
    if len(rankdirs)!=16:raise RuntimeError('Missing EP ranks '+repr(sorted(rankdirs)))
    json_write(Path(HOST_RUN)/'capture-result.json',result)

def restore(diag_id):
    stop_diag(diag_id);d=baseline_check()
    if not d['State']['Running']:
        try:release=wait_released(timeout=120)
        except Exception as e:status('RESTORE_DEFERRED_DEVICE_NOT_RELEASED',error=repr(e));return False
        json_write(Path(HOST_RUN)/'release-before-restore.json',release)
        cmd(['docker','start',BASE_ID],timeout=120)
    result=wait_health(BASE,timeout=2400);guard=owner_guard(BASE)
    if serve_hash(BASE)!=SERVE_SHA:raise RuntimeError('Restored serve SHA mismatch')
    json_write(Path(HOST_RUN)/'restore-result.json',{'status':'PASS_ORIGINAL_BASELINE_RESTORED','id':BASE_ID,'image':IMAGE,'health':result,'all_npu_pids_owned':True,'npu_pids':guard['npu_pids'],'prefix_cache_cold_reseed_required':True,'source_code_or_serve_modified':False})
    status('RESTORED',health=result,ownership=guard,serve_sha=SERVE_SHA,baseline_id=BASE_ID,image=IMAGE,note='Original baseline restored; prefix cache cold, re-seed for next comparison')
    return True

def main():
    lock_path=Path(ROOT+REL)/'.device-diagnostic.lock';lock=open(lock_path,'a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    diag_id=prepare();parent=os.getpid();raw=Path('/proc',str(parent),'stat').read_text();parent_start=raw[raw.rfind(')')+2:].split()[19];child=os.fork()
    if child==0:watchdog(parent,parent_start,diag_id)
    def requested_stop(signum,frame):raise RuntimeError('Finite diagnostic interrupted/ownership monitor stop: '+str(signum))
    signal.signal(signal.SIGUSR1,requested_stop);signal.signal(signal.SIGTERM,requested_stop);signal.signal(signal.SIGINT,requested_stop)
    samples=[];stop=threading.Event();sampler=threading.Thread(target=sample_loop,args=(stop,samples),daemon=True);failed=None
    try:
        status('PREPARED',diag_id=diag_id,watchdog_pid=child,observer_only=True)
        baseline_check();health(BASE,True);owner_guard(BASE)
        cmd(['docker','stop','--time','45',BASE_ID],timeout=100)
        json_write(Path(HOST_RUN)/'release-before-diagnostic.json',wait_released(timeout=120))
        cmd(['docker','start',diag_id],timeout=120);sampler.start();status('DIAGNOSTIC_LOADING',diag_id=diag_id,watchdog_pid=child)
        wait_health(DIAG,timeout=2400);owner_guard(DIAG);stage_write('runtime-versions.json',runtime_versions())
        status('WARMUP_FIXED_CASE_RUNNING');client('warmup');arm_after_warmup()
        status('OBSERVED_FIXED_CASE_RUNNING');client('baseline');health(DIAG,True);inventory()
        status('VALUES_COLLECTED_PENDING_SEMANTIC_REVIEW')
    except BaseException as e:
        failed=repr(e);stage_write('controller-error.json',{'error':repr(e),'traceback':traceback.format_exc()});raise
    finally:
        stop.set()
        if sampler.is_alive():sampler.join(timeout=40)
        stage_write('npu-smi-samples.json',samples)
        restored=restore(diag_id)
        if not restored:raise RuntimeError('Original baseline restore deferred; see explicit state')
        json_write(Path(HOST_RUN)/'controller-finished.json',{'state':'RESTORED_AFTER_ERROR' if failed else 'CAPTURED_AND_RESTORED','capture_error':failed,'watchdog_pid':child,'baseline_id':BASE_ID,'overall_goal_complete':False})
        # Child keeps the inherited flock until this parent exits and it verifies
        # restoration. That prevents another owner operation from racing recovery.
        lock.close()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        try:stage_write('top-level-error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        except BaseException:pass
        raise
