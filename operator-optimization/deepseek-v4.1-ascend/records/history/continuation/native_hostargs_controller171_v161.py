"""Opt-in single native probe window; only own two fixed containers are managed."""
import argparse,json,hashlib,os,re,time,subprocess,traceback,urllib.request
from pathlib import Path
ROOT=Path('/data/tiankuan/wio/migration-171-20261102/operator-evidence-v161')
MAIN='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb'
STAGE='117d57490a2f2f253495f7a30a438eb64784787b5d71351104b90c9008a8db2d'
def cmd(argv,timeout=30):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    assert p.returncode==0,{'argv_prefix':argv[:4],'rc':p.returncode,'stdout':p.stdout[-1800:],'stderr':p.stderr[-1800:]}
    return p.stdout
def inspect(cid):
    d=json.loads(cmd(['docker','inspect',cid]))[0]
    assert d['Id']==cid and d['Config']['Labels']['wio.migration']=='166-to-171-20261102'
    return d
def busy():
    s=cmd(['npu-smi','info']);ids=[]
    for line in s.splitlines():
        f=[x.strip() for x in line.split('|')]
        if len(f)>4 and re.fullmatch(r'\d+\s+\d+',f[1]) and f[2].isdigit():ids.append(int(f[2]))
    return set(ids),s
def health():
    op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open('http://127.0.0.1:30075/health',timeout=3) as r:return r.status
def execute():
    prep=json.loads((ROOT/'PREPARED-PROBE.json').read_text())
    approval=json.loads((ROOT/'APPROVAL.json').read_text())
    assert approval['status']=='AUTHORIZED_ONCE' and approval['probe_source_sha256']==prep['probe_source_sha256']
    assert approval['bundle']==prep['bundle_inside'] and approval.get('user_reply')
    status=ROOT/'NATIVE-STATUS.json';assert not status.exists(),'Submitted window exists;inspect it;never restart from observation timeout'
    state={'state':'GUARD','pid':os.getpid(),'native_attempts':0,'probe_approval_id':approval['approval_id'],'main_container':MAIN,'stage_container':STAGE,'started_unix':time.time(),'model_loads_during_probe':0,'graph_captures':0,'graph_replays':0}
    def save():
        p=status.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(status)
    main_stopped=False;stage_started=False
    save()
    try:
        d=inspect(MAIN);assert d['State']['Running']
        own={int(s.strip()) for s in cmd(['docker','top',MAIN,'-eo','pid']).splitlines() if s.strip().isdigit()}
        pids,smi=busy();assert pids and pids<=own,'Other NPU process present or ownership unknown;stop before changing own service'
        (ROOT/'npu-before-window.txt').write_text(smi)
        stage=inspect(STAGE);assert not stage['State']['Running']
        bundle=Path(prep['bundle_host']);assert hashlib.sha256((bundle/'probe.py').read_bytes()).hexdigest()==prep['probe_source_sha256']
        assert hashlib.sha256((bundle/'libhost_adapter.so').read_bytes()).hexdigest()==prep['adapter_binary']['sha256']
        assert not (bundle/'probe-result.json').exists() and not list((bundle/'trace').glob('*.jsonl'))
        import shutil
        service_log=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/logs/service.log')
        shutil.copy2(service_log,ROOT/'service-log-before-probe.log')
        cmd(['docker','stop','--time','60',MAIN],90);main_stopped=True;state.update(state='OWN_SERVICE_STOPPED');save()
        idle_deadline=time.monotonic()+30
        while True:
            pids,smi=busy()
            if not pids:break
            assert time.monotonic()<idle_deadline,'No fresh idle-device window;no probe submitted'
            time.sleep(1)
        (ROOT/'npu-idle-before-probe.txt').write_text(smi)
        cmd(['docker','start',STAGE],30);stage_started=True
        childapproval=dict(approval);(bundle/'APPROVED.json').write_text(json.dumps(childapproval))
        inside=prep['bundle_inside']
        env={'ASCEND_VISIBLE_DEVICES':'0','ASCEND_RT_VISIBLE_DEVICES':'0','ASCEND_DEVICE_ID':'0','LD_PRELOAD':inside+'/libhost_adapter.so','WIO_NATIVE_PROBE_ACK':approval['approval_id'],'WIO_HOST_TRACE_ENABLE':'1','WIO_HOST_TRACE_DIR':inside+'/trace','WIO_HOST_TRACE_RUN_ID':'pad171v161','WIO_HOST_TRACE_ARM_FILE':inside+'/armed','WIO_HOST_TRACE_MAX_RECORDS':'2','WIO_HOST_TRACE_MAX_BYTES':'65536','WIO_HOST_TRACE_MAX_SECONDS':'60','WIO_HOST_TRACE_QUERY_IDS':'0','WIO_HOST_TRACE_REAL_LIBRARY':'/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libascendcl.so','ASCEND_SLOG_PRINT_TO_STDOUT':'0','ASCEND_GLOBAL_LOG_LEVEL':'3','ASCEND_PROCESS_LOG_PATH':inside+'/ascend-logs','HOME':inside+'/home'}
        (bundle/'home').mkdir(mode=0o700,exist_ok=True)
        argv=['docker','exec']
        for k,v in env.items():argv+=['-e',k+'='+v]
        argv+=[STAGE,'timeout','--signal=TERM','--kill-after=5s','45s','python3','-I','-B',inside+'/probe.py','--run','--bundle',inside]
        state.update(state='ONE_NATIVE_ATTEMPT_SUBMITTED',native_attempts=1,submitted_unix=time.time());save()
        with (ROOT/'probe-stdout.log').open('xb') as out,(ROOT/'probe-stderr.log').open('xb') as err:
            p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err);state['owned_docker_exec_pid']=p.pid;save();deadline=time.monotonic()+60
            while p.poll() is None:
                total=sum(x.stat().st_size for x in bundle.rglob('*') if x.is_file() and x.name not in ('probe.py','libhost_adapter.so','APPROVED.json'))+(ROOT/'probe-stdout.log').stat().st_size+(ROOT/'probe-stderr.log').stat().st_size
                if total>8*1024*1024 or time.monotonic()>deadline:
                    cmd(['docker','stop','--time','5',STAGE],20);stage_started=False
                    if p.poll() is None:p.terminate()
                    p.wait(timeout=10);raise RuntimeError('Owned probe exceeded output/time guard;not resubmitted')
                time.sleep(0.25)
            rc=p.returncode
        state['native_exit_code']=rc
        records=[];files=[]
        for f in sorted((bundle/'trace').glob('*.jsonl')):
            b=f.read_bytes();assert len(b)<=65536;files.append({'path':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});records.extend(json.loads(line) for line in b.splitlines())
        sample=json.loads((bundle/'probe-result.json').read_text()) if (bundle/'probe-result.json').exists() else None
        packets=[r for r in records if r.get('type')=='host_args']
        for row in packets:assert row['args_size']<=4096 and len(bytes.fromhex(row['args_hex']))==row['args_size'] and len(row['placeholders'])<=32
        summary={'scope':'ONE_OWN_EAGER_PAD_PAIR','native_exit_code':rc,'sample':sample,'trace_files':files,'trace_records':records,'host_packet_records':len(packets),'native_trace_row_cap':2,'original_model_operands_recovered':0,'all_error_or_bypass_paths_validated':False,'captured_BOOL_creation_route_validated':False,'full_model_capture_ready':False,'overall_complete':False}
        (ROOT/'NATIVE-RESULT.json').write_text(json.dumps(summary,indent=2))
        assert rc==0 and sample and sample['status']=='EAGER_CONTROL_OBSERVED_VALUES_MATCH','Inspect the single terminal probe result;do not repeat automatically'
        state.update(state='PROBE_FINISHED',host_packet_records=len(packets),interposition_seen_for_this_case=bool(packets),full_model_capture_ready=False);save()
    except BaseException:
        state.update(state='PROBE_WINDOW_FAILED_OR_GUARD_STOPPED',error=traceback.format_exc());save()
    finally:
        try:
            if stage_started:cmd(['docker','stop','--time','10',STAGE],30)
            if main_stopped:
                pids,smi=busy();(ROOT/'npu-before-restore.txt').write_text(smi)
                if pids:state['restore']='DEFERRED_OTHER_OR_UNCLEARED_NPU_PROCESSES';state['restore_busy_pids']=sorted(pids)
                else:
                    cmd(['python3','/data/tiankuan/wio/migration-171-20261102/environment_control171.py','start'],150)
                    state['restore']='LOADING_OWN_ORIGINAL_SERVICE';save();deadline=time.monotonic()+2400
                    while True:
                        d=inspect(MAIN);assert d['State']['Running'],'Own original service exited during restore'
                        try:
                            if health()==200:break
                        except Exception:pass
                        assert time.monotonic()<deadline,'Original-service restore deadline reached';time.sleep(5)
                    state['restore']='OWN_ORIGINAL_SERVICE_HEALTH_200'
        except BaseException:state['restore_error']=traceback.format_exc()
        state['finished_unix']=time.time();save()
        print(json.dumps(state),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print(json.dumps({'status':'PREPARED_NOT_EXECUTED','stop_only_own_migrated_service':MAIN,'temporary_probe_container':STAGE,'restore_original_service_after_probe':True,'one_native_attempt_only':True,'approval_file':str(ROOT/'APPROVAL.json')}));return
    execute()
if __name__=='__main__':main()
