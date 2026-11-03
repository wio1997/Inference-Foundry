"""CPU-only mocks: no SSH, Docker calls, or NPU execution."""
from pathlib import Path
import ast,contextlib,io,json,multiprocessing as mp,subprocess,sys,tempfile,time
from unittest import mock
import remote_guard

here=Path(__file__).resolve().parent
header='| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |'
bar='+===========================+===============+====================================================+'
empty='\n'.join([header,bar]+sum(([f'| No running processes found in NPU {i} |',bar] for i in range(8)),[]))
busy='\n'.join([header,bar]+sum(([f'| {i}       0 | {1000+i} | python | 42 |',bar] for i in range(8)),[]))
assert remote_guard.parse_npu_processes(empty)==set()
assert remote_guard.parse_npu_processes(busy)==set(range(1000,1008))
for bad in ('',header+'\n'+bar,empty.replace('NPU 7','NPU x'),busy.replace('1007','pid'),busy[:-3]):
    try:remote_guard.parse_npu_processes(bad)
    except ValueError:pass
    else:raise AssertionError('malformed table accepted')

def worker(directory,q):
    def fake_stage(path,timeout):return subprocess.Popen([sys.executable,'-u','-c',remote_guard.HOLDER_CODE,path,str(timeout)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    holder=remote_guard.acquire_owner_fence(directory,timeout=5,_launcher=fake_stage)
    remote_guard.ensure_owner_fence(holder)
    q.put(('enter',time.monotonic()));time.sleep(.15);q.put(('leave',time.monotonic()))
    holder.stdin.close();holder.wait(timeout=5)
with tempfile.TemporaryDirectory(prefix='mock-fence-',dir=here) as d:
    ctx=mp.get_context('fork');q=ctx.Queue();workers=[ctx.Process(target=worker,args=(d,q)) for _ in range(2)]
    [x.start() for x in workers];events=[q.get(timeout=5) for _ in range(4)];[x.join(5) for x in workers]
    assert [kind for kind,_ in events]==['enter','leave','enter','leave'],events
    assert all(x.exitcode==0 for x in workers)

src=(here/'recover_binding.py').read_text();tree=ast.parse(src)
remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets))
BASE_ID='b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092';DIAG_ID='d'*64;RUN_ID='a'*32

def recovery_case(diag,backup,running=True):
    calls=[];bid={'Id':BASE_ID,'State':{'Running':running}}
    identity={'baseline_id':BASE_ID,'diagnostic_name':'own-diag','binding_run_id':RUN_ID,'diagnostic_id':DIAG_ID}
    def run(args,**kw):
        calls.append(args)
        if args[:3]==['docker','exec','stage'] and args[3]=='cat':
            if backup and args[-1].endswith('identity-'+RUN_ID+'.json'):return subprocess.CompletedProcess(args,0,json.dumps(identity),'')
            return subprocess.CompletedProcess(args,1,'','missing')
        raise AssertionError(args)
    def check(args,**kw):
        calls.append(args)
        if args[:2]==['docker','inspect']:
            obj=bid if args[2]==BASE_ID else {'Id':DIAG_ID,'State':{'Running':True},'Config':{'Labels':{'wio.task':'operator-ceiling169-20261029','wio.binding_run_id':RUN_ID}}}
            return json.dumps([obj])
        if args[:2]==['docker','ps']:return DIAG_ID+'\town-diag\n' if diag else ''
        if args[:3]==['docker','stop','--time']:
            assert args[-1]==DIAG_ID;return DIAG_ID+'\n'
        raise AssertionError(args)
    env=dict(ROOT='/unused',BASE_ID=BASE_ID,RUN_ID=RUN_ID,DIAG='own-diag',STAGE='stage',WORK='/task',acquire_owner_fence=lambda x:0,parse_npu_processes=remote_guard.parse_npu_processes,ensure_owner_fence=lambda x:None,verified_release=remote_guard.verified_release,wait_for_empirical_release=remote_guard.wait_for_empirical_release)
    out=io.StringIO()
    with mock.patch('subprocess.run',side_effect=run),mock.patch('subprocess.check_output',side_effect=check),contextlib.redirect_stdout(out):
        try:exec(compile(remote,'mock-recovery','exec'),env)
        except SystemExit as e:assert e.code==0
    result=json.loads(out.getvalue());assert result['state']==('BASELINE_ALREADY_RUNNING' if running else 'RESTORE_DEFERRED_UNATTRIBUTED_HBM')
    if diag:assert any(x[:3]==['docker','stop','--time'] and x[-1]==DIAG_ID for x in calls)
    else:assert not any(x[:2]==['docker','stop'] for x in calls)
    assert not any('npu-smi' in x for x in calls),calls
    return result

no_identity=recovery_case(False,False)
backup=recovery_case(True,True)
missing_stopped=recovery_case(False,False,False)
assert missing_stopped['reason']=='NO_EMPIRICAL_REFERENCE'
for filename,key in [('start_binding_recapture.py','REMOTE'),('supervise_binding_recapture.py','CAPTURE'),('supervise_binding_recapture.py','RESTORE')]:
    t=ast.parse((here/filename).read_text());v=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==key for x in n.targets));compile(v,key,'exec')
    assert 'acquire_owner_fence(' in v
    assert 'parse_npu_processes(' in v or 'wait_for_empirical_release(' in v
real=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/005-after-switch-npu.txt').read_text()
h1=Path('/Users/wio/work/169-sglang-deepseek-v41/evidence/001-npu-preflight.txt').read_text()
h2=Path('/Users/wio/work/169-sglang-deepseek-v41/evidence/009-npu-occupancy-current.txt').read_text()
raw=json.loads((here.parent/'ownership-check-20261029.raw.json').read_text())
ref=remote_guard.build_empirical_reference(raw['smi']['stdout'],h1,h2,set(raw['own_sorted']))
assert ref['observed_spread_margin_mb']==48 and len(ref['envelope_mb'])==16
release=remote_guard.verified_release(real,ref)
assert not release['satisfies_policy'] and release['reason']=='UNATTRIBUTED_HBM_ABOVE_REFERENCE' and len(release['excess_mb'])==16
assert not release['pids'] and max(x[0] for x in release['hbm'].values())==63996
assert remote_guard.verified_release(h1,ref)['satisfies_policy'] and remote_guard.verified_release(h2,ref)['satisfies_policy']
assert not remote_guard.verified_release(h1,None)['satisfies_policy']
snapshots=[]
def get_snapshot():snapshots.append(1);return h1
assert remote_guard.wait_for_empirical_release(get_snapshot,ref,timeout=1,interval=0)['satisfies_policy'] and len(snapshots)==2
# Execute only the generated save() function against a task-local temp folder.
t=ast.parse((here/'start_binding_recapture.py').read_text());startremote=next(ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='REMOTE' for x in n.targets))
t=ast.parse(startremote);savefunc=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='save')
with tempfile.TemporaryDirectory(prefix='mock-atomic-',dir=here) as d:
    def fake_cmd(a):
        assert a[:4]==['docker','exec','stage','python3'];exec(a[-1],{})
    env={'json':json,'WORK':d,'REL':'','RUN_ID':RUN_ID,'STAGE':'stage','cmd':fake_cmd}
    exec(compile(ast.Module(body=[savefunc],type_ignores=[]),'save-test','exec'),env)
    ref_payload={'envelope_mb':{'0:0':123}}
    obj={'state':'PREPARED','diagnostic_id':DIAG_ID,'baseline_id':BASE_ID,'binding_run_id':RUN_ID,'hbm_reference':ref_payload}
    env['save'](obj);obj['state']='STARTED';env['save'](obj)
    assert json.loads((Path(d)/'launch.json').read_text())['state']=='STARTED'
    identity=Path(d)/('identity-'+RUN_ID+'.json')
    assert json.loads(identity.read_text())['state']=='STARTED'
    verify=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='verify' for x in n.targets))
    env.update(reference=ref_payload,baseline={'Id':BASE_ID},diag_id=DIAG_ID)
    exec(compile(ast.Module(body=[verify],type_ignores=[]),'verify-test','exec'),env);exec(env['verify'],{})
    identity.write_text('{}')
    try:exec(env['verify'],{})
    except AssertionError:pass
    else:raise AssertionError('corrupted recovery reference accepted')
supervisor_text=(here/'supervise_binding_recapture.py').read_text()
recover_text=(here/'recover_binding.py').read_text()
watchdog_text=(here/'watchdog_binding.py').read_text()
assert 'ssh(RESTORE,timeout=400)' in supervisor_text and 'ssh(REMOTE,400)' in recover_text
assert 'timeout=3000' in watchdog_text and 'Baseline recovery deferred' in supervisor_text
print(json.dumps({'status':'PASS','scope':'CPU-only; no SSH or NPU','parser':'full/malformed/incomplete passed','fence':'two processes serialized','no_identity_no_diag':no_identity,'no_identity_stopped':missing_stopped,'identity_backup_and_exact_stop':backup,'start_capture_restore_fence':'present','atomic_identity_backup':'PASS','two_reference_readback':'PASS','recovery_timeout_budgets':'PASS','real_empty_pid_high_hbm':'refused','historical_two_snapshot':'PASS','data_derived_margin_mb':48},indent=2))
