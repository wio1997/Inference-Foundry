"""H6/H5 and target FULL retained; RoPE layout correctness only; no timing claim."""
import hashlib
import importlib.util
import json
import os
import math
import statistics
import re
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "functional_plan.json").read_text())
old = ROOT.parent / "GLM-RUN-0253/pd_event_compare.py"
spec = importlib.util.spec_from_file_location("accepted_PD_client", old)
oldclient = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oldclient)
x, m = oldclient.x, oldclient.m
oldclient.ROOT = x.ROOT = m.ROOT = ROOT
oldclient.PLAN = x.PLAN = PLAN
fetch = x.fetch


def native_maps(roles, expected):
    """Check the actual loaded library in every owned worker, without imports."""
    rows = []
    for pid in roles["device_owners"]:
        stat = Path("/proc/%d/stat" % pid)
        assert stat.read_text().rsplit(")", 1)[1].split()[19] == roles["worker_start_ticks"][str(pid)]
        paths = {line.split(maxsplit=5)[5] for line in
                 Path("/proc/%d/maps" % pid).read_text().splitlines()
                 if len(line.split(maxsplit=5)) == 6
                 and line.split(maxsplit=5)[5].endswith("/libtorch_npu.so")}
        assert paths == {expected}, (pid, paths)
        rows.append(dict(pid=pid, library=expected))
    return rows

def spec_counters():
    """Read existing scheduler-derived counters; no instrumentation or NPU call."""
    status, raw = fetch("http://172.16.10.167:9900/metrics")
    assert status == 200
    pattern = re.compile(r'^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens|accepted_tokens_per_pos)_total)(\{[^}]+\})\s+([^\s]+)$')
    values = {}
    lines = []
    names = set()
    for line in raw.decode().splitlines():
        match = pattern.fullmatch(line)
        if match:
            name, labels, value = match.groups()
            assert 'engine="0"' in labels and 'model_name="glm-53"' in labels
            value = float(value)
            assert math.isfinite(value) and value >= 0 and value.is_integer()
            key = name + labels
            if name.endswith('_per_pos_total'):
                assert 'position="0"' in labels
            assert key not in values
            values[key] = value
            names.add(name)
            lines.append(line)
    expected = {"vllm:spec_decode_num_%s_total" % name for name in
                ("drafts", "draft_tokens", "accepted_tokens", "accepted_tokens_per_pos")}
    assert names == expected and len(values) == 4, (names, values)
    return dict(counters=values, raw_lines=lines)

def request(label, complete=False):
    before = spec_counters()
    m.write(label + "_metrics_before.json", before)
    result = x.request(label, complete)
    after = spec_counters()
    m.write(label + "_metrics_after.json", after)
    assert before["counters"].keys() == after["counters"].keys()
    delta = {key: after["counters"][key] - value for key, value in before["counters"].items()}
    normalized = {key.split("{", 1)[0]: int(value) for key, value in delta.items()}
    drafts = normalized["vllm:spec_decode_num_drafts_total"]
    draft_tokens = normalized["vllm:spec_decode_num_draft_tokens_total"]
    accepted = normalized["vllm:spec_decode_num_accepted_tokens_total"]
    accepted_per_pos = normalized["vllm:spec_decode_num_accepted_tokens_per_pos_total"]
    checks = dict(nonnegative_monotonic=all(value >= 0 and value.is_integer() for value in delta.values()),
                  valid_K1_counters=0 <= accepted == accepted_per_pos <= draft_tokens <= drafts)
    signature = dict(num_drafts=drafts, num_draft_tokens=draft_tokens,
                     num_accepted_tokens=accepted, accepted_position0=accepted_per_pos, invalid_draft_tokens=drafts - draft_tokens)
    m.write(label + "_workload.json", dict(before=before, after=after, delta=delta,
        signature=signature, checks=checks, SSE_output_groups=result["chunks"],
        counter_semantics="Scheduler generated-token acceptance before stop truncation; not per-step ordered trace"))
    if not all(checks.values()):
        raise RuntimeError("MEASUREMENT_INVALID: scheduler counters; preserved raw metrics")
    if complete:
        expected = PLAN["complete_golden"]
        assert result["semantic_accepted"] and result["finish_reason"] == "stop"
        assert result["prompt_tokens"] == expected["prompt_tokens"]
        assert result["token_ids"] == expected["token_ids"]
        assert result["final_content"] == expected["final_content"]
    else:
        assert result["prompt_tokens"] == 2334
    # Legitimate grouping/acceptance variation is evidence, not a correctness failure.
    result = dict(result, workload_signature=signature,
                  SSE_grouping_is_engine_step_trace=False)
    return result

def rpc(host, action, *values):
    argv = ["/usr/bin/python3", str(Path(__file__).resolve()), action, host, *map(str, values)]
    if host == "167":
        argv = ["ssh", "-o", "BatchMode=yes", "root@172.16.10.167", shlex.join(argv)]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (action, result.stderr[-3000:])
    return json.loads(result.stdout)

def epoch_root():
    name = json.loads((ROOT / "active_epoch.json").read_text())["epoch"]
    assert name in ("candidate", "baseline_recovery")
    return ROOT / "epochs" / name

def verify_artifact():
    candidate = PLAN["native_candidate"]
    library = Path(candidate["library"])
    assert hashlib.sha256(library.read_bytes()).hexdigest() == candidate["sha256"]
    audit_path = Path(candidate["CPU_audit_path"])
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == candidate["CPU_audit_sha256"]
    audit = json.loads(audit_path.read_text())
    assert audit["CPU_import_passed"] and not audit["NPU_initialized"]
    assert audit["library_sha256"] == candidate["sha256"]
    assert all(audit["V4_symbols"].values())
    assert audit["schemas"] == json.loads(Path(candidate["stock_audit_path"]).read_text())["schemas"]
    loader_path = Path(candidate["loader_identity_path"])
    assert hashlib.sha256(loader_path.read_bytes()).hexdigest() == candidate["loader_identity_sha256"]
    loader = json.loads(loader_path.read_text())
    assert loader["native_sha256"] == candidate["sha256"]
    assert loader["Python_backend"] == candidate["Python_backend"]
    extension = Path(candidate["Python_backend"]) / "torch_npu" / PLAN["python_extension"]["name"]
    assert hashlib.sha256(extension.read_bytes()).hexdigest() == PLAN["python_extension"]["sha256"]
    return dict(artifact_gate=True, CPU_only=True)

def ready():
    deadline = time.monotonic() + 1800
    while True:
        row = rpc("167", "poll")
        m.write("readiness.json", row)
        if row["failure"]:
            raise RuntimeError("Owned D initialization failed")
        if row["ready"]:
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Owned D readiness")
        time.sleep(10)

def guard(host,fresh=False):
    if host=='167' and fresh:m.ROOT=epoch_root();root=m.new_owner(host)
    else:root=PLAN['resident_roles'][host]
    assert root and m.live(root)
    members=m.tree(root['pid'],m.catalogue());owners,_=m.device_owners()
    assert len(owners)==16 and owners.issubset({p['pid'] for p in members})
    m.assert_idle(PLAN['ports'][host]);status,_=fetch('http://127.0.0.1:%d/health'%PLAN['ports'][host]);assert status==200
    namespaces={str(pid):int(next(l for l in Path('/proc/%d/status'%pid).read_text().splitlines() if l.startswith('NSpid:')).split()[-1]) for pid in owners}
    if host=='167':
        if fresh:
            protected=dict(PLAN['protected_source_hashes'])
            epoch=json.loads((ROOT/'active_epoch.json').read_text())['epoch']
            protected[PLAN['target_source']]=PLAN['shim_sha256'] if epoch=='candidate' else PLAN['original_sha256']
            protected[PLAN['runner_target']]=PLAN['runner_patch_sha256'] if epoch=='candidate' else PLAN['runner_original_sha256']
            protected[PLAN['proposer_target']]=PLAN['proposer_shim_sha256']
            protected[PLAN['runner_target']]=PLAN['runner_patch_sha256']
            protected[PLAN['rotary_target']]=PLAN['rotary_shim_sha256'] if epoch=='candidate' else PLAN['rotary_original_sha256']
        else:
            protected=dict(PLAN['before_source_hashes'])
        code="from pathlib import Path;import hashlib,json;print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in %r}))"%list(protected)
        got=json.loads(subprocess.check_output(['docker','exec','glm52-single',m.PYTHON,'-c',code],text=True));assert got==protected,(got,protected)
        assert (Path(PLAN['previous_event_root'])/'event_mode.bin').read_bytes()==b'\x01'
        assert (Path(PLAN['previous_projection_root'])/'projection_mode.bin').read_bytes()==b'\x00'
        assert (Path(PLAN['previous_projection_root'])/'projection_gate.bin').read_bytes()==b'\x00'
        if fresh:assert (runtime_witness_root()/'stream_order_mode.bin').read_bytes()==b'\x00'
        assert (ROOT/'mc2_mode.bin' if fresh else Path(PLAN['previous_MC2_root'])/'mc2_mode.bin').read_bytes()==b'\x01'
    starts={str(p['pid']):p['start_ticks'] for p in members if p['pid'] in owners}
    return dict(root=root,device_owners=sorted(owners),worker_start_ticks=starts,worker_namespace_pids=namespaces,health=status,idle=True)

def runtime_witness_root():
    return ROOT if json.loads((ROOT/'active_epoch.json').read_text())['epoch']=='candidate' else epoch_root()

def launch(name):
    assert name in ('candidate','baseline_recovery') and not m.device_owners()[0]
    verify_artifact();target=ROOT/'epochs'/name
    assert target.is_dir() and not (target/'native_167.log').exists()
    assert (ROOT/'mc2_mode.bin').read_bytes()==b'\x01'
    (ROOT/'active_epoch.json').write_text(json.dumps(dict(epoch=name))+'\n')
    env=json.loads(Path(PLAN['baseline_env']['167']).read_text())['environment'];env.update(PLAN['environment']['167'])
    env['PYTHONPATH']=':'.join(p for p in env.get('PYTHONPATH','').split(':') if '/glm52-pd/deploy/plugins/' not in p)
    env['LD_LIBRARY_PATH']=str(Path(PLAN['native_candidate']['library']).parent)+':'+env.get('LD_LIBRARY_PATH','')
    env['GLM_MC2_CAPABILITY_MODE_FILE']=str(ROOT/'mc2_mode.bin')
    env['PYTHONPATH']=PLAN['native_candidate']['Python_backend']+':'+env.get('PYTHONPATH','')
    env['PYTHONDONTWRITEBYTECODE']='1';env['GLM_EVENT_WITNESS_SCOPE']='research279'+name
    env['VLLM_USE_BREAKABLE_CUDAGRAPH']='1'
    env.pop('GLM_GRAPH_DIAGNOSTIC_ROOT',None);env.pop('GLM_PD_ID_DIAGNOSTIC_ROOT',None)
    env.pop('GLM_EARLY_MTP_DIAGNOSTIC_ROOT',None)
    env.pop('GLM_MTP_GRAPH_DIAGNOSTIC_ROOT',None)
    env.pop('VLLM_ASCEND_GLM_K1_MTP_GRAPH',None)
    if name in ('candidate','baseline_recovery'):
        witness_root=ROOT if name=='candidate' else target
        if name=='baseline_recovery':
            (witness_root/'witnesses').mkdir(exist_ok=True)
            for filename in ('stream_order_mode.bin','mtp_graph_mode.bin'):
                p=witness_root/filename;assert not p.exists();p.write_bytes(b'\x00')
        env['GLM_STREAM_ORDER_DIAGNOSTIC_ROOT']=str(witness_root)
        env['GLM_MTP_GRAPH_DIAGNOSTIC_ROOT']=str(witness_root)
        env['VLLM_ASCEND_GLM_K1_MTP_GRAPH']='1'
    env.pop('GLM_ROPE_LAYOUT_DIAGNOSTIC_ROOT',None)
    if name=='candidate':env['GLM_ROPE_LAYOUT_DIAGNOSTIC_ROOT']=str(ROOT)
    argv=['docker','exec','-d']
    for key,value in sorted(env.items()):argv+=['-e',key+'='+value]
    argv+=['glm52-single',m.PYTHON,str(Path(__file__).resolve()),'native','167',name]
    subprocess.run(argv,check=True,timeout=30,capture_output=True)
    (target/'launch.json').write_text(json.dumps(dict(epoch=name,launch_exit=0,H6_retained=True,H5_retained=True,environment_not_exported=True))+'\n')
    return dict(epoch=name,launch_exit=0)

def witness_stack(mode):
    witness_root=runtime_witness_root();roles=guard('167',True);target=ROOT/'witness_roles.json';target.write_text(json.dumps(roles,indent=2)+'\n');c=PLAN['native_candidate']
    cmd=['/usr/bin/python3',str(ROOT/'witness_mc2_native.py'),'--library',c['library'],'--sha256',c['sha256'],'--roles',str(target),'--mode-file',str(ROOT/'mc2_mode.bin'),'--mode','1']
    res=subprocess.run(cmd,capture_output=True,text=True,timeout=90);assert res.returncode==0,res.stderr[-3000:];native=json.loads(res.stdout)
    epoch=json.loads((ROOT/'active_epoch.json').read_text())['epoch'];previous=Path(PLAN['previous_event_root']);event=[json.loads(p.read_text()) for p in sorted((previous/'witnesses').glob('research279%s_mode1_rank*.json'%epoch))]
    assert len(event)==16 and {z['rank'] for z in event}==set(range(16))
    assert {z['pid'] for z in event}==set(roles['worker_namespace_pids'].values())
    assert all(z['mode']==1 and not z['configured_overlap'] and z['returned_none'] and z['candidate_utils_sha256']==PLAN['retained_H5_helper_candidate_sha256'] for z in event)
    assert (witness_root/'stream_order_mode.bin').read_bytes()==b'\x00'
    assert (witness_root/'mtp_graph_mode.bin').read_bytes()==bytes([mode])
    rows=[];mtp=[]
    if epoch in ('candidate','baseline_recovery'):
        capabilities=[json.loads(p.read_text()) for p in (witness_root/'witnesses').glob('h11_capability_rank*.json')]
        assert len(capabilities)==16 and {z['rank'] for z in capabilities}==set(range(16))
        assert {z['pid'] for z in capabilities}==set(roles['worker_namespace_pids'].values())
        assert all(z['eligible'] and z['use_cuda_graph'] and z['offloader']=='NoopOffloader' for z in capabilities)
        rows=[json.loads(p.read_text()) for p in sorted((witness_root/'witnesses').glob('mode0_rank*.json'))]
        assert len(rows)==16 and {z['rank'] for z in rows}==set(range(16))
        assert all(z['mode']==0 and not z['eligible'] and z['target'] and not z['skipped_host_sync'] and 'FULL' in z['runtime_mode'] and z['capture_num_graphs']>=1 and z['eager_breaks']==0 and z['num_tokens']==2 and not z['target_enforce_eager'] and not z['MTP_enforce_eager'] and z['capture_sizes']==[2] for z in rows)
        paths=list((witness_root/'witnesses').glob('h11_mode%d_transition*_rank*.json'%mode));assert paths, 'NOT_APPLICABLE: no scoped MTP pointer witnesses';latest=max(json.loads(path.read_text())['transition'] for path in paths)
        mtp=[json.loads(path.read_text()) for path in paths if json.loads(path.read_text())['transition']==latest]
        assert len(mtp)==16 and {z['rank'] for z in mtp}==set(range(16))
        assert {z['pid'] for z in mtp}==set(roles['worker_namespace_pids'].values())
        assert all(z['mode']==mode and z['proposer_sha256']==PLAN['proposer_shim_sha256'] and not z['enforce_eager'] and z['use_cuda_graph'] and not z['H9_opt_in'] for z in mtp)
        if mode:
            assert all(all(v['actual_replay'] and v['contract_nonempty'] and v['matches_capture'] and v['owned_output'] and v['output_shape']==[1,1] and v['num_prefills']==0 and v['greedy'] and v['async_scheduling'] for v in z['rows']) for z in mtp), 'NOT_APPLICABLE: all-rank actual MTP graph contract/replay not established'
            assert all(z['rows'][0]['positions']!=z['rows'][1]['positions'] and z['rows'][0]['seq']!=z['rows'][1]['seq'] for z in mtp)
    else:assert mode==0
    return dict(H11_rows=mtp,H11_mode=mode,H9_rows=rows,H9_mode=0,H6=native,H5_rows=event,H6_mode=1,H5_event_mode=1,H10=False,H8=False,same_worker_identity=roles)


def switch(mode):
    guard('167',True);assert mode in (0,1)
    assert (ROOT/'mtp_graph_mode.bin').read_bytes()==b'\x00'
    fd=os.open(ROOT/'rope_layout_mode.bin',os.O_WRONLY)
    try:assert os.pwrite(fd,bytes([mode,1]),0)==2;os.fsync(fd)
    finally:os.close(fd)
    return dict(H12_mode=mode,H11_mode=0,H6_mode=1,H5_mode=1,idle=True)


def transfer_gate(label,request_id):
    guard('167',True)
    assert re.fullmatch(r'[\w-]+',request_id),request_id
    raw=(epoch_root()/'native_167.log').read_text(errors='replace')
    matches=[line for line in raw.splitlines() if 'KV cache transfer for request '+request_id+' took ' in line]
    ranks=[int(re.search(r'local_device_id (\d+)',line).group(1)) for line in matches]
    assert len(ranks)==16 and set(ranks)==set(range(16)),(label,ranks)
    assert not any(request_id in line and ('Mooncake transfer failed' in line or 'Failed to transfer KV' in line) for line in raw.splitlines())
    return dict(label=label,remote_request_id=request_id,all16_native_transfer_success=True,rows=matches,source='existing native INFO logs; no transfer observer')

def install(kind='candidate'):
    assert kind in ('candidate','original') and not m.device_owners()[0]
    target=PLAN['rotary_target'];artifact=ROOT/('shim' if kind=='candidate' else 'original')/'rotary_embedding.py'
    wanted=PLAN['rotary_shim_sha256'] if kind=='candidate' else PLAN['rotary_original_sha256']
    code="from pathlib import Path;import hashlib,os,json;p=Path(%r);b=Path(%r).read_bytes();assert hashlib.sha256(p.read_bytes()).hexdigest() in %r;assert hashlib.sha256(b).hexdigest()==%r;t=p.with_name(p.name+'.run279.tmp');assert not t.exists();t.write_bytes(b);os.replace(t,p);print(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()}))"%(target,str(artifact),{PLAN['rotary_original_sha256'],PLAN['rotary_shim_sha256']},wanted)
    return json.loads(subprocess.check_output(['docker','exec','glm52-single',m.PYTHON,'-c',code],text=True))




def observe_off():
    guard('167',True)
    fd=os.open(ROOT/'rope_layout_mode.bin',os.O_WRONLY)
    try:assert os.pwrite(fd,b'\x00',1)==1;os.fsync(fd)
    finally:os.close(fd)
    return dict(heavy_observer_off=True)

def witness(mode):
    base=witness_stack(0)
    if json.loads((ROOT/'active_epoch.json').read_text())['epoch']=='baseline_recovery':return base
    paths=list((ROOT/'witnesses').glob('h12_mode%d_transition*_rank*.json'%mode));assert paths
    latest=max(json.loads(p.read_text())['transition'] for p in paths)
    rows=[json.loads(p.read_text()) for p in paths if json.loads(p.read_text())['transition']==latest]
    roles=base['same_worker_identity'];assert len(rows)==16 and {z['rank'] for z in rows}==set(range(16))
    assert {z['pid'] for z in rows}==set(roles['worker_namespace_pids'].values())
    assert all(z['source_sha256']==PLAN['rotary_shim_sha256'] and z['mode']==mode and len(z['rows'])==4 for z in rows)
    for z in rows:
        assert all(t['shape']==[1048576,64] and t['stride']==[128,1] and t['dtype']=='torch.bfloat16' and t['format']==2 for t in z['baseline'])
        assert all(t['shape']==[1048576,64] and t['stride']==[64,1] and t['dtype']=='torch.bfloat16' and t['is_contiguous'] and t['format']==2 for t in z['candidate'])
        assert all(v['use_cache'] for v in z['rows']), 'current MTP ordinary SFA builder must preserve shared persistent outputs'
        assert all([v['shape'] for v in r['output']]==[[2,1,1,64],[2,1,1,64]] for r in z['rows'])
        assert all(all(a['data_ptr']==b['data_ptr'] for a,b in zip(r['output'],r['persistent'])) for r in z['rows'] if r['use_cache'])
    return dict(base,H12_rows=rows,H12_mode=mode)

def workflow():
    owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
    assert owner['run_id']==PLAN['run_id'] and owner['status']=='running' and m.live(owner['owner'])
    retired=False;correct=False;results=[];main_error=None
    try:
        rpc('167','directories');rpc('167','artifact');m.write('guards_before.json',{h:rpc(h,'guard') for h in ('166','167')})
        retired=True;rpc('167','retire_old');rpc('167','install');rpc('167','launch','candidate');ready()
        for mode in (0,1):
            rpc('166','guard');rpc('167','switch',mode)
            label='correctness_mode%d_short'%mode;results.append(request(label))
            rid=json.loads((ROOT/(label+'_P.raw')).read_text())['kv_transfer_params']['remote_request_id'];m.write(label+'_transfer.json',rpc('167','transfergate',label,rid))
            m.write('correctness_mode%d_witness.json'%mode,rpc('167','witness',mode));rpc('167','observeoff')
        label='correctness_mode1_complete';results.append(request(label,True))
        rid=json.loads((ROOT/(label+'_P.raw')).read_text())['kv_transfer_params']['remote_request_id'];m.write(label+'_transfer.json',rpc('167','transfergate',label,rid))
        m.write('complete_witness.json',rpc('167','witness',1));correct=True
    except BaseException as error:
        main_error=repr(error);m.write('failure.json',dict(error=main_error,results=results));raise
    finally:
        if retired:
            terminal_error=None;recovered=False;terminal_ok=False;witness=None;guards=None
            try:
                rpc('167','guardfresh');rpc('166','guard')
                assert rpc('167','epoch')['epoch']=='candidate'
                rpc('167','switch',0);request('retained_mode_warm');witness=rpc('167','witness',0);rpc('167','observeoff')
                guards={h:rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')};terminal_ok=True
            except BaseException as error:
                terminal_error=repr(error);m.write('terminal_failure.json',dict(error=terminal_error));recovered=True
                # One recovery for initialization, request, witness, or final-guard failure.
                try:
                    m.write('baseline_recovery_cleanup.json',rpc('167','recoverycleanup'));rpc('167','restore');rpc('167','recoverymode');rpc('167','launch','baseline_recovery');ready()
                    request('H6_H5_recovery_warm');witness=rpc('167','witness',0)
                    guards={h:rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')};terminal_ok=True
                except BaseException as recovery_error:
                    m.write('recovery_failure.json',dict(error=repr(recovery_error),first_error=terminal_error));raise
            passed=correct and terminal_ok and not recovered and not main_error
            m.write('retained_witness.json',witness);m.write('guards_after.json',guards)
            m.write('retained_stack.json',dict(H6=True,H5=True,H11=False,H12=False,H12_mode=0,target_FULL=True,pure_KV_API_repairs=True,diagnostic_completed=passed,recovery_used=recovered,terminal_verified=terminal_ok,Current=None,formal_product_KEEP=False))
            m.write('diagnostic_status.json',dict(status='completed' if passed else 'failed',results=results,native_correctness=passed,request_checks_passed=correct,terminal_verified=terminal_ok,recovery_used=recovered,error=main_error or terminal_error,H12_performance_gain=None,Current=None))
            if correct and terminal_error:raise RuntimeError('terminal failure invalidates candidate correctness; one recovery retained H6/H5')



def action(name,host,values):
    if name=='directories':
        (ROOT/'witnesses').mkdir(parents=True,exist_ok=True);assert (ROOT/'witnesses').is_dir();return dict(witness_dir=True)
    if name=='guard':return guard(host)
    if name=='guardfresh':return guard(host,True)
    if name=='artifact':return verify_artifact()
    if name=='install':return install()
    if name=='restore':return install('original')
    if name=='epoch':return json.loads((ROOT/'active_epoch.json').read_text())
    if name=='recoverymode':
        assert not m.device_owners()[0];(ROOT/'stream_order_mode.bin').write_bytes(b'\x00');(ROOT/'mtp_graph_mode.bin').write_bytes(b'\x00');(ROOT/'rope_layout_mode.bin').write_bytes(b'\x00\x00');return dict(H6=True,H5=True,H10=False,H9=False)
    if name=='retire_old':
        roles=guard('167');m.stop_tree(m.tree(roles['root']['pid'],m.catalogue()),roles['root']);assert not m.device_owners()[0];return dict(owned_D_retired=True)
    if name=='launch':return launch(values[0])
    if name=='native':
        assert values[0] in ('candidate','baseline_recovery');m.ROOT=ROOT/'epochs'/values[0]
        native_plan=dict(PLAN)
        if values[0]=='baseline_recovery':native_plan['native_args']=PLAN['baseline_native_args']
        return m.native('167',native_plan)
    if name=='poll':m.ROOT=epoch_root();return m.poll(host,PLAN)
    if name=='recoverycleanup':
        if (ROOT/'active_epoch.json').exists():m.ROOT=epoch_root();m.cleanup(host,PLAN)
        else:
            root=PLAN['resident_roles'][host]
            if m.live(root):m.stop_tree(m.tree(root['pid'],m.catalogue()),root)
        assert not m.device_owners()[0];return dict(owned_D_cleaned=True)
    if name=='switch':return switch(int(values[0]))
    if name=='witness':return witness(int(values[0]))
    if name=='observeoff':return observe_off()
    if name=='transfergate':return transfer_gate(values[0],values[1])
    raise ValueError(name)
def decide(results):
    raise RuntimeError('Correctness diagnostic has no performance decision')


if __name__ == '__main__':
    if sys.argv[1]=='workflow':workflow()
    else:print(json.dumps(action(sys.argv[1],sys.argv[2],sys.argv[3:])))
