"""H6+H5 retained throughout; one-reload stack versus stack+H8, then keep validated research stack."""
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



def guard(host,fresh=False):
    if host=='167' and fresh:m.ROOT=epoch_root();root=m.new_owner(host)
    else:root=PLAN['resident_roles'][host]
    assert root and m.live(root)
    members=m.tree(root['pid'],m.catalogue());owners,_=m.device_owners()
    assert len(owners)==16 and owners.issubset({p['pid'] for p in members})
    m.assert_idle(PLAN['ports'][host]);status,_=fetch('http://127.0.0.1:%d/health'%PLAN['ports'][host]);assert status==200
    namespaces={str(pid):int(next(l for l in Path('/proc/%d/status'%pid).read_text().splitlines() if l.startswith('NSpid:')).split()[-1]) for pid in owners}
    if host=='167':
        protected=dict(PLAN['protected_source_hashes'])
        protected[PLAN['target_source']]=PLAN['shim_sha256'] if fresh else PLAN['original_sha256']
        code="from pathlib import Path;import hashlib,json;print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in %r}))"%list(protected)
        got=json.loads(subprocess.check_output(['docker','exec','glm52-single',m.PYTHON,'-c',code],text=True));assert got==protected
        assert (Path(PLAN['previous_event_root'])/'event_mode.bin').read_bytes()==b'\x01'
        if fresh:assert (ROOT/'mc2_mode.bin').read_bytes()==b'\x01'
    starts={str(p['pid']):p['start_ticks'] for p in members if p['pid'] in owners}
    return dict(root=root,device_owners=sorted(owners),worker_start_ticks=starts,worker_namespace_pids=namespaces,health=status,idle=True)





def switch(mode):
    guard('167',True);assert mode in (0,1)
    assert (ROOT/'mc2_mode.bin').read_bytes()==b'\x01'
    assert (Path(PLAN['previous_event_root'])/'event_mode.bin').read_bytes()==b'\x01'
    fd=os.open(ROOT/'projection_mode.bin',os.O_WRONLY)
    try: assert os.pwrite(fd,bytes([mode]),0)==1;os.fsync(fd)
    finally: os.close(fd)
    return dict(projection_mode=mode,MC2_mode=1,H5_event_mode=1,idle=True)




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


def launch(name):
    assert name in ('candidate','baseline_recovery') and not m.device_owners()[0]
    verify_artifact()
    target=ROOT/'epochs'/name
    assert target.is_dir() and not (target/'native_167.log').exists()
    assert (ROOT/'mc2_mode.bin').read_bytes()==b'\x01'
    (ROOT/'active_epoch.json').write_text(json.dumps(dict(epoch=name))+'\n')
    env=json.loads(Path(PLAN['baseline_env']['167']).read_text())['environment']
    env.update(PLAN['environment']['167'])
    env['PYTHONPATH']=':'.join(p for p in env.get('PYTHONPATH','').split(':') if '/glm52-pd/deploy/plugins/' not in p)
    env['LD_LIBRARY_PATH']=str(Path(PLAN['native_candidate']['library']).parent)+':'+env.get('LD_LIBRARY_PATH','')
    env['GLM_MC2_CAPABILITY_MODE_FILE']=str(ROOT/'mc2_mode.bin')
    env['PYTHONPATH']=PLAN['native_candidate']['Python_backend']+':'+env.get('PYTHONPATH','')
    env['PYTHONDONTWRITEBYTECODE']='1'
    env['GLM_EVENT_WITNESS_SCOPE']='research262'
    argv=['docker','exec','-d']
    for key,value in sorted(env.items()):argv+=['-e',key+'='+value]
    argv+=['glm52-single',m.PYTHON,str(Path(__file__).resolve()),'native','167',name]
    subprocess.run(argv,check=True,timeout=30,capture_output=True)
    (target/'launch.json').write_text(json.dumps(dict(epoch=name,launch_exit=0,H6_retained=True,environment_not_exported=True))+'\n')
    return dict(epoch=name,launch_exit=0)



def witness(mode):
    roles=guard('167',True);target=ROOT/('witness_roles_projection%d.json'%mode);target.write_text(json.dumps(roles,indent=2)+'\n')
    c=PLAN['native_candidate'];cmd=['/usr/bin/python3',str(ROOT/'witness_mc2_native.py'),'--library',c['library'],'--sha256',c['sha256'],'--roles',str(target),'--mode-file',str(ROOT/'mc2_mode.bin'),'--mode','1']
    res=subprocess.run(cmd,capture_output=True,text=True,timeout=90);assert res.returncode==0,res.stderr[-3000:];native=json.loads(res.stdout)
    assert (ROOT/'projection_mode.bin').read_bytes()==bytes([mode])
    previous=Path(PLAN['previous_event_root']);event=[json.loads(p.read_text()) for p in sorted((previous/'witnesses').glob('research262_mode1_rank*.json'))]
    assert len(event)==16 and {z['rank'] for z in event}==set(range(16))
    assert {z['pid'] for z in event}==set(roles['worker_namespace_pids'].values())
    assert all(z['mode']==1 and not z['configured_overlap'] and z['returned_none'] and z['candidate_utils_sha256']==PLAN['retained_H5_helper_candidate_sha256'] for z in event)
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'witnesses').glob('mode%d_rank*.json'%mode))]
    assert len(rows)==16 and {z['rank'] for z in rows}==set(range(16))
    assert {z['pid'] for z in rows}==set(roles['worker_namespace_pids'].values())
    assert all(z['mode']==mode and z['forward_local'] and z['candidate_sha256']==PLAN['candidate_sha256'] for z in rows)
    checks=[]
    if mode:
        for rank in range(16):
            raw=(ROOT/'witnesses'/('bytes_rank%d.jsonl'%rank)).read_text()
            xs=[json.loads(line) for line in raw.splitlines()]
            assert all(z['rank']==rank and z['pid']==next(row['pid'] for row in rows if row['rank']==rank) and z['exact_reference_bytes_equal'] and z['candidate_sha256']==PLAN['candidate_sha256'] for z in xs)
            by_size={t:{z['layer'] for z in xs if z['x_shape'][0]==t} for t in PLAN['expected_native_token_rows']}
            assert all(len(names)==PLAN['expected_indexer_layers'] for names in by_size.values()),by_size
            checks.append(dict(rank=rank,rows=xs,token_rows_to_layers={str(k):sorted(v) for k,v in by_size.items()}))
    return dict(H6=native,H5_rows=event,H8_rows=rows,H8_byte_checks=checks,projection_mode=mode,same_worker_identity=roles,H6_mode=1,H5_event_mode=1)





def stock_witness():
    roles = guard("167", True)
    stock = PLAN["native_stock"]
    code = "from pathlib import Path;import hashlib;print(hashlib.sha256(Path(%r).read_bytes()).hexdigest())" % stock["library"]
    digest = subprocess.check_output(["docker", "exec", "glm52-single", m.PYTHON, "-c", code], text=True).strip()
    assert digest == stock["sha256"]
    return dict(all_rank_original_library=True, library_sha256=digest,
                rows=native_maps(roles, stock["library"]))


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


def workflow():
    owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text())
    assert owner['run_id']==PLAN['run_id'] and owner['status']=='running' and m.live(owner['owner'])
    rpc('167','artifact')
    retirement_attempted=False;results=[];retained=0;completed=False
    try:
        m.write('guards_before.json',{h:rpc(h,'guard') for h in ('166','167')})
        retirement_attempted=True
        rpc('167','retire_old')
        rpc('167','install')
        rpc('167','launch','candidate');ready()
        m.write('metrics_gate.json',spec_counters())
        for mode in (0,1):
            rpc('166','guard');rpc('167','switch',mode)
            request('correctness_mode%d_short'%mode)
            request('correctness_mode%d_complete'%mode,True)
            m.write('correctness_mode%d_witness.json'%mode,rpc('167','witness',mode))
        rpc('167','gateoff')
        for mode,name in ((0,'A1'),(1,'B1'),(0,'A2'),(1,'B2')):
            rpc('166','guard');rpc('167','switch',mode)
            request(name+'_warm')
            m.write(name+'_witness.json',rpc('167','witness',mode))
            for index in range(2):results.append(request(name+'_measure%d'%index))
            results.append(request(name+'_complete',True))
            m.write('comparison_results.json',results)
        decision=decide(results);retained=int(decision['research_positive'])
        m.write('research_decision.json',decision);completed=True
        m.write('comparison_status.json',dict(status='completed',results=results,native_correctness=True,H6_retained=True,Current=None,formal_SLA=False,full_API_contract=False))
    except BaseException as error:
        m.write('failure.json',dict(error=repr(error),results=results));raise
    finally:
        if retirement_attempted:
            # Successful existing patches stay resident. Failure only rolls back H8.
            try:
                rpc('167','guardfresh')
            except Exception:
                m.write('baseline_recovery_cleanup.json',rpc('167','recoverycleanup'))
                rpc('167','install')
                rpc('167','recoverymode')
                rpc('167','launch','baseline_recovery');ready()
                request('H6_H5_recovery_warm')
                retained=0
            rpc('166','guard');rpc('167','switch',retained)
            if not completed: request('H6_H5_failure_baseline_warm')
            m.write('retained_witness.json',rpc('167','witness',retained))
            m.write('guards_after.json',{h:rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')})
            m.write('retained_stack.json',dict(H6=True,H5=True,H8=bool(retained),projection_mode=retained,event_mode=1,H4=False,MC2_mode=1,comparison_completed=completed,diagnostic_selectors_resident=True,Current=None,formal_product_KEEP=False))



def action(name,host,values):
    if name=='guard':return guard(host)
    if name=='guardfresh':return guard(host,True)
    if name=='artifact':return verify_artifact()
    if name=='install':return install()
    if name=='recoverymode':
        assert not m.device_owners()[0]
        assert (ROOT/'mc2_mode.bin').read_bytes()==b'\x01'
        (ROOT/'projection_mode.bin').write_bytes(b'\x00');return dict(H6=True,H5=True,H8=False)
    if name=='retire_old':
        roles=guard('167');m.stop_tree(m.tree(roles['root']['pid'],m.catalogue()),roles['root'])
        assert not m.device_owners()[0];return dict(owned_D_retired=True)
    if name=='launch':return launch(values[0])
    if name=='native':
        assert values[0] in ('candidate','baseline_recovery')
        m.ROOT=ROOT/'epochs'/values[0];return m.native('167',PLAN)
    if name in ('poll','cleanup'):
        m.ROOT=epoch_root();return m.poll(host,PLAN) if name=='poll' else m.cleanup(host,PLAN)
    if name=='recoverycleanup':
        pointer=ROOT/'active_epoch.json'
        if pointer.exists():m.ROOT=epoch_root();m.cleanup(host,PLAN)
        else:
            root=PLAN['resident_roles'][host]
            if m.live(root):m.stop_tree(m.tree(root['pid'],m.catalogue()),root)
        assert not m.device_owners()[0], 'foreign or untracked owner; no relaunch'
        return dict(owned_D_cleaned=True)
    if name=='switch':return switch(int(values[0]))
    if name=='witness':return witness(int(values[0]))
    if name=='gateoff':
        guard('167',True)
        fd=os.open(ROOT/'projection_gate.bin',os.O_WRONLY)
        try: assert os.pwrite(fd,b'\x00',0)==1;os.fsync(fd)
        finally: os.close(fd)
        return dict(byte_gates_off=True)
    raise ValueError(name)




def install():
    assert not m.device_owners()[0], 'only after owned D retirement'
    assert hashlib.sha256((ROOT/'shim/sfa_v1.py').read_bytes()).hexdigest()==PLAN['shim_sha256']
    code="from pathlib import Path;import hashlib,json;p=Path(%r);b=Path(%r).read_bytes();assert hashlib.sha256(p.read_bytes()).hexdigest() in %r;assert hashlib.sha256(b).hexdigest()==%r;p.write_bytes(b);print(json.dumps({'installed_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"%(PLAN['target_source'],str(ROOT/'shim/sfa_v1.py'),(PLAN['original_sha256'],PLAN['shim_sha256']),PLAN['shim_sha256'])
    out=json.loads(subprocess.check_output(['docker','exec','glm52-single',m.PYTHON,'-c',code],text=True));assert out['installed_sha256']==PLAN['shim_sha256'];return out



def decide(results):
    rows={z['label']:z for z in results}
    complete={p:rows[p+'_complete'] for p in ('A1','B1','A2','B2')}
    matched=all(z['workload_signature']==complete['A1']['workload_signature'] for z in complete.values())
    shortmatched=all(rows[p+'_measure%d'%i]['workload_signature']==rows['A1_measure0']['workload_signature'] for p in complete for i in range(2))
    drift=max(abs(complete['A2']['D_wall_s']-complete['A1']['D_wall_s']),abs(complete['B2']['D_wall_s']-complete['B1']['D_wall_s']))
    pairs=[]
    for a,b in (('A1','B1'),('A2','B2')):
        x,y=complete[a],complete[b]
        sm={p:statistics.median(rows[p+'_measure%d'%i]['TPOT_ms'] for i in range(2)) for p in (a,b)}
        pairs.append(dict(pair=a+'->'+b,D_saving_s=x['D_wall_s']-y['D_wall_s'],P_saving_s=x['P_wall_s']-y['P_wall_s'],PD_saving_s=x['PD_wall_s']-y['PD_wall_s'],TPOT_reduction=1-y['TPOT_ms']/x['TPOT_ms'],short_TPOT_medians=sm,short_improved=sm[b]<sm[a]))
    positive=matched and shortmatched and all(z['D_saving_s']>drift and z['TPOT_reduction']>0 and z['PD_saving_s']>0 and z['short_improved'] for z in pairs)
    return dict(research_positive=positive,H6_retained=True,H5_retained=True,H8_retained=positive,complete_work_matched=matched,short_work_matched=shortmatched,D_drift_s=drift,pairs=pairs,Current=None,formal_product_KEEP=False)

if __name__ == "__main__":
    if sys.argv[1] == "workflow":
        workflow()
    else:
        print(json.dumps(action(sys.argv[1], sys.argv[2], sys.argv[3:])))
