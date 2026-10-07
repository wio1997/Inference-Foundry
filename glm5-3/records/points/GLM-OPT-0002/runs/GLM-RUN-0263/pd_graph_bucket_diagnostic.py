"""H6/H5 retained, H8 off. One fixed-bucket replay-support diagnostic; no performance claim."""
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
        protected[PLAN['target_source']]=PLAN['shim_sha256'] if fresh and json.loads((ROOT/'active_epoch.json').read_text())['epoch']=='candidate' else PLAN['original_sha256']
        if not fresh:protected[PLAN['retained_SFA_source']]=PLAN['SFA_before_sha256']
        code="from pathlib import Path;import hashlib,json;print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in %r}))"%list(protected)
        got=json.loads(subprocess.check_output(['docker','exec','glm52-single',m.PYTHON,'-c',code],text=True));assert got==protected,(got,protected)
        assert (Path(PLAN['previous_event_root'])/'event_mode.bin').read_bytes()==b'\x01'
        assert (Path(PLAN['previous_projection_root'])/'projection_mode.bin').read_bytes()==b'\x00'
        assert (Path(PLAN['previous_projection_root'])/'projection_gate.bin').read_bytes()==b'\x00'
        assert (ROOT/'mc2_mode.bin' if fresh else Path(PLAN['previous_MC2_root'])/'mc2_mode.bin').read_bytes()==b'\x01'
    starts={str(p['pid']):p['start_ticks'] for p in members if p['pid'] in owners}
    return dict(root=root,device_owners=sorted(owners),worker_start_ticks=starts,worker_namespace_pids=namespaces,health=status,idle=True)










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
    verify_artifact();target=ROOT/'epochs'/name
    assert target.is_dir() and not (target/'native_167.log').exists()
    assert (ROOT/'mc2_mode.bin').read_bytes()==b'\x01'
    (ROOT/'active_epoch.json').write_text(json.dumps(dict(epoch=name))+'\n')
    env=json.loads(Path(PLAN['baseline_env']['167']).read_text())['environment'];env.update(PLAN['environment']['167'])
    env['PYTHONPATH']=':'.join(p for p in env.get('PYTHONPATH','').split(':') if '/glm52-pd/deploy/plugins/' not in p)
    env['LD_LIBRARY_PATH']=str(Path(PLAN['native_candidate']['library']).parent)+':'+env.get('LD_LIBRARY_PATH','')
    env['GLM_MC2_CAPABILITY_MODE_FILE']=str(ROOT/'mc2_mode.bin')
    env['PYTHONPATH']=PLAN['native_candidate']['Python_backend']+':'+env.get('PYTHONPATH','')
    env['PYTHONDONTWRITEBYTECODE']='1';env['GLM_EVENT_WITNESS_SCOPE']='research263'+name
    env['VLLM_USE_BREAKABLE_CUDAGRAPH']='1' if name=='candidate' else '0'
    if name=='candidate':env['GLM_GRAPH_DIAGNOSTIC_ROOT']=str(ROOT)
    else:env.pop('GLM_GRAPH_DIAGNOSTIC_ROOT',None)
    argv=['docker','exec','-d']
    for key,value in sorted(env.items()):argv+=['-e',key+'='+value]
    argv+=['glm52-single',m.PYTHON,str(Path(__file__).resolve()),'native','167',name]
    subprocess.run(argv,check=True,timeout=30,capture_output=True)
    (target/'launch.json').write_text(json.dumps(dict(epoch=name,launch_exit=0,H6_retained=True,H5_retained=True,environment_not_exported=True))+'\n')
    return dict(epoch=name,launch_exit=0)




def witness():
    roles=guard('167',True);target=ROOT/'witness_roles.json';target.write_text(json.dumps(roles,indent=2)+'\n');c=PLAN['native_candidate']
    cmd=['/usr/bin/python3',str(ROOT/'witness_mc2_native.py'),'--library',c['library'],'--sha256',c['sha256'],'--roles',str(target),'--mode-file',str(ROOT/'mc2_mode.bin'),'--mode','1']
    res=subprocess.run(cmd,capture_output=True,text=True,timeout=90);assert res.returncode==0,res.stderr[-3000:];native=json.loads(res.stdout)
    epoch=json.loads((ROOT/'active_epoch.json').read_text())['epoch'];previous=Path(PLAN['previous_event_root']);event=[json.loads(p.read_text()) for p in sorted((previous/'witnesses').glob('research263%s_mode1_rank*.json'%epoch))]
    assert len(event)==16 and {z['rank'] for z in event}==set(range(16))
    assert {z['pid'] for z in event}==set(roles['worker_namespace_pids'].values())
    assert all(z['mode']==1 and not z['configured_overlap'] and z['returned_none'] and z['candidate_utils_sha256']==PLAN['retained_H5_helper_candidate_sha256'] for z in event)
    return dict(H6=native,H5_rows=event,H6_mode=1,H5_event_mode=1,H8=False,same_worker_identity=roles)








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
    owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text());assert owner['run_id']==PLAN['run_id'] and owner['status']=='running' and m.live(owner['owner'])
    retired=False;completed=False;results=[]
    try:
        rpc('167','artifact');m.write('guards_before.json',{h:rpc(h,'guard') for h in ('166','167')})
        retired=True;rpc('167','retire_old');rpc('167','install');rpc('167','launch','candidate');ready()
        m.write('startup_witness.json',rpc('167','graphrows'))
        for label,complete in (('graph_short',False),('graph_complete',True)):
            rpc('166','guard');rpc('167','scope',label)
            result=request(label,complete);results.append(result);m.write('diagnostic_results.json',results)
            rows=rpc('167','graphrows');m.write(label+'_witness.json',rows)
            assert all(not row['errors'] for row in rows), 'graph observation errors'
            m.write(label+'_stack_witness.json',rpc('167','witness'))
        verdict=adjudicate(rows);m.write('diagnostic_decision.json',verdict)
        assert verdict['real_dynamic_replay_supported'],verdict
        completed=True;m.write('diagnostic_status.json',dict(status='completed',results=results,decision=verdict,performance_claim=False,Current=None))
    except BaseException as error:
        m.write('failure.json',dict(error=repr(error),results=results));raise
    finally:
        if retired and not completed:
            m.write('recovery_cleanup.json',rpc('167','recoverycleanup'));rpc('167','restore')
            rpc('167','launch','baseline_recovery');ready();request('H6_H5_recovery_warm')
        if retired:
            m.write('retained_witness.json',rpc('167','witness'))
            m.write('guards_after.json',{h:rpc(h,'guard' if h=='166' else 'guardfresh') for h in ('166','167')})
            m.write('retained_stack.json',dict(H6=True,H5=True,H8=False,H4=False,MC2_mode=1,event_mode=1,graph_diagnostic_supported=completed,temporary_graph_configuration=completed,observer_resident=completed,performance_claim=False,Current=None))




def action(name,host,values):
    if name=='guard':return guard(host)
    if name=='guardfresh':return guard(host,True)
    if name=='artifact':return verify_artifact()
    if name=='install':return install()
    if name=='restore':return install('original')
    if name=='retire_old':
        roles=guard('167');m.stop_tree(m.tree(roles['root']['pid'],m.catalogue()),roles['root']);assert not m.device_owners()[0];return dict(owned_D_retired=True)
    if name=='launch':return launch(values[0])
    if name=='native':
        name=values[0];assert name in ('candidate','baseline_recovery');m.ROOT=ROOT/'epochs'/name
        plan=dict(PLAN)
        if name=='baseline_recovery':plan['native_args']=PLAN['baseline_native_args']
        return m.native('167',plan)
    if name=='poll':m.ROOT=epoch_root();return m.poll(host,PLAN)
    if name=='recoverycleanup':
        if (ROOT/'active_epoch.json').exists():m.ROOT=epoch_root();m.cleanup(host,PLAN)
        else:
            root=PLAN['resident_roles'][host]
            if m.live(root):m.stop_tree(m.tree(root['pid'],m.catalogue()),root)
        assert not m.device_owners()[0];return dict(owned_D_cleaned=True)
    if name=='scope':
        guard('167',True);assert values[0] in ('graph_short','graph_complete');(ROOT/'scope.txt').write_text(values[0]+'\n');return dict(scope=values[0])
    if name=='graphrows':return graphrows()
    if name=='witness':return witness()
    raise ValueError(name)





def install(kind='shim'):
    assert kind in ('shim','original') and not m.device_owners()[0]
    expected=PLAN[kind+'_sha256']
    code="from pathlib import Path;import hashlib,json;p=Path(%r);b=Path(%r).read_bytes();assert hashlib.sha256(p.read_bytes()).hexdigest() in %r;assert hashlib.sha256(b).hexdigest()==%r;p.write_bytes(b);s=Path(%r);b=Path(%r).read_bytes();assert hashlib.sha256(s.read_bytes()).hexdigest() in %r;assert hashlib.sha256(b).hexdigest()==%r;s.write_bytes(b);print(json.dumps({'graph_source':hashlib.sha256(p.read_bytes()).hexdigest(),'SFA_source':hashlib.sha256(s.read_bytes()).hexdigest()}))"%(PLAN['target_source'],str(ROOT/kind/'breakable_aclgraph.py'),(PLAN['original_sha256'],PLAN['shim_sha256']),expected,PLAN['retained_SFA_source'],str(ROOT/'original/sfa_v1.py'),(PLAN['SFA_before_sha256'],PLAN['retained_SFA_sha256']),PLAN['retained_SFA_sha256'])
    return json.loads(subprocess.check_output(['docker','exec','glm52-single',m.PYTHON,'-c',code],text=True))







def graphrows():
    roles=guard('167',True)
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'witnesses').glob('*_rank*.json'))]
    for model in ('target','draft'):
        selected=[row for row in rows if row['draft']==(model=='draft')]
        if model=='draft' and (ROOT/'scope.txt').read_text().strip()=='startup' and not selected:continue
        assert len(selected)==16 and {row['rank'] for row in selected}==set(range(16)),(model,len(selected))
        assert {row['pid'] for row in selected}==set(roles['worker_namespace_pids'].values())
    return rows

def adjudicate(rows):
    details=[]
    for row in rows:
        if row['draft']:
            assert all(':CUDAGraphMode.NONE:' in k for k in row['counts']),row['counts']
            assert not row['entries'], 'GLM MTP must remain eager'
            continue
        cc=row['effective'];assert cc['capture_sizes']==[2] and cc['max_capture']==2
        assert not cc['target_enforce_eager'] and cc['speculative_enforce_eager']
        assert 'FULL_DECODE_ONLY' in cc['cudagraph_mode'] and ('NONE' in cc['compilation_mode'] or cc['compilation_mode']=='0')
        assert len(row['entries'])==1 and row['entries'][0]['captured'] and row['entries'][0]['graphs']>=1 and row['entries'][0]['eager_breaks']==0
        assert not row['errors']
        snapshots=[z for z in row['snapshots'] if z['scope']=='graph_complete' and z['kind']=='replay_before']
        assert len(snapshots)==2
        captures=[z for z in row['snapshots'] if z['kind']=='capture_before'];assert len(captures)==1
        captured=captures[0]
        a,b=snapshots;assert a['descriptor']==b['descriptor'] and a['descriptor']['num_tokens']==2
        assert a['inputs'].keys()==b['inputs'].keys()
        assert all(a['inputs'][k]['ptr']==b['inputs'][k]['ptr'] for k in a['inputs'])
        assert len(captured['metadata'])==len(a['metadata'])==len(b['metadata'])>0
        assert captured['inputs'].keys()==a['inputs'].keys()
        assert all(captured['inputs'][k]['ptr']==a['inputs'][k]['ptr'] for k in a['inputs'])
        for mc,ma in zip(captured['metadata'],a['metadata']):
            assert mc['first_layer']==ma['first_layer']
            for key in ('seq_lens','cum_query_lens','slot_mapping','block_table','sin','cos'):
                assert mc['fields'][key]['ptr']==ma['fields'][key]['ptr'],('capture_to_real',row['rank'],key)
            for key in ('seq_lens','slot_mapping','block_table'):
                assert mc['fields']['dcp'][key]['ptr']==ma['fields']['dcp'][key]['ptr'],('capture_to_real_dcp',row['rank'],key)
        for ma,mb in zip(a['metadata'],b['metadata']):
            assert ma['first_layer']==mb['first_layer'];fa,fb=ma['fields'],mb['fields']
            for key in ('seq_lens','cum_query_lens','slot_mapping','block_table','sin','cos'):
                assert fa[key]['ptr']==fb[key]['ptr'] and fa[key]['shape']==fb[key]['shape'],(row['rank'],key)
            assert fa['seq_lens']['first32'][0]>=58 and fb['seq_lens']['first32'][0]>fa['seq_lens']['first32'][0]
            assert fa['slot_mapping']['first32']!=fb['slot_mapping']['first32']
            assert fa['dcp'] and fb['dcp']
            for key in ('seq_lens','slot_mapping','block_table'):assert fa['dcp'][key]['ptr']==fb['dcp'][key]['ptr']
        assert row['counts'].get('graph_complete:target:CUDAGraphMode.FULL:replay_after',0)>=2
        details.append(dict(rank=row['rank'],actual_replay_count=row['counts'].get('graph_complete:target:CUDAGraphMode.FULL:replay_after'),persistent_dynamic_metadata=True))
    return dict(real_dynamic_replay_supported=len(details)==16,all_rank_details=details,bucket=2,MTP_eager=True,complete_API=False,performance_claim=False,scope='one uniform K1 request; larger batches fall back; not concurrency1-4 acceptance')

if __name__ == "__main__":
    if sys.argv[1] == "workflow":
        workflow()
    else:
        print(json.dumps(action(sys.argv[1], sys.argv[2], sys.argv[3:])))
