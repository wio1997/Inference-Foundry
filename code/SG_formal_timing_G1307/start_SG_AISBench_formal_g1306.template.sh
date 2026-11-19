python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, re, socket, subprocess, time, urllib.request

assert socket.gethostname()=='<DECODE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
client=root/'client_ds41_g1122'
runner=client/'benchmark_pipeline_SG_g1183/run_aisbench_cold_hot_g1306.py'
base_gate=Path(__GATE_FILE__).resolve(strict=True)
usage_path=Path(__USAGE_CONTRACT_FILE__).resolve(strict=True)
phase_seconds={'cold':__COLD_TIMEOUT__,'hot':__HOT_TIMEOUT__}
assert base_gate.is_relative_to(client) and usage_path.is_relative_to(client)
assert all(type(seconds) is int and seconds>0 for seconds in phase_seconds.values())
required_seconds=sum(value+90 for value in phase_seconds.values())+300
runner_sha=hashlib.sha256(runner.read_bytes()).hexdigest()
assert runner_sha=='3276647c3cea77f9570ee5e6279b8bffcc3cc7e58df8194ad8448fe7f034134f'
gate_raw=base_gate.read_bytes()
gate=json.loads(gate_raw)
valid=datetime.datetime.fromisoformat(gate['validated_at'])
now=datetime.datetime.now(datetime.timezone.utc)
age=(now-valid).total_seconds()
assert 0<=age<600
assert gate['profile_epoch']==1178 and gate['workload_guard_actual_verified']
assert gate['stage']=='formal' and not gate['warmup_only']
assert gate['completed_warmups']==5 and gate['next_warmup'] is None and gate['P_compiler_cache_verified_after_warmups']
runtime_raw=(base_gate.parent/'runtime.json').read_bytes()
assert hashlib.sha256(runtime_raw).hexdigest()==gate['source_receipts_sha256']['PD_runtime']
runtime=json.loads(runtime_raw)
assert runtime['terminal_verified'] and runtime['raw_files_verified']
router=runtime['handle']['router']
run_id=gate['run_id']
assert Path(run_id).name==run_id and run_id not in ('.','..')
length=gate['input_tokens']
run_output=client/'runs'/'sglang'/run_id
original_manifest=run_output/'run_manifest.json'
previous_state=run_output/'state.json'
def verify_prior_files():
    assert hashlib.sha256(original_manifest.read_bytes()).hexdigest()==gate['original_run_manifest_sha256']
    assert hashlib.sha256(previous_state.read_bytes()).hexdigest()==gate['previous_state_sha256']

verify_prior_files()
old=json.loads(original_manifest.read_bytes())
warmup_phase_bound=old['phase_timeout_seconds']
warmup_total_bound=old['total_timeout_seconds']
start=datetime.datetime.fromisoformat(old['benchmark_started_at'])
warmup_deadline=start+datetime.timedelta(seconds=warmup_total_bound)
warm=json.loads(previous_state.read_bytes())
assert warm['phase']=='warmup_diagnostics_complete_pending_compile_audit' and warm['readiness_passed']
assert start<=datetime.datetime.fromisoformat(warm['at'])<=min(now,warmup_deadline)
deadlines={'pair_workload_lock':valid+datetime.timedelta(seconds=gate['guard_remaining_seconds'])}
for role in ('prefill','decode'):
    source_raw=(base_gate.parent/(role+'.json')).read_bytes()
    key={'prefill':'P_readiness','decode':'D_readiness'}[role]
    assert hashlib.sha256(source_raw).hexdigest()==gate['source_receipts_sha256'][key]
    role_receipt=json.loads(source_raw)
    created=datetime.datetime.fromisoformat(role_receipt['handle']['at'])
    deadlines[role]=created+datetime.timedelta(seconds=role_receipt['state']['lifetime_seconds'])
    if role=='prefill':
        current_router=role_receipt['actual_router']
        assert current_router['router_PID']==router['router_PID']
        assert current_router['configured_request_deadline_seconds']==router['configured_request_deadline_seconds']
contract=json.loads(usage_path.read_bytes())
assert contract['reviewed'] and contract['actual_SGLang_final_usage']['prompt_tokens']==32768
assert contract['actual_SGLang_final_usage']['completion_tokens']==1024
for row in contract['evidence']:
    path=(usage_path.parent/row['path']).resolve(strict=True)
    assert path.parent==usage_path.parent
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']

cid='d8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
image='sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
container=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=10))[0]
assert container['State']['Running'] and container['Image']==image
assert not container['HostConfig']['Devices']
cpu_state=json.loads((client/'controller_state.json').read_text())
assert cpu_state['pid']==2760614 and container['Config']['Labels']['codex.controller_pid']==str(cpu_state['pid'])
deadlines['CPU_client']=datetime.datetime.fromisoformat(cpu_state['at'])+datetime.timedelta(seconds=70000)
def verify_formal_window(at):
    remaining={name:(deadline-at).total_seconds() for name,deadline in deadlines.items()}
    assert min(remaining.values())>required_seconds,(required_seconds,remaining)
    return remaining

verify_formal_window(now)
procs=subprocess.check_output(['docker','top',cid,'-eo','pid,args'],text=True,timeout=10).splitlines()
assert len(procs)==2 and 'sleep 72000' in procs[1]

control=client/'SG_AISBench_control_g1233'/run_id/'formal'
assert not control.exists(), 'Inspect the existing stage job instead of submitting again'
gate['formal_wall_budget']={'phase_seconds':phase_seconds,
    'resource_deadlines_UTC':{name:value.isoformat() for name,value in deadlines.items()},
    'router_request_timeout_seconds':current_router['configured_request_deadline_seconds'],
    'source_service_gate_SHA256':hashlib.sha256(gate_raw).hexdigest()}
gate_path=base_gate.with_name('service_gate_for_scoring.json')
with gate_path.open('x') as stream:
    json.dump(gate,stream,indent=2)
    stream.write('\n')
gate_scoring_sha=hashlib.sha256(gate_path.read_bytes()).hexdigest()
cmd=['python3','-u',str(runner),'--root',str(root),'--engine','sglang','--container',cid,
     '--image-id',image,'--gate',str(gate_path),'--lengths',str(length),'--warmup-passes','5',
     '--phase-timeout',str(warmup_phase_bound),'--total-timeout',str(warmup_total_bound),
     '--max-gate-age','600','--usage-contract',str(usage_path),'--run-id',run_id,'--resume']
preflight=subprocess.run(cmd+['--preflight-only'],cwd=client,capture_output=True,text=True,timeout=60)
print('G1233_SG_AISBENCH_PREFLIGHT_JSON '+json.dumps({
    'at':now.isoformat(),'returncode':preflight.returncode,'stdout':preflight.stdout,'stderr':preflight.stderr}),flush=True)
assert preflight.returncode==0
review=json.loads(preflight.stdout)
assert review['preflight']=='passed' and not review['benchmark_executed']
assert review['resume_completed_to_skip']==[f'warmup{i}_{length}' for i in range(1,6)]
queues={
    'prefill':('num_prefill_bootstrap_queue_reqs','num_prefill_inflight_queue_reqs'),
    'decode':('num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs'),
}
observed={}
for role,names in queues.items():
    with urllib.request.urlopen(gate['roles'][role]['metrics_url'],timeout=12) as response:
        assert response.status==200
        metrics=response.read().decode()
    observed[role]={}
    for name in ('num_running_reqs','num_queue_reqs',*names):
        values=[float(x) for x in re.findall(r'^sglang:'+name+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',metrics,re.M)]
        observed[role][name]=values
        assert values and all(value==0 for value in values),(role,name,values)
verify_prior_files()
assert base_gate.read_bytes()==gate_raw and hashlib.sha256(gate_path.read_bytes()).hexdigest()==gate_scoring_sha
submitted_at=datetime.datetime.now(datetime.timezone.utc)
assert (submitted_at-valid).total_seconds()<600
remaining_at_launch=verify_formal_window(submitted_at)
control.mkdir(parents=True,exist_ok=False)
with (control/'runner.log').open('x') as logfile:
    child=subprocess.Popen(cmd,cwd=client,stdin=subprocess.DEVNULL,
                           stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
handle={'at':submitted_at.isoformat(),'outer_PID':child.pid,'argv':cmd,'run_id':run_id,'stage':'formal','step':'formal',
        'length':length,'gate_sha256':hashlib.sha256(gate_path.read_bytes()).hexdigest(),'runner_sha256':runner_sha,
        'usage_sha256':hashlib.sha256(usage_path.read_bytes()).hexdigest(),
        'original_warmup_deadline_UTC':warmup_deadline.isoformat(),'formal_phase_seconds':phase_seconds,
        'minimum_resource_remaining_s_at_launch':min(remaining_at_launch.values()),
        'queue_samples_before_submit':observed,
        'formal_submissions_authorized_by_gate':True,
        'previous_actual_warmups_reaudited_and_skipped':review['resume_completed_to_skip'],
        'scope':'One bounded checkpoint stage of the existing campaign with durable raw outputs.'}
with (control/'actual_launch_handle.json').open('x') as stream:
    json.dump(handle,stream,indent=2)
    stream.write('\n')
time.sleep(2)
try:
    args=Path('/proc',str(child.pid),'cmdline').read_bytes().split(b'\0')
except FileNotFoundError:
    args=[]
print('G1233_SG_AISBENCH_ACTUAL_STAGE_JSON '+json.dumps({
    **handle,'actual_outer_live':child.poll() is None and str(runner).encode() in args,
    'immediate_exit_code':child.poll()}),flush=True)
PY
