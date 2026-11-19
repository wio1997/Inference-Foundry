python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, re, socket, subprocess, time, urllib.request

assert socket.gethostname()=='<DECODE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
client=root/'client_ds41_g1122'
runner=client/'benchmark_pipeline_SG_g1183/run_aisbench_cold_hot_g1294.py'
gate_path=Path(__GATE_FILE__).resolve(strict=True)
usage_path=Path(__USAGE_CONTRACT_FILE__).resolve(strict=True)
requested_phase_bound=__PHASE_TIMEOUT__
requested_total_bound=__TOTAL_TIMEOUT__
assert gate_path.is_relative_to(client) and usage_path.is_relative_to(client)
assert 60 <= requested_phase_bound <= 14400
assert requested_phase_bound+120 <= requested_total_bound <= 57600
runner_sha=hashlib.sha256(runner.read_bytes()).hexdigest()
assert runner_sha=='62bf953970fe0f2dec4cd8fd196f8e8f651cd97809c8f20e76f1d86ac41ab2e0'
gate_raw=gate_path.read_bytes()
gate=json.loads(gate_raw)
valid=datetime.datetime.fromisoformat(gate['validated_at'])
now=datetime.datetime.now(datetime.timezone.utc)
age=(now-valid).total_seconds()
assert 0<=age<600
assert gate['profile_epoch']==1178 and gate['workload_guard_actual_verified']
assert gate['stage'] in ('remaining','formal') and gate['warmup_only']==(gate['stage']=='remaining')
runtime_raw=(gate_path.parent/'runtime.json').read_bytes()
assert hashlib.sha256(runtime_raw).hexdigest()==gate['source_receipts_sha256']['PD_runtime']
runtime=json.loads(runtime_raw)
assert runtime['terminal_verified'] and runtime['raw_files_verified']
assert runtime['handle']['router']['router_PID']==2299882
assert runtime['handle']['router']['configured_request_deadline_seconds']>=requested_phase_bound
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
assert requested_total_bound==old['total_timeout_seconds'] and requested_phase_bound<=old['phase_timeout_seconds']
start=datetime.datetime.fromisoformat(old['benchmark_started_at'])
deadline=start+datetime.timedelta(seconds=requested_total_bound)
assert now>=start
completed=gate['completed_warmups']
if gate['warmup_only']:
    assert 1<=completed<=4 and gate['next_warmup']==completed+1
    assert gate.get('warmup_pause_after')==(completed+1 if completed<4 else None)
    step=f'warmup{completed+1}'
    stage_lifetime_bound=requested_phase_bound+95
else:
    assert completed==5 and gate['next_warmup'] is None and gate['P_compiler_cache_verified_after_warmups']
    step='formal'
    stage_lifetime_bound=2*(requested_phase_bound+95)
assert (deadline-now).total_seconds()>stage_lifetime_bound+120
assert gate['guard_remaining_seconds']-age > stage_lifetime_bound+300
for role in ('prefill','decode'):
    source_raw=(gate_path.parent/(role+'.json')).read_bytes()
    key={'prefill':'P_readiness','decode':'D_readiness'}[role]
    assert hashlib.sha256(source_raw).hexdigest()==gate['source_receipts_sha256'][key]
    role_receipt=json.loads(source_raw)
    created=datetime.datetime.fromisoformat(role_receipt['handle']['at'])
    role_remaining=role_receipt['state']['lifetime_seconds']-(now-created).total_seconds()
    assert role_remaining>stage_lifetime_bound+300, (role,role_remaining)
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
cpu_lease=json.loads(subprocess.check_output(['python3',str(client/'cpu_client_lease_g1310.py'),'report'],timeout=15))
assert cpu_lease['container_ID']==cid
cpu_deadline=datetime.datetime.fromisoformat(cpu_lease['conservative_deadline_UTC'])
remaining_cpu=(cpu_deadline-now).total_seconds()
assert remaining_cpu>stage_lifetime_bound+300
procs=subprocess.check_output(['docker','top',cid,'-eo','pid,args'],text=True,timeout=10).splitlines()
assert len(procs)==2 and 'sleep 72000' in procs[1]

control=client/'SG_AISBench_control_g1233'/run_id/step
assert not control.exists(), 'Inspect the existing stage job instead of submitting again'
cmd=['python3','-u',str(runner),'--root',str(root),'--engine','sglang','--container',cid,
     '--image-id',image,'--gate',str(gate_path),'--lengths',str(length),'--warmup-passes','5',
     '--phase-timeout',str(requested_phase_bound),'--total-timeout',str(requested_total_bound),
     '--max-gate-age','600','--usage-contract',str(usage_path),'--run-id',run_id,'--resume']
preflight=subprocess.run(cmd+['--preflight-only'],cwd=client,capture_output=True,text=True,timeout=60)
print('G1233_SG_AISBENCH_PREFLIGHT_JSON '+json.dumps({
    'at':now.isoformat(),'returncode':preflight.returncode,'stdout':preflight.stdout,'stderr':preflight.stderr}),flush=True)
assert preflight.returncode==0
review=json.loads(preflight.stdout)
assert review['preflight']=='passed' and not review['benchmark_executed']
assert review['resume_completed_to_skip']==[f'warmup{i}_{length}' for i in range(1,completed+1)]
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
assert gate_path.read_bytes()==gate_raw
submitted_at=datetime.datetime.now(datetime.timezone.utc)
assert (submitted_at-valid).total_seconds()<600
assert (deadline-submitted_at).total_seconds()>stage_lifetime_bound+120
assert gate['guard_remaining_seconds']-(submitted_at-valid).total_seconds()>stage_lifetime_bound+300
assert (cpu_deadline-submitted_at).total_seconds()>stage_lifetime_bound+300
control.mkdir(parents=True,exist_ok=False)
with (control/'runner.log').open('x') as logfile:
    child=subprocess.Popen(cmd,cwd=client,stdin=subprocess.DEVNULL,
                           stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
handle={'at':submitted_at.isoformat(),'outer_PID':child.pid,'argv':cmd,'run_id':run_id,'stage':gate['stage'],'step':step,
        'length':length,'gate_sha256':hashlib.sha256(gate_path.read_bytes()).hexdigest(),'runner_sha256':runner_sha,
        'usage_sha256':hashlib.sha256(usage_path.read_bytes()).hexdigest(),
        'campaign_deadline_UTC':deadline.isoformat(),'stage_bound_seconds':stage_lifetime_bound,
        'CPU_client_lease':cpu_lease,
        'queue_samples_before_submit':observed,
        'warmup_pause_after':gate.get('warmup_pause_after'),'formal_submissions_authorized_by_gate':not gate['warmup_only'],
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
