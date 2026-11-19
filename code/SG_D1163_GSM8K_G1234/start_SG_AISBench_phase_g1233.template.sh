python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, socket, subprocess, time

assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
client=root/'client_ds41_g1122'
runner=client/'benchmark_pipeline_SG_g1183/run_aisbench_cold_hot_g1222.py'
gate_path=Path(__GATE_FILE__).resolve(strict=True)
usage_path=client/'SG_PD_usage_g1227/usage_contract.json'
requested_phase_bound=__PHASE_TIMEOUT__
requested_total_bound=__TOTAL_TIMEOUT__
assert gate_path.is_relative_to(client)
assert 60 <= requested_phase_bound <= 5400
assert 300 <= requested_total_bound <= 21600
runner_sha=hashlib.sha256(runner.read_bytes()).hexdigest()
assert runner_sha=='157316e0055792f2a311492ae7aeedec7ae0461e1ec2eebb62caeda4f5b5fc2e'
gate=json.loads(gate_path.read_bytes())
valid=datetime.datetime.fromisoformat(gate['validated_at'])
now=datetime.datetime.now(datetime.timezone.utc)
age=(now-valid).total_seconds()
assert 0<=age<600
assert gate['profile_epoch']==1178 and gate['workload_guard_actual_verified']
assert gate['guard_remaining_seconds']-age > requested_total_bound+300
assert gate['stage'] in ('initial','remaining','formal')
assert gate['warmup_only']==(gate['stage']!='formal')
assert (gate['stage']=='initial')==(gate.get('warmup_pause_after')==3)
run_id=gate['run_id']
assert Path(run_id).name==run_id and run_id not in ('.','..')
length=gate['input_tokens']
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
remaining_cpu=70000-(now-datetime.datetime.fromisoformat(cpu_state['at'])).total_seconds()
assert remaining_cpu>requested_total_bound+300
procs=subprocess.check_output(['docker','top',cid,'-eo','pid,args'],text=True,timeout=10).splitlines()
assert len(procs)==2 and 'sleep 72000' in procs[1]

control=client/'SG_AISBench_control_g1233'/run_id/gate['stage']
assert not control.exists(), 'Inspect the existing stage job instead of submitting again'
cmd=['python3','-u',str(runner),'--root',str(root),'--engine','sglang','--container',cid,
     '--image-id',image,'--gate',str(gate_path),'--lengths',str(length),'--warmup-passes','5',
     '--phase-timeout',str(requested_phase_bound),'--total-timeout',str(requested_total_bound),
     '--max-gate-age','600','--usage-contract',str(usage_path),'--run-id',run_id]
if gate['stage']!='initial':
    cmd.append('--resume')
preflight=subprocess.run(cmd+['--preflight-only'],cwd=client,capture_output=True,text=True,timeout=60)
print('G1233_SG_AISBENCH_PREFLIGHT_JSON '+json.dumps({
    'at':now.isoformat(),'returncode':preflight.returncode,'stdout':preflight.stdout,'stderr':preflight.stderr}),flush=True)
assert preflight.returncode==0 and json.loads(preflight.stdout)['preflight']=='passed'
control.mkdir(parents=True,exist_ok=False)
with (control/'runner.log').open('x') as logfile:
    child=subprocess.Popen(cmd,cwd=client,stdin=subprocess.DEVNULL,
                           stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
handle={'at':now.isoformat(),'outer_PID':child.pid,'argv':cmd,'run_id':run_id,'stage':gate['stage'],
        'length':length,'gate_sha256':hashlib.sha256(gate_path.read_bytes()).hexdigest(),'runner_sha256':runner_sha,
        'usage_sha256':hashlib.sha256(usage_path.read_bytes()).hexdigest(),
        'scope':'One bounded stage with durable raw outputs. Observe this exact handle.'}
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
