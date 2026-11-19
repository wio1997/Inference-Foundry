python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, os, socket, subprocess, time

assert socket.gethostname() == '<ROLE_HOSTNAME>' and os.geteuid() == 0
root = Path('/PRIVATE_TASK_ROOT')
run = root / 'sg_pd_g1178'
source = run / 'verify_SG_PD_runtime_g1190.py'
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha == 'f1e059b9753f3a78359bebb05ad93d9d292ee4d1b45e827dc2fbefec7ba75447'
assert not (run / 'PD_runtime_gate_g1190').exists()

# Load the frozen verifier without executing its request-producing entry point.
module = {'__name__': 'preflight'}
exec(compile(source.read_bytes(), str(source), 'exec'), module)
router = module['verify_epoch']()
guard = json.loads((root / 'workload_guard_g1120' / 'state.json').read_text())
assert guard['phase'] == 'holding' and guard['pid'] == router['guard_PID']
now = datetime.datetime.now(datetime.timezone.utc)
remaining = guard['bound_s'] - (now - datetime.datetime.fromisoformat(guard['at'])).total_seconds()
assert remaining > 905, remaining

control = run / 'PD_runtime_gate_control_g1216'
control.mkdir(exist_ok=False)
argv = ['timeout', '--signal=TERM', '--kill-after=12', '620', 'python3', '-u', str(source)]
with (control / 'runtime.log').open('x') as log:
    worker = subprocess.Popen(argv, cwd=run, stdin=subprocess.DEVNULL,
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
handle = {'at': now.isoformat(), 'outer_PID': worker.pid, 'argv': argv, 'router': router,
          'source_sha256': source_sha, 'guard_remaining_s_at_submit': remaining,
          'result': str(run / 'PD_runtime_gate_g1190' / 'result.json')}
with (control / 'actual_launch_handle.json').open('x') as f:
    json.dump(handle, f, indent=2)
time.sleep(2)
try:
    cmdline = Path('/proc', str(worker.pid), 'cmdline').read_bytes().split(b'\0')
except FileNotFoundError:
    cmdline = []
print('G1216_SG_PD_RUNTIME_LAUNCH_JSON ' + json.dumps({
    **handle, 'actual_outer_live': worker.poll() is None and str(source).encode() in cmdline,
    'immediate_exit_code': worker.poll(),
    'scope': 'Exactly one bounded submission. Inspect this same handle and raw results.'}))
PY
