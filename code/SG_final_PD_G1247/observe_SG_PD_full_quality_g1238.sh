python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, re, socket, subprocess, urllib.request

root = Path('/PRIVATE_TASK_ROOT')
assert socket.gethostname() == '<ROLE_HOSTNAME>'
client = root / 'client_ds41_g1122'
control = client / 'SG_PD_g1178_full_GSM8K_control_g1217'
output = client / 'SG_PD_g1178_full_GSM8K_g1192'
handle = json.loads((control / 'actual_launch_handle.json').read_text())
assert handle['outer_PID']>0 and handle['argv'][-1].endswith('/run_full_gsm8k_SG_PD_g1192.py')
name = handle['argv'][-1]
top = subprocess.check_output(['docker','top',handle['CPU_container_ID'],'-eo','pid,args'], text=True)
workers = []
for line in top.splitlines()[1:]:
    pid = int(line.split()[0])
    process = Path('/proc', str(pid), 'cmdline')
    try:
        args = process.read_bytes().split(b'\0')
    except FileNotFoundError:
        continue
    if args[0].endswith(b'python3') and name.encode() in args:
        workers.append(pid)
state = output / 'state.json'
result = output / 'result.json'
log = output / 'results.jsonl'
raw_rows = log.read_bytes().split(b'\n')[:-1] if log.exists() else []
rows = [json.loads(line) for line in raw_rows]
assert len({row['index'] for row in rows}) == len(rows)
with urllib.request.urlopen('http://127.0.0.1:19094/metrics', timeout=10) as response:
    metrics = response.read().decode()
counts = {key: sum(float(x) for x in re.findall(r'^sglang:' + key + r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)', metrics, re.M)) for key in ['num_running_reqs','num_queue_reqs']}
running = Path('/proc',str(handle['outer_PID']),'cmdline')
outer_live = running.exists() and name.encode() in running.read_bytes()
report = {
    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'actual_inner_python_pids': workers,
    'actual_outer_live': outer_live,
    'state': json.loads(state.read_text()) if state.is_file() else None,
    'result': json.loads(result.read_text()) if result.is_file() else None,
    'completed': len(rows),
    'correct': sum(row['correct'] for row in rows),
    'transport_failures': sum(not row['transport_ok'] for row in rows),
    'length_limited': sum(row.get('finish_reason') == 'length' for row in rows),
    'error_sample': [{'index':row['index'],'error':row.get('error')} for row in rows if not row['transport_ok']][:5],
    'metrics': counts,
    'source_SHA256': hashlib.sha256((client / Path(name).name).read_bytes()).hexdigest(),
}
if not workers and not outer_live:
    report['log_tail'] = (control / 'quality.log').read_bytes()[-4000:].decode(errors='replace')
print('G1238_SG_PD_FULL_GSM8K_OBSERVATION_JSON '+json.dumps(report))
PY
