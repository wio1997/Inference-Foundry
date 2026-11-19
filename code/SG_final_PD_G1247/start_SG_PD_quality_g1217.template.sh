python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, os, re, socket, subprocess, time, urllib.request

assert socket.gethostname() == '<ROLE_HOSTNAME>' and os.geteuid() == 0
root = Path('/PRIVATE_TASK_ROOT')
run = root / 'sg_pd_g1178'
client = root / 'client_ds41_g1122'
output = client / 'SG_PD_g1178_full_GSM8K_g1192'
control = client / 'SG_PD_g1178_full_GSM8K_control_g1217'
source = client / 'run_full_gsm8k_SG_PD_g1192.py'
assert not output.exists() and not control.exists()

proof = __P_RUNTIME_PROOF__
now = datetime.datetime.now(datetime.timezone.utc)
assert -60 <= (now - datetime.datetime.fromisoformat(proof['at'])).total_seconds() < 300
assert proof['terminal_verified'] and proof['raw_files_verified'] and not (proof['outer_live'] or proof['inner_live'])
assert {row['rank'] for row in proof['P_all8_runtime_query_policy']} == set(range(8))
assert proof['result']['math1'] == '91' and proof['result']['completed'] == 3
for key in ('transfer_count_delta_by_P_rank', 'transfer_mb_delta_by_P_rank', 'D_full_graph_counter_delta_by_rank'):
    assert set(map(int, proof['result'][key])) == set(range(8)) and min(proof['result'][key].values()) > 0
router = proof['handle']['router']
guard = proof['fresh_live_guard']
budget_s = 14535
assert guard['phase'] == 'holding' and guard['pid'] == router['guard_PID'] == 2227468
remaining = guard['bound_s'] - (now-datetime.datetime.fromisoformat(guard['at'])).total_seconds()
assert remaining > budget_s + 300, remaining

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def proc_argv(pid):
    try:
        return Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return []

def http(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as response:
        assert response.status == 200
        return response.read()

def fds(pid):
    result = set()
    for p in Path('/proc', str(pid), 'fd').iterdir():
        try:
            result.add(os.readlink(p))
        except FileNotFoundError:
            continue
    return result

def check_owner(owner_name, pid, locks, directory):
    prefix = Path('/PRIVATE_COORD_ROOT') / owner_name
    owner = json.loads(prefix.with_suffix('.owner.json').read_text())
    assert owner['state'] == 'claimed' and owner['pid'] == pid
    assert str(prefix.with_suffix('.lock')) in locks and owner['run_directory'] == str(directory)

state = json.loads((run / 'state.json').read_text())
handle = json.loads((run / 'actual_launch_handle.json').read_text())
dpid = router['D_controller_PID']
assert state['generation'] == 1178 and state['role'] == 'd' and state['pid'] == handle['pid'] == dpid
assert state['container_id'] == router['D_container_ID']
assert str(run / 'node_controller.py').encode() in proc_argv(dpid)
decode = json.loads(subprocess.check_output(['docker', 'inspect', state['container_id']], timeout=10))[0]
assert decode['State']['Running'] and decode['Config']['Labels']['codex.controller_pid'] == str(dpid)
assert decode['Image'] == 'sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'
locks = fds(dpid)
for rank in range(8):
    check_owner(f'<ROLE_HOSTNAME>-npu{rank}', dpid, locks, run)
d_q_runtime = [json.loads(f.read_text()) for f in run.glob('Q_head_norm_policy.*.rank*.json')]
assert {row['rank'] for row in d_q_runtime} == set(range(8))
assert all(row['is_dsv41'] and not row['configured_q_head_norm'] and
           row['actual_action'] == 'skip_per_head_RMSNorm' for row in d_q_runtime)
assert all(row['source_SHA256'] == '35a9ff6a87e299a380104162c49e29cbe1b7b9bcd04b592080a630a17f95771c'
           for row in d_q_runtime)

pbase, dbase, proxy = 'http://<ROLE_HOST>:19093', 'http://<ROLE_HOST>:19094', 'http://<ROLE_HOST>:18090'
for base in (pbase, dbase):
    http(base, '/ready')
    metrics = http(base, '/metrics').decode()
    for metric in ('num_running_reqs', 'num_queue_reqs'):
        values = [float(v) for v in re.findall(r'^sglang:' + metric + r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)', metrics, re.M)]
        assert values and sum(values) == 0, (base, metric, values)
workers = json.loads(http(proxy, '/workers'))['workers']
assert len(workers) == 2
assert {(w['worker_type'], w['url']) for w in workers
        if w['is_healthy'] and w['load'] == 0 and w['model_id'] == 'ds41flash'} == {('prefill', pbase), ('decode', dbase)}
assert json.loads(http(proxy, '/v1/models'))['data'][0]['id'] == 'ds41flash'

cpid = 2760614
client_state = json.loads((client / 'controller_state.json').read_text())
assert client_state['pid'] == cpid and client_state['phase'] == 'ready_for_separately_gated_diagnostic'
assert str(client / 'cpu_client_controller.py').encode() in proc_argv(cpid)
assert 70000 - (now-datetime.datetime.fromisoformat(client_state['at'])).total_seconds() > budget_s + 300
cpu = json.loads(subprocess.check_output(['docker', 'inspect', 'yxy-ds41-aisbench-g1122'], timeout=10))[0]
assert cpu['Id'] == 'd8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
assert cpu['State']['Running'] and cpu['Config']['Labels']['codex.controller_pid'] == str(cpid)
assert cpu['Image'] == 'sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
cfg = cpu['HostConfig']
assert not cfg['Devices'] and not cfg['Privileged'] and cfg['ReadonlyRootfs'] and cfg['CpusetCpus']=='184-187'
check_owner('<ROLE_HOSTNAME>-cpu184-187', cpid, fds(cpid), client)
processes = subprocess.check_output(['docker', 'top', cpu['Id'], '-eo', 'pid,args'], text=True, timeout=10)
assert len(processes.splitlines()) == 2 and 'sleep 72000' in processes.splitlines()[1]
assert any(m['Source'] == str(client) and m['Destination'] == '/task/client_ds41_g1122' and m['RW'] for m in cpu['Mounts'])
source_sha = digest(source)
assert source_sha == 'aefe2729cd0803b037c828e900ebe24d40e76dcafb7ba4f96905bd05894ad245'
data = root / 'client_ds41_g503' / 'quality_g655'
for entry in json.loads((data / 'dataset_provenance.json').read_text())['files']:
    assert digest(data / entry['path']) == entry['sha256']

control.mkdir(exist_ok=False)
gate = {'at': now.isoformat(), 'P_runtime_proof': proof, 'D_runtime_query_policy': d_q_runtime,
        'guard_remaining_s': remaining, 'CPU_controller_PID': cpid, 'CPU_container_ID': cpu['Id'],
        'source_SHA256': source_sha, 'exact_P_D_router_workers': workers}
with (control / 'gate.json').open('x') as f:
    json.dump(gate, f, indent=2)
argv = ['docker', 'exec', cpu['Id'], 'timeout', '--signal=TERM', '--kill-after=15', '14520',
        'python3', '-u', '/task/client_ds41_g1122/' + source.name]
with (control / 'quality.log').open('x') as log:
    worker = subprocess.Popen(argv, cwd=control, stdin=subprocess.DEVNULL, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
report = {'at': now.isoformat(), 'outer_PID': worker.pid, 'argv': argv, 'output': str(output),
          'CPU_container_ID': cpu['Id'], 'source_SHA256': source_sha,
          'scope': 'Full pinned GSM8K through final SGLang P/D. Preserve results. No automatic relaunch.'}
with (control / 'actual_launch_handle.json').open('x') as f:
    json.dump(report, f, indent=2)
time.sleep(2)
print('G1217_SG_PD_FULL_QUALITY_LAUNCH_JSON ' + json.dumps({
    **report, 'actual_outer_live': worker.poll() is None and argv[-1].encode() in proc_argv(worker.pid),
    'immediate_exit_code': worker.poll()}))
PY
