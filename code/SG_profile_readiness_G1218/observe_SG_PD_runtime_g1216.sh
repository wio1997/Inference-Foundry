python3 - <<'PY'
from pathlib import Path
import datetime, hashlib, json, socket

assert socket.gethostname() == '<ROLE_HOSTNAME>'
root = Path('/PRIVATE_TASK_ROOT')
run = root / 'sg_pd_g1178'
control = run / 'PD_runtime_gate_control_g1216'
handle = json.loads((control / 'actual_launch_handle.json').read_text())
source = run / 'verify_SG_PD_runtime_g1190.py'
result_path = run / 'PD_runtime_gate_g1190' / 'result.json'
assert Path(handle['result']) == result_path
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha == handle['source_sha256'] == 'f1e059b9753f3a78359bebb05ad93d9d292ee4d1b45e827dc2fbefec7ba75447'

def actual_argv(pid):
    try:
        return Path('/proc',str(pid),'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    except FileNotFoundError:
        return []

worker = run / 'PD_runtime_gate_g1190'
launch = json.loads((worker / 'launch_state.json').read_text()) if (worker / 'launch_state.json').is_file() else {}
current_outer = actual_argv(handle['outer_PID'])
outer_live = str(source).encode() in current_outer
inner_live = 'pid' in launch and str(source).encode() in actual_argv(launch['pid'])
result = json.loads(result_path.read_text()) if result_path.is_file() else None
report = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'outer_live': outer_live,
          'inner_live': inner_live, 'handle': handle, 'result': result}
if result and not (outer_live or inner_live):
    assert result['phase'] == 'complete_requires_parent_review'
    assert result['math1'] == '91' and result['completed'] == 3 and launch['router'] == handle['router']
    for name, digest in result['file_SHA256'].items():
        assert name == Path(name).name
        assert hashlib.sha256((worker / name).read_bytes()).hexdigest() == digest, name
    module = {'__name__': 'preflight'}
    exec(compile(source.read_bytes(), str(source), 'exec'), module)
    assert module['verify_epoch']() == handle['router']
    guard = json.loads((root / 'workload_guard_g1120/state.json').read_text())
    assert guard['phase']=='holding' and guard['pid']==handle['router']['guard_PID']
    rows = [json.loads(f.read_text()) for f in run.glob('Q_head_norm_policy.*.rank*.json')]
    assert {r['rank'] for r in rows} == set(range(8))
    assert all(r['is_dsv41'] and not r['configured_q_head_norm'] and
               r['actual_action']=='skip_per_head_RMSNorm' for r in rows)
    assert all(r['source_SHA256']=='35a9ff6a87e299a380104162c49e29cbe1b7b9bcd04b592080a630a17f95771c' for r in rows)
    report.update(terminal_verified=True, fresh_live_guard=guard, P_all8_runtime_query_policy=rows,
                  raw_files_verified=True, result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
else:
    report.update(terminal_verified=False)
    fail = worker / 'failure.json'
    if fail.is_file():
        report['failure'] = json.loads(fail.read_text())
print('G1216_SG_PD_RUNTIME_OBSERVATION_JSON ' + json.dumps(report))
PY
