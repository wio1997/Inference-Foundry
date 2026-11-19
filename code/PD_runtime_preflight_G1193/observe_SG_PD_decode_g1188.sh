python3 - <<'PY'
from pathlib import Path
import datetime
import json
import os
import re
import socket
import subprocess
import urllib.error
import urllib.request

host = socket.gethostname()
assert host == '<DECODE_HOSTNAME>'
root = Path('/PRIVATE_TASK_ROOT')
run = root / 'sg_pd_g1178'
handle_path = run / 'actual_launch_handle.json'
assert handle_path.is_file(), 'No actual Decode launch handle'
handle = json.loads(handle_path.read_text())
assert handle['role'] == 'd' and handle['epoch'] == 1178
pid = handle['pid']
process = Path('/proc', str(pid))
def live():
    try:
        return str(run / 'node_controller.py').encode() in (process / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False

out = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'handle': handle,
       'actual_controller_live': live(), 'ready': None}
state_path = run / 'state.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {}
out['state'] = state
if state.get('container_id'):
    cid = state['container_id']
    container = json.loads(subprocess.check_output(['docker', 'inspect', cid], timeout=10))[0]
    labels = container['Config']['Labels']
    assert container['Image'] == handle['image'] and labels['codex.controller_pid'] == str(pid)
    assert labels['codex.task'] == root.name and labels['codex.run'] == run.name
    out['container_ID'] = cid
    out['container_actual_running'] = container['State']['Running']
ledgers = {}
for kind in ('target', 'draft'):
    rows = [json.loads(path.read_text()) for path in sorted((run / 'weight_ledgers').glob(kind + '.*.json'))]
    ledgers[kind] = {'rows': len(rows), 'passed': sum(x['status'] == 'passed' and x['missing'] == [] for x in rows),
                     'parameter_counts': sorted({row.get('parameter_count') for row in rows})}
out['ledgers'] = ledgers
capture = {kind: {stage: {} for stage in ('begin', 'end')} for kind in ('target', 'draft')}
pattern = re.compile(r'TP(\d+) EP(\d+)\] Capture (target|draft) verify NPU graph (begin|end)\. (.*)')
log = run / 'server.log'
if log.exists():
    with log.open(errors='replace') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                tp, ep, model, step, details = match.groups()
                assert tp == ep
                capture[model][step][int(tp)] = details
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size - 3500))
        out['log_tail'] = f.read().decode(errors='replace')
out['verify_graph_captures'] = capture
out['query_installs'] = len(list(run.glob('Q_head_norm_policy_install.pid*.json')))
def get(path):
    with urllib.request.urlopen('http://127.0.0.1:19094'+path, timeout=6) as response:
        assert response.status == 200
        return response.read()

try:
    get('/ready')
except (urllib.error.URLError, TimeoutError) as exc:
    out['ready_error'] = repr(exc)
else:
    assert live() and out.get('container_actual_running')
    paths = set()
    for fd in (process / 'fd').iterdir():
        try: paths.add(os.readlink(fd))
        except FileNotFoundError: pass
    for rank in range(8):
        base = Path('/PRIVATE_COORD_ROOT', f'{host}-npu{rank}')
        owner = json.loads(base.with_suffix('.owner.json').read_text())
        assert owner['state'] == 'claimed' and owner['pid'] == pid and owner['run_directory'] == str(run)
        assert str(base.with_suffix('.lock')) in paths
    out['all8_ownership_and_FDs_verified'] = True
    for kind, count, width in [('target', 1384, 6), ('draft', 73, 5)]:
        assert ledgers[kind] == {'rows': 8, 'passed': 8, 'parameter_counts': [count]}
        begins = capture[kind]['begin']
        ends = capture[kind]['end']
        assert set(begins) == set(ends) == set(range(8))
        assert all(re.search(r'backend=full, num_tokens_per_req='+str(width)+r'(?:,|$)', v) for v in begins.values())
        assert all(16 in [int(n) for n in re.search(r'bs=\[([^\]]+)\]', v)[1].split(',')] for v in ends.values())
    assert out['query_installs'] == 8
    result = json.loads(get('/server_info'))
    a = result.get('server_args', result)
    assert (a['tp_size'], a['ep_size'], a['pp_size']) == (8,8,1)
    assert (a['disaggregation_mode'], a['disaggregation_transfer_backend']) == ('decode','ascend')
    assert a['speculative_algorithm'] == 'DSPARK' and a['speculative_num_draft_tokens'] == 6
    assert a['cuda_graph_config']['decode']['backend'] == 'full' and a['cuda_graph_config']['decode']['max_bs'] == 16
    assert not a['disable_overlap_schedule'] and not a['disable_radix_cache']
    assert a['disaggregation_decode_enable_radix_cache'] and a['max_running_requests'] == 16
    assert a['context_length'] == 263296 and a['chunked_prefill_size'] == 1024
    out['server_args'] = {name: a[name] for name in ('tp_size','ep_size','pp_size','disaggregation_mode',
        'disaggregation_transfer_backend','speculative_algorithm','speculative_num_draft_tokens',
        'cuda_graph_config','disable_overlap_schedule','disable_radix_cache','disaggregation_decode_enable_radix_cache',
        'context_length','chunked_prefill_size','max_running_requests','mem_fraction_static')}
    metrics = get('/metrics').decode()
    out['idle'] = {key: [float(v) for v in re.findall(r'^sglang:'+key+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',metrics,re.M)]
                   for key in ('num_running_reqs','num_queue_reqs')}
    assert all(v and sum(v)==0 for v in out['idle'].values())
    out['ready'] = 200
    out['graph_capture_all8_verified'] = True
    out['scope'] = 'Actual Decode loading and capture verified. Runtime replays and P/D transfer require real requests.'
print('G1188_SG_D_OBSERVATION_JSON '+json.dumps(out))
PY
