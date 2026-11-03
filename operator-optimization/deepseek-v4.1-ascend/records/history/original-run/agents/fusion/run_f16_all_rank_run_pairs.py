#!/usr/bin/env python3
"""Run one pinned read-only F16 query in the dedicated CPU staging container."""
import hashlib, json, shlex, subprocess
from pathlib import Path
T = Path(__file__).resolve().parents[2]
S = T / 'agents/fusion/query_f16_all_rank_run_pairs.py'
OUT = T / 'agents/fusion/f16-layer24-all-rank-run-pairs.json'
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER = 'wio-vllm-v41-repro169-stage-20261028'
TASK = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
TRACE = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
script = S.read_bytes()
script_sha = hashlib.sha256(script).hexdigest()
assert script_sha == 'dc2b8a16a1e8eee94550135dc2625211b2d5b40bbeb894aae11feb9c031b5538'
if OUT.exists() or OUT.is_symlink():
    raise FileExistsError('fixed output already exists; refusing overwrite')
command = shlex.join(['docker', 'exec', CONTAINER, 'python3', '-c', script.decode('utf-8'), '--task-root', TASK, '--trace-root', TRACE])
result = subprocess.run([SSH, command], capture_output=True, text=True, timeout=240)
if result.returncode:
    raise RuntimeError('read-only query failed with exit code %d' % result.returncode)
x = json.loads(result.stdout)
rows = x['rows']
assert len(rows) == 48 and len({r['rank'] for r in rows}) == 16
assert len({(r['rank'],r['run']) for r in rows}) == 48
assert {r['run'] for r in rows} == {1, 2, 3} and all(r['layer'] == 24 and r['model'] == '49' for r in rows)
assert len(x['inputs']['kernel_csv_sha256']) == 16
assert sum(c['windows'] for c in x['counts']) == 48
assert x['logical_mac_if_standard_dense'] == {'first': 100663296, 'second': 125829120}
for r in rows:
    assert r['first_count'] == len(r['first']) and r['second_count'] == len(r['second'])
    unique = len(r['first']) == len(r['second']) == 1
    ordered = unique and r['first'][0]['end_rel_ns'] <= r['second'][0]['start_rel_ns']
    assert r['unique_ordered_candidates'] == ordered
x['query_sha256'] = script_sha
x['wrapper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
x['no_device_work'] = True
body = (json.dumps(x, ensure_ascii=False, indent=2) + '\n').encode()
with OUT.open('xb') as f:
    f.write(body)
print(json.dumps({'status': 'READ_ONLY_QUERY_COMPLETED', 'output': str(OUT), 'sha256': hashlib.sha256(body).hexdigest(),
                  'counts': x['counts'], 'windows': len(rows)}, ensure_ascii=False))
