#!/usr/bin/env python3
"""Run pinned fixed F16 counter extraction using only saved profile CSVs."""
import hashlib, json, shlex, subprocess
from pathlib import Path
T = Path(__file__).resolve().parents[2]
QUERY = T / 'agents/fusion/query_f16_fixed_counters.py'
SOURCE = T / 'agents/fusion/f16-layer24-all-rank-run-pairs.json'
OUT = T / 'agents/fusion/f16-layer24-fixed-counters.json'
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER = 'wio-vllm-v41-repro169-stage-20261028'
TASK = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
TRACE = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
query = QUERY.read_bytes()
query_sha = hashlib.sha256(query).hexdigest()
assert query_sha == 'ec3da2bbeae9c366bb38f57dda83d5e5bfa4c74c705889f0d441d4ed11e65a76'
source = SOURCE.read_bytes()
source_sha = hashlib.sha256(source).hexdigest()
assert source_sha == '26d0ab676539995816c26f697d0702941e7284983f020ff9f221f86c3d81840f'
data = json.loads(source)
assert len(data['rows']) == 48
anchors = json.dumps({'sha256':source_sha, 'rows':data['rows']}, separators=(',',':'))
if OUT.exists() or OUT.is_symlink():
    raise FileExistsError('fixed output already exists; refusing overwrite')
command = shlex.join(['docker','exec',CONTAINER,'python3','-c',query.decode(),'--task-root',TASK,'--trace-root',TRACE,'--anchors-json',anchors])
result = subprocess.run([SSH,command],capture_output=True,text=True,timeout=240)
if result.returncode:
    raise RuntimeError('read-only query failed with exit code %d' % result.returncode)
x = json.loads(result.stdout)
assert x['anchors_sha256'] == source_sha and len(x['samples']) == 96 and len(x['csv_hashes']) == 16
assert len({(r['rank'],r['run'],r['label']) for r in x['samples']}) == 96
assert {r['label'] for r in x['samples']} == {'first','second'} and {r['run'] for r in x['samples']} == {1,2,3}
assert set(x['csv_hashes']) == {r['rank'] for r in x['samples']}
for r in x['samples']:
    assert r['model'] == '49' and r['start_ns'] < r['end_ns'] and r['fields']['Type'] in ('TransposeBatchMatMul','MatMulV2')
x['query_sha256'] = query_sha
x['wrapper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
x['no_device_work'] = True
body = (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:
    f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),
 'samples':len(x['samples']),'missing_columns':sorted({c for r in x['samples'] for c in r['missing_columns']})},ensure_ascii=False))
