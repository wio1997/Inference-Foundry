#!/usr/bin/env python3
"""Dispatch fixed CPU-only query inside existing dedicated staging container."""
from pathlib import Path
import hashlib, json, shlex, subprocess

T = Path(__file__).resolve().parents[2]
D = T / 'agents/diagnosis'
inp = D / 'draft-six-mc2-exported-fields.json'
dist = D / '16rank-draft-alltoall-distribution.json'
query = D / 'query_draft_mc2_preceding_csv.py'
out = D / 'draft-six-mc2-preceding-csv.json'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
if sha(inp) != '0a703b6028f1ffbace5e085a48714f07e05ff914189a199d0cdde1d374bd6568' or sha(dist) != '633011028e6d5953f03f1493c197085bdd1a810c095cc646e4f5b0e115f61ec6' or out.exists():
    raise RuntimeError('input identity mismatch or output exists')
d = json.loads(inp.read_text())
ranks = {}
for sample in d['samples']:
    ranks.setdefault(sample['rank'], []).append(sample)
if sorted(map(len, ranks.values())) != [3,3]:
    raise RuntimeError('unexpected selection')
prior = json.loads(dist.read_text())['file_sha256_by_rank']
if any(prior[rank]['csv'] != d['csv_sha256_by_rank'][rank] for rank in ranks):
    raise RuntimeError('frozen CSV summaries disagree')
spec = {'ranks': ranks, 'csv_sha256_by_rank': d['csv_sha256_by_rank'],
        'db_sha256_by_rank': {rank: prior[rank]['db'] for rank in ranks}}
code = query.read_text().replace('FROZEN_INPUT', repr(json.dumps(spec, separators=(',', ':'))))
ssh = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
container = 'wio-vllm-v41-repro169-stage-20261028'
trace = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r = subprocess.run([ssh, shlex.join(['docker','exec',container,'python3','-c',code,'--trace-root',trace])], capture_output=True, text=True, timeout=180)
if r.returncode:
    raise RuntimeError('query failed: ' + r.stderr[-2000:])
result = json.loads(r.stdout)
result['input_sha256'] = {str(inp.relative_to(T)): sha(inp), str(dist.relative_to(T)): sha(dist)}
result['query_sha256'] = sha(query)
result['wrapper_sha256'] = sha(Path(__file__))
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'sha256':sha(out)}))
