#!/usr/bin/env python3
"""Run fixed read-only stage inventory, then exclusively write task result."""
from pathlib import Path
import hashlib, json, shlex, subprocess

T = Path(__file__).resolve().parents[2]
S = T / 'agents/diagnosis/query_mc2_saved_dump_inventory.py'
O = T / 'agents/diagnosis/mc2-saved-dump-inventory.json'
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER = 'wio-vllm-v41-repro169-stage-20261028'
REMOTE_TASK = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
REMOTE_TRACE = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
EXPECTED = 'b2c6df7126ff56f0c54a4ec90df1bf815a634a53c0911162e430df7d9435fb53'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

if digest(S) != EXPECTED:
    raise RuntimeError('STOP_QUERY_SCRIPT_HASH_MISMATCH')
if O.exists():
    raise RuntimeError('STOP_RESULT_ALREADY_EXISTS')
cmd = ['docker','exec',CONTAINER,'python3','-c',S.read_text(), '--task-root',REMOTE_TASK,'--trace-root',REMOTE_TRACE]
result = subprocess.run([SSH, shlex.join(cmd)], capture_output=True, timeout=180)
if result.returncode != 0:
    raise RuntimeError('STOP_READONLY_STAGE_QUERY_FAILED')
if len(result.stdout) > 2_000_000:
    raise RuntimeError('STOP_RESULT_TOO_LARGE')
payload = json.loads(result.stdout)
if payload.get('no_device_work') is not True or len(payload.get('samples', [])) != 2:
    raise RuntimeError('STOP_RESULT_SCOPE_MISMATCH')
payload['query_script_sha256'] = EXPECTED
payload['wrapper_script_sha256'] = digest(Path(__file__).resolve())
with O.open('x') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status':'QUERY_COMPLETED','samples':len(payload['samples']), 'output':str(O), 'output_sha256':digest(O), 'no_device_work':True}))
