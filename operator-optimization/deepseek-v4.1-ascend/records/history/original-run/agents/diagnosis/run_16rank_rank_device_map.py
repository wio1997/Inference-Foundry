#!/usr/bin/env python3
"""Run the fixed read-only query inside the existing staging container."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

root = Path(__file__).resolve().parents[2]
script = root / 'agents/diagnosis/query_16rank_rank_device_map.py'
ssh = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
container = 'wio-vllm-v41-repro169-stage-20261028'
remote_task = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
remote_trace = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
output = root / 'agents/diagnosis/16rank-rank-device-map.json'
if output.exists():
    raise SystemExit('STOP: output already exists')
source = script.read_text()
cmd = shlex.join(['docker', 'exec', container, 'python3', '-c', source,
                  '--task-root', remote_task, '--trace-root', remote_trace])
result = subprocess.run([ssh, cmd], capture_output=True, text=True, timeout=180)
if result.returncode:
    raise RuntimeError('read-only query failed: ' + result.stderr[:2000])
payload = json.loads(result.stdout)
if len(payload.get('samples', [])) != 16:
    raise RuntimeError('expected 16 DB results')
payload['script_sha256'] = hashlib.sha256(source.encode()).hexdigest()
payload['no_npu_work'] = True
# Exclusive create prevents accidental replacement of a previous result.
with output.open('x', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'status': 'QUERY_COMPLETED', 'output': str(output), 'databases': 16}))
