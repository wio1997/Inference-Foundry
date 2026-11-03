#!/usr/bin/env python3
"""Run the bounded read-only query inside the existing staging container."""
from pathlib import Path
from hashlib import sha256
import json,shlex,subprocess
T=Path(__file__).resolve().parents[2]
S=T/'agents/diagnosis/query_layer0_dp_type_counters.py'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
D='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
out=T/'agents/diagnosis/layer0-dp-type-counters.json'
if out.exists():raise RuntimeError('STOP output exists')
command=shlex.join(['docker','exec',C,'python3','-c',S.read_text(),'--task-root',R,'--trace-root',D])
r=subprocess.run([SSH,command],capture_output=True,text=True,timeout=240)
if r.returncode!=0:raise RuntimeError('STOP remote query failed: '+r.stderr[-2000:])
if len(r.stdout)>1000000:raise RuntimeError('STOP oversized response')
x=json.loads(r.stdout)
if len(x['window_coverage'])!=48 or len(x['mc2_by_window'])!=48:raise RuntimeError('STOP incomplete result')
x['query_script_sha256']=sha256(S.read_bytes()).hexdigest();x['no_device_work']=True
out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'windows':48,'sha256':sha256(out.read_bytes()).hexdigest()}))
