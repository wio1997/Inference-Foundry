#!/usr/bin/env python3
import base64
from pathlib import Path
import shlex
import subprocess

p=Path(__file__).resolve().parent
payload=base64.b64encode((p/'trace_ep8_flow.py').read_bytes()).decode()
code='import base64; exec(compile(base64.b64decode('+repr(payload)+'), "trace_ep8_flow.py", "exec"))'
args=['docker','exec','wio-vllm-v41-repro169-stage-20261028','ionice','-c','3','nice','-n','19','python3','-c',code]
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(args)],capture_output=True,text=True,timeout=900)
(p/'ep8-layer14-flow-audit.json').write_text(r.stdout)
(p/'ep8-layer14-flow-run.stderr').write_text(r.stderr)
print('single-rank flow audit exit',r.returncode)
raise SystemExit(r.returncode)
