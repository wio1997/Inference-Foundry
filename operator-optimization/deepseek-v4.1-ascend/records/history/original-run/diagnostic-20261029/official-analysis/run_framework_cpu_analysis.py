"""Recompute the anonymous CPU summary from existing stage-container files."""
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
source=(HERE/'analyze_framework_cpu.py').read_text()
compile(source,'analyze_framework_cpu.py','exec')
args=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',source]
r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=45)
if r.returncode:raise RuntimeError('offline analysis failed: '+r.stderr[-3000:])
data=json.loads(r.stdout)
(HERE/'framework-cpu.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print('Saved anonymous analysis for',data['run_id'])
