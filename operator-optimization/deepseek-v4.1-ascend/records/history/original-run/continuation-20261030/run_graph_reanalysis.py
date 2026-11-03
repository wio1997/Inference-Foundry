#!/usr/bin/env python3
import base64
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
payload = base64.b64encode((root/'reanalyze_graph_executions.py').read_bytes()).decode()
code = "import base64;exec(compile(base64.b64decode('" + payload + "'),'<graph_reanalysis>','exec'))"
command = 'docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c "' + code + '"'
start = time.time()
with (root/'graph-executions.json').open('w') as out, (root/'graph-reanalysis.stderr').open('w') as err:
    p = subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh', command], stdout=out, stderr=err, timeout=180, check=False)
(root/'graph-reanalysis-receipt.json').write_text(json.dumps({'returncode': p.returncode, 'started_epoch': start, 'ended_epoch': time.time(), 'source': str(root/'reanalyze_graph_executions.py'), 'no_device_work': True}, indent=2)+'\n')
print('returncode', p.returncode)
sys.exit(p.returncode)
