#!/usr/bin/env python3
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
original = json.loads((root/'graph-executions.json').read_text())
specs = []
for x in original['rows']:
    if x['window'] != 2:
        continue
    groups = [g for g in x['model_streams'] if all(e['main_shape_compatible_40_moe'] and e['dominant_signature'] and e['record_count'] == e['unique_task_id_count'] for e in g['executions'])]
    assert len(groups) == 1 and len(groups[0]['executions']) == 12
    specs.append({k:x[k] for k in ('rank','window','source_dir','kernel_sha256')})
    specs[-1]['executions'] = [{k:e[k] for k in ('start_epoch_ns','end_epoch_ns')} for e in groups[0]['executions']]
assert len(specs) == 16
script = 'BOUNDARIES='+repr(specs)+'\n'+(root/'reanalyze_iteration_gaps.py').read_text()
payload = base64.b64encode(script.encode()).decode()
code = "import base64;exec(compile(base64.b64decode('"+payload+"'),'<gap_reanalysis>','exec'))"
command = shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code])
start=time.time()
with (root/'iteration-gaps.json').open('w') as out, (root/'gap-reanalysis.stderr').open('w') as err:
    p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=180,check=False)
(root/'gap-reanalysis-receipt.json').write_text(json.dumps({'returncode':p.returncode,'started_epoch':start,'ended_epoch':time.time(),'no_device_work':True},indent=2)+'\n')
print('returncode',p.returncode)
sys.exit(p.returncode)
