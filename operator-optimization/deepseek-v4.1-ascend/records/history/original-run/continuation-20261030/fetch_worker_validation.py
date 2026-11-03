#!/usr/bin/env python3
"""Fetch only counters, procfs and anonymous timing fields from the failed stack run."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

source=r'''
import json,pathlib,hashlib
d=pathlib.Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/logs/worker-framework-20261030T105344Z-f556c759')
results=json.loads((d/'measure-results.json').read_text())
timing=[{k:x[k] for k in ('start_ns','start_wall_ns','end_ns','end_wall_ns','first_nonempty_ns','usage','chunks') if k in x} for x in results]
windows=json.loads((d/'sampler-windows.json').read_text())
out={'run_id':d.name,'request_timings':timing,'counters':json.loads((d/'measure-counters.json').read_text()),'proc':json.loads((d/'worker-proc.json').read_text()),'final_health':json.loads((d/'FINAL_HEALTH.json').read_text()),'samplers':{k:{a:v[a] for a in v if a!='argv'} for k,v in windows.items()},'stack_files_present':sorted(p.name for p in d.glob('*.speedscope.json')),'sha256':{n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ('measure-results.json','measure-counters.json','worker-proc.json','sampler-windows.json','FINAL_HEALTH.json')},'pyspy_stdout_tail':{p.name:p.read_text()[-1800:] for p in d.glob('pyspy-*.stdout')}}
print(json.dumps(out,indent=2))
'''
payload=base64.b64encode(source.encode()).decode()
code="import base64;exec(base64.b64decode('"+payload+"'))"
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-20261028','python3','-c',code])
root=Path(__file__).resolve().parent
with (root/'worker-validation.json').open('w') as out,(root/'worker-fetch.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=30,check=False)
print('returncode',result.returncode)
sys.exit(result.returncode)
