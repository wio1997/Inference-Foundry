#!/usr/bin/env python3
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
old=json.loads((ROOT.parent/'diagnostic-20261029/official-analysis/operators.json').read_text())
specs=[{'rank':x['rank'],'source_dir':x['source_dir'],'sha256':x['sha256']['kernel_details.csv']} for x in old['rows'] if x['window_ordinal']==2]
assert len(specs)==16
source='SPECS='+repr(specs)+'\n'+(ROOT/'analyze_head_sampler_exports.py').read_text()
payload=base64.b64encode(source.encode()).decode();code="import base64;exec(compile(base64.b64decode('"+payload+"'),'<head_sampler_exports>','exec'))"
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code])
with (ROOT/'head-sampler-exports.json').open('w') as out,(ROOT/'head-sampler-analysis.stderr').open('w') as err:
 p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=90,check=False)
print('returncode',p.returncode);sys.exit(p.returncode)
