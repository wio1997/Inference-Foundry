#!/usr/bin/env python3
"""Select final HC/copy-compatible shapes in original 16-rank CSVs, read only."""
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
script=(ROOT/'analyze_head_sampler_exports.py').read_text()
before="""   vocab=any(v in shapes for v in ('129280','16160','16192'))
   name_match=bool(re.search(r'reject|argmax|arg_max|sample',t,re.I))
   if not (vocab or name_match):continue"""
after="""   tail_dims=bool(re.search(r'(?<![0-9,])(?:24,4,5120|24,20480|8192,20480|24,4)(?![0-9,])',shapes))
   if not tail_dims:continue"""
assert script.count(before)==1
script=script.replace(before,after).replace(
    'same official capture, later window, all16rank; exact shapes or sampler-name families',
    'same official capture, later window, all16rank; lexical full-token HC/state-copy compatible dimensions; not invocation or alias proof')
source='SPECS='+repr(specs)+'\n'+script
payload=base64.b64encode(source.encode()).decode()
code="import base64;exec(compile(base64.b64decode('"+payload+"'),'<tail_state_exports>','exec'))"
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code])
with (ROOT/'tail-state-exports.json').open('x') as out,(ROOT/'tail-state-analysis.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=90,check=False)
print(json.dumps({'returncode':result.returncode,'output':'tail-state-exports.json','read_only_existing_csvs':True}))
sys.exit(result.returncode)
