#!/usr/bin/env python3
"""Hash-check original later-window DispatchFFNCombine interface families only."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
W=Path(__file__).resolve().parent
old=json.loads((W.parent/'diagnostic-20261029/official-analysis/operators.json').read_text())
specs=[{'rank':x['rank'],'source_dir':x['source_dir'],'sha256':x['sha256']['kernel_details.csv']} for x in old['rows'] if x['window_ordinal']==2]
assert len(specs)==16
script=(W/'analyze_head_sampler_exports.py').read_text()
before="""   vocab=any(v in shapes for v in ('129280','16160','16192'))
   name_match=bool(re.search(r'reject|argmax|arg_max|sample',t,re.I))
   if not (vocab or name_match):continue"""
assert script.count(before)==1
script=script.replace(before,"   if t != 'DispatchFFNCombine':continue")
script=script.replace('same official capture, later window, all16rank; exact shapes or sampler-name families',
    'same official capture later window all16rank; DispatchFFNCombine interface shapes/dtypes/counts only; no tensor values')
payload=base64.b64encode(('SPECS='+repr(specs)+'\n'+script).encode()).decode()
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',"import base64;exec(base64.b64decode('"+payload+"'))"])
with (W/'moe-interface-exports.json').open('x') as out,(W/'moe-interface-analysis.stderr').open('w') as err:
    result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],stdout=out,stderr=err,timeout=90,check=False)
print('returncode',result.returncode);sys.exit(result.returncode)
