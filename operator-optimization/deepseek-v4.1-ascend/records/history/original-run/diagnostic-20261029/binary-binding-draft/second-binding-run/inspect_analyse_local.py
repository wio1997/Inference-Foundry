#!/usr/bin/env python3
import subprocess,shlex,pathlib
p=pathlib.Path(__file__).resolve().parent
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
args=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',"from pathlib import Path; p=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/profiler.py'); s=p.read_text(); i=s.find('def analyse('); print(s[i:i+8000] if i>=0 else 'NO_ANALYSE_DEFINITION')"]
r=subprocess.run([ssh,shlex.join(args)],capture_output=True,text=True,check=True)
(p/'single-rank-analyse-source.txt').write_text(r.stdout)
print('captured',len(r.stdout),'bytes')
