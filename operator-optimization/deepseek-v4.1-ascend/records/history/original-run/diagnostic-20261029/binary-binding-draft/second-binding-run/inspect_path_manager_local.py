#!/usr/bin/env python3
import subprocess,shlex,pathlib
p=pathlib.Path(__file__).resolve().parent
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
args=['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',"from pathlib import Path; p=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler'); files=[x for x in p.rglob('*.py') if 'class ProfilerPathManager' in x.read_text(errors='ignore')]; print('FILES',*[str(x) for x in files]); [(print('SOURCE',x),print(x.read_text()[x.read_text().find('class ProfilerPathManager'):][:14000])) for x in files]"]
r=subprocess.run([ssh,shlex.join(args)],capture_output=True,text=True,check=True)
(p/'profiler-path-manager-source.txt').write_text(r.stdout)
print('captured',len(r.stdout),'bytes')
