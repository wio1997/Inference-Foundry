#!/usr/bin/env python3
import base64,pathlib,subprocess,shlex
r=pathlib.Path(__file__).resolve().parents[2]
b=base64.b64encode((r/'agents/fusion/query_f16_ep3_projection_provenance.py').read_bytes()).decode()
py='import base64;exec(compile(base64.b64decode('+repr(b)+'),"query.py","exec"))'
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',py])
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=40,check=True)
(r/'agents/fusion/f16-ep3-projection-provenance.json').write_text(p.stdout)
print('saved fixed EP3 projection provenance')
