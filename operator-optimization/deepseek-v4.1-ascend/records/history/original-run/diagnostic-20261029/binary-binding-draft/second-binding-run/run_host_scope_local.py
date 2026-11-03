#!/usr/bin/env python3
import base64, pathlib, shlex, subprocess
root=pathlib.Path(__file__).resolve().parent
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
container='wio-vllm-v41-repro169-stage-20261028'
remote='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029'
script=remote+'/binary-binding-draft/second-binding-run/correlate_host_scopes.py'
meta=remote+'/binding-diagnostic-selective2/runtime/metadata'
out=remote+'/binary-binding-draft/second-binding-run/host-scope-correlation.json'
encoded=base64.b64encode((root/'correlate_host_scopes.py').read_bytes()).decode('ascii')
put='import base64,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.write_bytes(base64.b64decode(sys.argv[2]))'
def run(args):
 cmd=' '.join(shlex.quote(x) for x in ['docker','exec',container]+args)
 result=subprocess.run([ssh,cmd],check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 return result.stdout
run(['python3','-c',put,script,encoded])
clock=run(['python3','-c','import json,time,sys; print(json.dumps({"python":sys.version,"executable":sys.executable,"perf_counter":time.get_clock_info("perf_counter").__dict__,"monotonic":time.get_clock_info("monotonic").__dict__}))'])
print('STAGE_CLOCK',clock.strip())
print(run(['nice','-n','15','python3',script,meta,'--output',out]).strip())
summary=run(['python3','-c','import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())',out])
(root/'host-scope-correlation.json').write_text(summary)
(root/'host-scope-stage-clock.json').write_text(clock)
