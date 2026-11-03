from pathlib import Path
import base64,shlex,subprocess
p=Path(__file__).resolve().parent;enc=base64.b64encode((p/'inspect_low_stack_ep8_copy.py').read_bytes()).decode();inner='import base64;exec(compile(base64.b64decode('+repr(enc)+'),"<low-stack-ep8>","exec"))';outer='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec','wio-vllm-v41-repro169-stage-20261028','nice','-n','15','python3','-c',inner])+'))';r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',outer])],capture_output=True,text=True,timeout=600)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
(p/'low-stack-ep8-copy-audit.json').write_text(r.stdout);print('saved bounded audit bytes',len(r.stdout))
