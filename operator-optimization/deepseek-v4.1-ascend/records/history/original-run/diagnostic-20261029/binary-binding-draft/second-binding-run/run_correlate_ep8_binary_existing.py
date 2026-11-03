from pathlib import Path
import base64,shlex,subprocess,sys
here=Path(__file__).resolve().parent
src=(here/'correlate_ep8_binary_existing.py').read_bytes();enc=base64.b64encode(src).decode()
inner='import base64;exec(compile(base64.b64decode('+repr(enc)+'),"<existing-ep8-audit>","exec"))'
outer='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec','wio-vllm-v41-repro169-stage-20261028','nice','-n','15','python3','-c',inner])+'))'
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',outer])],capture_output=True,text=True,timeout=120)
if p.returncode:print(p.stderr);raise SystemExit(p.returncode)
(here/'ep8-binary-existing-audit.json').write_text(p.stdout)
print(p.stdout)
