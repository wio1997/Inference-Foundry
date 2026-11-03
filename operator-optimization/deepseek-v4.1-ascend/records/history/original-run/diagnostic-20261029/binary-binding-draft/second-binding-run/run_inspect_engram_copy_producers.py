from pathlib import Path
import base64,shlex,subprocess
here=Path(__file__).resolve().parent
src=(here/'inspect_engram_copy_producers.py').read_bytes();enc=base64.b64encode(src).decode();inner='import base64;exec(compile(base64.b64decode('+repr(enc)+'),"<engram-copy-producers>","exec"))';outer='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec','wio-vllm-v41-repro169-stage-20261028','nice','-n','15','python3','-c',inner])+'))';r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',outer])],capture_output=True,text=True,timeout=600)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
(here/'engram-copy-producers.json').write_text(r.stdout);print('audit saved',len(r.stdout),'bytes')
