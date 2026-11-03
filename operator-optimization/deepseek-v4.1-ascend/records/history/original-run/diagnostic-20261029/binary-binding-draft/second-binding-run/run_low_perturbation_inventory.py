from pathlib import Path
import base64,shlex,subprocess
p=Path(__file__).resolve().parent;enc=base64.b64encode((p/'low_perturbation_inventory.py').read_bytes()).decode();inner='import base64;exec(compile(base64.b64decode('+repr(enc)+'),"<inventory>","exec"))';outer='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec','wio-vllm-v41-repro169-stage-20261028','nice','-n','15','python3','-c',inner])+'))';r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',outer])],capture_output=True,text=True,timeout=60)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
(p/'low-perturbation-inventory.json').write_text(r.stdout);print('inventory saved')
