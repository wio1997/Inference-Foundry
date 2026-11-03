#!/usr/bin/env python3
"""ZCode launches this reviewed, single-purpose own-container offline job."""
import base64
from pathlib import Path
import shlex
import subprocess

here = Path(__file__).resolve().parent
ssh = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
stage = 'wio-vllm-v41-repro169-stage-20261028'
script = (here / 'offline_ep8_db.py').read_bytes()
payload = base64.b64encode(script).decode('ascii')
code = 'import base64; exec(compile(base64.b64decode(' + repr(payload) + '), "offline_ep8_db.py", "exec"))'
command = ['docker', 'exec', '-e', 'OMP_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
           '-e', 'MKL_NUM_THREADS=1', stage, 'ionice', '-c', '3', 'nice', '-n', '19', 'python3', '-u', '-c', code]
with (here / 'offline-ep8-db.stdout').open('w') as out, (here / 'offline-ep8-db.stderr').open('w') as err:
    p = subprocess.run([ssh, shlex.join(command)], stdout=out, stderr=err, timeout=2400)
(here / 'offline-ep8-db.exit').write_text(str(p.returncode) + '\n')
print('single-rank CPU-only offline parser exit', p.returncode)
raise SystemExit(p.returncode)
