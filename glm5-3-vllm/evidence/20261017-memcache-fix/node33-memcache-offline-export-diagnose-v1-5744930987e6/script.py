#!/usr/bin/env python3
"""Diagnose offline msprof export on a disposable copy, without NPU access."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess


def main():
    assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-node33-fixed-v6')
    out = root/'profile-export-diagnose-v1'
    out.mkdir(exist_ok=False)
    rank = next((root/'profile').glob('*tp15_*ascend_pt'))
    source = next(rank.glob('PROF_*'))
    shutil.copytree(source, out/'raw')
    command = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--cpus', '8', '--memory', '16g', '--pids-limit', '2048',
               '-v', str(out/'raw')+':/profile',
               '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
               '--entrypoint', 'msprof',
               'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83',
               '--export=on', '--output=/profile']
    proc = subprocess.run(command, capture_output=True, timeout=180)
    (out/'stdout.txt').write_bytes(proc.stdout)
    (out/'stderr.txt').write_bytes(proc.stderr)
    report = {'returncode': proc.returncode, 'argv': command, 'source': str(source),
              'stdout': proc.stdout.decode(errors='replace')[-12000:],
              'stderr': proc.stderr.decode(errors='replace')[-12000:],
              'stdout_sha256': hashlib.sha256(proc.stdout).hexdigest(),
              'stderr_sha256': hashlib.sha256(proc.stderr).hexdigest()}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
