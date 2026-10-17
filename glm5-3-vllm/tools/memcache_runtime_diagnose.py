#!/usr/bin/env python3
"""CPU-only import/loader isolation; preserve real process exit codes and maps."""
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-diagnosis-v1')
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    ROOT.mkdir(parents=True, exist_ok=False)
    records = []
    for module in ('torch', 'memfabric_hybrid', 'memcache_hybrid'):
        code = "import " + module + "; import torch,pathlib,json; print('ABI',torch._C._GLIBCXX_USE_CXX11_ABI,flush=True); print(pathlib.Path('/proc/self/maps').read_text(),flush=True)"
        argv = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
                '--read-only', '--tmpfs', '/tmp:rw,size=512m', '--cpus', '2', '--memory', '4g',
                '--pids-limit', '512', '-e', 'OMP_NUM_THREADS=1',
                '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
                '--entrypoint', 'python3', IMAGE, '-c', code]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=90)
        (ROOT/(module+'.stdout')).write_text(p.stdout)
        (ROOT/(module+'.stderr')).write_text(p.stderr)
        libraries = sorted(set(line.split()[-1] for line in p.stdout.splitlines()
                               if '/' in line and any(n in line for n in ('memfabric', 'memcache', 'libstdc++'))))
        records.append({'module': module, 'rc': p.returncode, 'libraries': libraries,
                        'abi': [s for s in p.stdout.splitlines() if s.startswith('ABI')],
                        'stderr_tail': p.stderr[-2500:]})
    (ROOT/'result.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
