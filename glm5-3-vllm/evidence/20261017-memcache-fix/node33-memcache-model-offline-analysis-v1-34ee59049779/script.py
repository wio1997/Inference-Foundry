#!/usr/bin/env python3
"""Export retained Ascend raw profiles in an isolated CPU-only pinned-image container."""
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time


def main():
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    revision = 7 if node == 32 else 6
    root = Path(f'/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-node{node}-fixed-v{revision}')
    out = root/'offline-analysis-v1'
    out.mkdir(exist_ok=False)
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    command = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
               '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--cpus', '4', '--memory', '32g', '--pids-limit', '1024',
               '--tmpfs', '/tmp:rw,size=2g', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-v', str(root/'profile')+':/profile',
               '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
               '--entrypoint', 'python3', image, '-c',
               "from torch_npu.profiler.profiler import analyse; analyse('/profile', max_process_number=2, export_type='text')"]
    started = time.monotonic()
    with (out/'stdout.txt').open('xb') as stdout, (out/'stderr.txt').open('xb') as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=1100)
    result = {'node': node, 'ended_host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'returncode': proc.returncode, 'elapsed_s': time.monotonic()-started,
              'argv': command, 'npu_devices': [],
              'files': {name: {'path': str(out/name), 'bytes': (out/name).stat().st_size,
                               'sha256': hashlib.sha256((out/name).read_bytes()).hexdigest()}
                        for name in ('stdout.txt', 'stderr.txt')},
              'exported_trace_count': len(list((root/'profile').rglob('trace_view.json')))}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert proc.returncode == 0 and result['exported_trace_count'] == 16


if __name__ == '__main__':
    main()
