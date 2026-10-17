#!/usr/bin/env python3
"""Export retained profiles using CPU and driver queries, without compute devices."""
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
    out = root/'offline-analysis-v2'
    out.mkdir(exist_ok=False)
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    command = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
               '--name', f'glm53-profile-analysis-{node}-v2-20261017',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--cpus', '16', '--memory', '64g', '--pids-limit', '4096',
               '--tmpfs', '/tmp:rw,size=2g', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '--device', '/dev/davinci_manager',
               '-v', str(root/'profile')+':/profile',
               '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
               '--entrypoint', '/usr/bin/timeout', image,
               '--signal=TERM', '--kill-after=10', '1050', 'python3', '-c',
               "from torch_npu.profiler.profiler import analyse; analyse('/profile', max_process_number=4, export_type='text')"]
    started = time.monotonic()
    with (out/'stdout.txt').open('xb') as stdout, (out/'stderr.txt').open('xb') as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=1100)
    traces = []
    for path in sorted((root/'profile').rglob('trace_view.json')):
        item = {'path': str(path)}
        try:
            parsed = json.loads(path.read_bytes())
            events = parsed.get('traceEvents', []) if isinstance(parsed, dict) else parsed
            item['sdma_sqe_count'] = sum(e.get('name') == 'SDMA_SQE' and e.get('ph') == 'X' for e in events)
            item['valid_json'] = True
            del parsed, events
        except ValueError as exc:
            item['valid_json'] = False
            item['error'] = str(exc)
        traces.append(item)
    result = {'node': node, 'ended_host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'returncode': proc.returncode, 'elapsed_s': time.monotonic()-started,
              'argv': command, 'compute_devices': [], 'driver_query_devices': ['/dev/davinci_manager'],
              'files': {name: {'path': str(out/name), 'bytes': (out/name).stat().st_size,
                               'sha256': hashlib.sha256((out/name).read_bytes()).hexdigest()}
                        for name in ('stdout.txt', 'stderr.txt')},
              'traces': traces, 'exported_trace_count': len(traces)}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert proc.returncode == 0 and result['exported_trace_count'] == 16
    assert all(t['valid_json'] and t.get('sdma_sqe_count', 0) > 0 for t in traces)


if __name__ == '__main__':
    main()
