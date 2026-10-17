#!/usr/bin/env python3
"""Start the unchanged pinned official proxy in a bounded CPU-only container."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--revision',type=int,choices=range(1,9),default=1);parser.add_argument('--lifetime-seconds',type=int,choices=(3600,21000),default=3600);args=parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    root = Path(f'/data/wio/glm5-3-vllm-execution/20261017-official-pd/proxy-v{args.revision}')
    root.mkdir(parents=True, exist_ok=False)
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    source = '/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py'
    model = json.loads((root.parent/f'prefill-node32-v{args.revision}/launch.json').read_text())['container_id']
    sha = subprocess.check_output(['docker', 'exec', model, 'sha256sum', source], text=True).split()[0]
    assert sha == '85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469'
    name = f'glm53-official-proxy-32-v{args.revision}-20261017'
    assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
    with socket.socket() as sock:
        sock.bind(('172.16.12.32', 29880))
    argv = ['docker', 'run', '-d', '--pull', 'never', '--name', name, '--network', 'host',
            '--read-only', '--tmpfs', '/tmp:rw,size=128m', '--cpus', '4', '--memory', '4g',
            '--pids-limit', '256', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1',
            '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '--entrypoint', '/usr/bin/timeout', image, '--signal=TERM', '--kill-after=15', str(args.lifetime_seconds),
            'python3', source, '--host', '172.16.12.32', '--port', '29880',
            '--prefiller-hosts', '172.16.12.32', '--prefiller-ports', '29481',
            '--decoder-hosts', '172.16.12.33', '--decoder-ports', '29482']
    cid = subprocess.check_output(argv, text=True, timeout=30).strip()
    report = {'container_id': cid, 'argv': argv, 'source_sha256': sha, 'source_unmodified': True,
              'compute_devices': [], 'ready': False, 'scope': 'Official CPU proxy process, not proof of ready model backends or remote KV transfer.'}
    (root/'launch.json').write_text(json.dumps(report, indent=2)+'\n')
    for unused in range(30):
        try:
            with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck', timeout=2) as response:
                report.update(ready=True, health=json.load(response)); break
        except Exception:
            time.sleep(1)
    logs = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=20)
    raw = logs.stdout+b'\nSTDERR\n'+logs.stderr
    (root/'console.txt').write_bytes(raw)
    report['console_sha256'] = hashlib.sha256(raw).hexdigest()
    report['console_tail'] = raw.decode(errors='replace').splitlines()[-15:]
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['ready']


if __name__ == '__main__':
    main()
