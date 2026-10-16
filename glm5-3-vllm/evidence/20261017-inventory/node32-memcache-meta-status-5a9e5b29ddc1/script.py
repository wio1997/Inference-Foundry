#!/usr/bin/env python3
"""Read only the task-owned metadata container state and logs; no service mutation."""
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/meta32-v1')


def main():
    launch = json.loads((ROOT/'launch.json').read_text())
    container = launch['container_id']
    inspected = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
    assert inspected['Name'] == '/glm53-memcache-meta32-20261017'
    assert inspected['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    logs = subprocess.run(['docker', 'logs', container], capture_output=True, text=True, timeout=15)
    ready = {}
    for port in (27400, 27401, 27402):
        try:
            with socket.create_connection(('172.16.12.32', port), timeout=1):
                pass
            ready[port] = True
        except OSError:
            ready[port] = False
    print(json.dumps({'state': inspected['State'], 'ports': ready,
                      'logs': logs.stdout+logs.stderr}, indent=2))


if __name__ == '__main__':
    main()
