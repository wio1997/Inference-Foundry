#!/usr/bin/env python3
"""Read the task-owned PD container and API state without changing execution."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import urllib.request


def main():
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    root = Path(f'/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-node{node}-fixed-v4')
    container = json.loads((root/'launch.json').read_text())['container_id']
    state = json.loads(subprocess.check_output(['docker', 'inspect', container, '--format', '{{json .State}}']))
    p = subprocess.run(['docker', 'logs', '--timestamps', container], capture_output=True, timeout=40)
    raw = p.stdout+b'\nDOCKER_STDERR_STREAM\n'+p.stderr
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    console = root/('console-'+stamp+'.txt')
    console.write_bytes(raw)
    lines = raw.decode(errors='replace').splitlines()
    important = [s for s in lines if re.search(r'ERROR|Traceback|Exception|GiB|Loading.*weights|Loaded model|Registering KV|Memcache|SMEM|SDMA|Application startup complete|Uvicorn running', s)]
    result = {'node': node, 'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'state': state, 'api_ready': False, 'console': {'path': str(console), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)},
              'important_tail': important[-55:], 'console_tail': lines[-12:]}
    try:
        port = 29481 if node == 32 else 29482
        with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/v1/models', timeout=3) as response:
            result['models'] = json.load(response)
        result['api_ready'] = any(m['id'] == 'glm53-w8a8c8-memcache' for m in result['models']['data'])
    except Exception as e:
        result['api_error'] = str(e)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
