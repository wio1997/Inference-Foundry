#!/usr/bin/env python3
"""Snapshot only the launched node33 capacity container; no inference requests or service changes."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-capacity/node33-tp16-w8a8c8-v1')


def command(argv, timeout=20):
    p = subprocess.run(argv, capture_output=True, timeout=timeout)
    return {'argv': argv, 'rc': p.returncode, 'stdout': p.stdout.decode(errors='replace'),
            'stderr': p.stderr.decode(errors='replace')}


def main():
    launch = json.loads((ROOT/'launch.json').read_text())
    container = launch['container_id']
    identity = command(['docker', 'inspect', container, '--format', '{{.Image}}'])
    assert identity['stdout'].strip() == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    state = command(['docker', 'inspect', container, '--format', '{{json .State}}'])
    logs = command(['docker', 'logs', '--timestamps', container], timeout=30)
    # Both streams are retained: Docker can send container stderr to client stderr.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    console = ROOT/('console-'+stamp+'.txt')
    data = (logs['stdout'] + '\nDOCKER_STDERR_STREAM\n' + logs['stderr']).encode()
    console.write_bytes(data)
    result = {'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'container_id': container, 'state': json.loads(state['stdout']),
              'console': {'path': str(console), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)},
              'console_tail': data.decode(errors='replace').splitlines()[-100:],
              'npu_smi': command(['npu-smi', 'info']), 'api_ready': False}
    try:
        with urllib.request.urlopen('http://127.0.0.1:29480/v1/models', timeout=3) as response:
            models = json.load(response)
        result['api_models'] = models
        result['api_ready'] = any(m['id'] == 'glm53-w8a8c8-capacity' for m in models['data'])
    except Exception as exc:
        result['api_error'] = type(exc).__name__ + ': ' + str(exc)
    (ROOT/('snapshot-'+stamp+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
