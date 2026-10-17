#!/usr/bin/env python3
"""Close this task's completed metadata diagnostic and retain the final log."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/meta32-v1')


def main():
    launch = json.loads((ROOT/'launch.json').read_text())
    container = launch['container_id']
    inspected = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
    assert inspected['Name'] == '/glm53-memcache-meta32-20261017'
    assert inspected['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    stopped = subprocess.run(['docker', 'stop', '-t', '15', container], capture_output=True, text=True, timeout=25)
    logs = subprocess.run(['docker', 'logs', container], capture_output=True, timeout=15)
    data = logs.stdout+b'\nDOCKER_STDERR_STREAM\n'+logs.stderr
    path = ROOT/'console-final.txt'
    path.write_bytes(data)
    after = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]['State']
    report = {'container_id': container, 'stop_rc': stopped.returncode, 'before': inspected['State'],
              'after': after, 'console': {'path': str(path), 'bytes': len(data),
                                         'sha256': hashlib.sha256(data).hexdigest()},
              'reason': 'bounded transport diagnostics complete; no model PD started; native exit lifecycle remains unresolved'}
    (ROOT/'stop.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert stopped.returncode == 0 and not after['Running']


if __name__ == '__main__':
    main()
