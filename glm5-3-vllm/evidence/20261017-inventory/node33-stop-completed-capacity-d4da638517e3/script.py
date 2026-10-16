#!/usr/bin/env python3
"""Stop only the identified task-owned completed capacity diagnostic; preserve container/logs."""
import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-capacity/node33-tp16-w8a8c8-v1')


def main():
    launch = json.loads((ROOT/'launch.json').read_text())
    container = launch['container_id']
    inspected = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
    assert inspected['Name'] == '/glm53-exec-capacity-33-20261017'
    assert inspected['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    result = {'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'container_id': container, 'before': inspected['State'],
              'reason': 'capacity API ready and one functional request complete; release devices for isolated TE and PD validation'}
    stopped = subprocess.run(['docker', 'stop', '-t', '30', container], capture_output=True, text=True, timeout=40)
    result['stop'] = {'rc': stopped.returncode, 'stdout': stopped.stdout, 'stderr': stopped.stderr}
    inspected = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
    result['after'] = inspected['State']
    (ROOT/'stop.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert stopped.returncode == 0 and not inspected['State']['Running']


if __name__ == '__main__':
    main()
