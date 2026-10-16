#!/usr/bin/env python3
"""One short completion against the identified task-owned capacity service."""
import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-capacity/node33-tp16-w8a8c8-v1')
BASE = 'http://127.0.0.1:29480'


def main():
    launch = json.loads((ROOT/'launch.json').read_text())
    container = launch['container_id']
    inspected = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
    assert inspected['State']['Running'] and not inspected['State']['OOMKilled']
    assert inspected['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    with urllib.request.urlopen(BASE+'/v1/models', timeout=5) as r:
        models = json.load(r)
    assert any(m['id'] == 'glm53-w8a8c8-capacity' for m in models['data'])
    request = {'model': 'glm53-w8a8c8-capacity', 'prompt': 'Question: What is 2 + 3?\nAnswer:',
               'max_tokens': 32, 'temperature': 0, 'seed': 1024, 'stream': False}
    start = time.perf_counter_ns()
    req = urllib.request.Request(BASE+'/v1/completions', data=json.dumps(request).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
        status = response.status
    elapsed = (time.perf_counter_ns()-start)/1e9
    report = {'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'container_id': container, 'request': request, 'http_status': status,
              'response': result, 'client_elapsed_s': elapsed,
              'scope': 'single functional request; not an accuracy or performance acceptance test'}
    path = ROOT/'functional-1.json'
    assert not path.exists(), 'Do not overwrite an earlier request'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert status == 200 and result['usage']['completion_tokens'] > 0
    assert result['choices'] and result['choices'][0]['text']


if __name__ == '__main__':
    main()
