#!/usr/bin/env python3
"""Finite readiness observer: metadata GET only, no inference or parameter changes."""
import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-capacity/node33-tp16-w8a8c8-v1')


def main():
    container = json.loads((ROOT/'launch.json').read_text())['container_id']
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = ROOT/('readiness-watch-'+stamp+'.jsonl')
    deadline, previous, count = time.monotonic()+360, None, 0
    with path.open('x') as f:
        while True:
            p = subprocess.run(['docker', 'inspect', container, '--format', '{{json .State}}'],
                               capture_output=True, text=True, timeout=10)
            state = json.loads(p.stdout) if p.returncode == 0 else {'Status': 'inspect_error'}
            ready = False
            try:
                with urllib.request.urlopen('http://127.0.0.1:29480/v1/models', timeout=2) as response:
                    ready = any(m['id']=='glm53-w8a8c8-capacity' for m in json.load(response)['data'])
            except Exception:
                pass
            status = 'ready' if ready else ('failed' if state['Status']!='running' else 'loading')
            result = {'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'status': status, 'container_state': state, 'api_ready': ready,
                      'observer_log': str(path), 'poll_index': count}
            f.write(json.dumps(result)+'\n'); f.flush(); count += 1
            if status != previous:
                print(json.dumps({'status': status, 'host_utc': result['host_utc']}), flush=True)
                previous = status
            if status != 'loading' or time.monotonic() >= deadline:
                break
            time.sleep(10)
    (ROOT/('readiness-result-'+stamp+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN',flush=True)
    print(json.dumps(result),flush=True)
    print('GLM53_RESULT_JSON_END',flush=True)
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
