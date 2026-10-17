#!/usr/bin/env python3
"""Bounded read-only API readiness watch for the two task-owned model instances."""
import datetime
import json
from pathlib import Path
import socket
import time
import urllib.request


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = root/('pd-readiness-watch-'+stamp+'.jsonl')
    deadline = time.monotonic()+1000
    previous = None
    with path.open('x') as out:
        while True:
            result = {'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'nodes': {}}
            for node, port in ((32, 29481), (33, 29482)):
                try:
                    with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/v1/models', timeout=2) as response:
                        data = json.load(response)
                    ready = any(m['id'] == 'glm53-w8a8c8-memcache' for m in data['data'])
                    result['nodes'][node] = {'ready': ready}
                except Exception as exc:
                    result['nodes'][node] = {'ready': False, 'error': str(exc)}
            out.write(json.dumps(result)+'\n')
            out.flush()
            state = tuple(v['ready'] for v in result['nodes'].values())
            if state != previous:
                print(json.dumps(result), flush=True)
                previous = state
            if all(state) or time.monotonic() >= deadline:
                break
            time.sleep(10)
    result['observer_log'] = str(path)
    result['both_ready'] = all(v['ready'] for v in result['nodes'].values())
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result)+'\nGLM53_RESULT_JSON_END', flush=True)
    return 0 if result['both_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
