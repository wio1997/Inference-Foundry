#!/usr/bin/env python3
"""Exercise the first P request after disabling the documented C8/AICPU conflict."""
import datetime
import json
from pathlib import Path
import socket
import time
import urllib.request


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-node32-fixed-v7/p-arithmetic-smoke')
    url = 'http://172.16.12.32:29481'
    with urllib.request.urlopen(url+'/v1/models', timeout=5) as response:
        assert any(m['id'] == 'glm53-w8a8c8-memcache' for m in json.load(response)['data'])
    root.mkdir(exist_ok=False)
    payload = {'model': 'glm53-w8a8c8-memcache', 'prompt': 'Question: What is 2 + 3?\nAnswer:',
               'max_tokens': 8, 'temperature': 0, 'seed': 1024, 'stream': False}
    report = {'started_host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'request': payload, 'completed': False}
    (root/'request.json').write_text(json.dumps(payload, indent=2)+'\n')
    started = time.monotonic()
    try:
        request = urllib.request.Request(url+'/v1/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            report['http_status'] = response.status
            report['response'] = json.load(response)
        text = report['response']['choices'][0]['text']
        assert text.strip().startswith('5'), 'Arithmetic smoke did not start with the expected result'
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    report['elapsed_s'] = time.monotonic()-started
    report['ended_host_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
