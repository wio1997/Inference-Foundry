#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
import time
import urllib.request

root = Path(os.environ['WIO_ROOT']) / 'projects/910c-deepseek-v4.1-flash'
out = root / 'runs' / datetime.now().astimezone().strftime('smoke-%Y%m%dT%H%M%S%z')
out.mkdir(parents=True, exist_ok=True)
base = 'http://127.0.0.1:8172'
def get(path):
    with urllib.request.urlopen(base + path, timeout=30) as response:
        data = response.read().decode()
        (out / (path.strip('/').replace('/', '-') + '.json')).write_text(data or '{}')
        return data
get('/health')
models = json.loads(get('/v1/models'))
assert any(x['id'] == 'dsv41' for x in models['data']), models
request = {'model': 'dsv41', 'messages': [{'role': 'user', 'content': '请直接回答：1加1等于几？'}], 'temperature': 0, 'max_tokens': 128, 'stream': False}
(out / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
start = time.monotonic()
req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=600) as response:
    result = json.load(response)
(out / 'response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
message = result['choices'][0]['message']
content = message.get('content') or ''
assert content.strip(), result
summary = {'status': 'PASS', 'elapsed_seconds': time.monotonic() - start, 'content': content, 'usage': result.get('usage'), 'scope': 'health, model registration and one text generation only; not an accuracy or performance qualification'}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('EVIDENCE_DIR=' + str(out))
