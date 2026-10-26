import argparse
import json
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument('--golden', required=True)
p.add_argument('--out', required=True)
p.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
a = p.parse_args()
long_rows = json.loads(Path(a.golden).read_text())
long_ok = len(long_rows) == 4 and all(
    r['completion_tokens'] == 128 and len(r['output']) > 100 and len(r['prompt_sha256']) == 64
    for r in long_rows
)
cases = [
    ('Return only the number 42.', '42'),
    ('Reply with exactly OK.', 'OK'),
    ('What is 2+2? Reply with only the numeral.', '4'),
]
short = []
for prompt, expected in cases:
    response = requests.post(a.url, json={
        'model': 'dsv4', 'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0, 'max_tokens': 32,
    }, timeout=300)
    response.raise_for_status()
    data = response.json()
    content = data['choices'][0]['message'].get('content') or ''
    short.append({'prompt': prompt, 'expected': expected, 'content': content,
                  'completion_tokens': data['usage']['completion_tokens'],
                  'pass': content.strip() == expected})
result = {'pass': long_ok and all(x['pass'] for x in short),
          'long_128_token_outputs_valid': long_ok, 'short': short}
Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not result['pass']:
    raise SystemExit(1)
