import argparse
import hashlib
import json
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True)
p.add_argument('--out', required=True)
p.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
a = p.parse_args()
rows = [json.loads(x) for x in Path(a.dataset).read_text().splitlines()]
results = []
for i in (0, 1, 6, 34):
    prompt = rows[i]['question']
    body = {'model': 'dsv4', 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': 128, 'ignore_eos': True}
    resp = requests.post(a.url, json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    message = data['choices'][0]['message']
    output = (message.get('reasoning') or '') + (message.get('content') or '')
    assert output
    results.append({'index': i, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'output_sha256': hashlib.sha256(output.encode()).hexdigest(),
                    'completion_tokens': data['usage']['completion_tokens'], 'output': output})
Path(a.out).write_text(json.dumps(results, indent=2))
print(json.dumps([{k: v for k, v in x.items() if k != 'output'} for x in results], indent=2))
