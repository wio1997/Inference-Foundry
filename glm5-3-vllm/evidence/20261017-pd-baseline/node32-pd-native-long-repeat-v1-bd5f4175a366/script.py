#!/usr/bin/env python3
"""Test fresh native local-prefix references for the failed long-input fixture."""
import json
from pathlib import Path
import socket
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')
    fixture = json.loads((base/'baseline-c1-v1/fixture-4096-128.json').read_text())
    out = base/'native-long-repeat-v1'
    out.mkdir(exist_ok=False)
    result = {'scope': 'Fresh native D full prefill then native local-prefix reuse, no producer or new pool transfer; diagnose reproducibility independently of transport.', 'waves': []}
    for i in range(4):
        payload = {'model': 'glm53-w8a8c8-memcache', 'prompt': fixture['prompt'], 'max_tokens': 1,
                   'cache_salt': 'native-long-'+uuid.uuid4().hex, 'temperature': 0, 'seed': 1024,
                   'ignore_eos': True, 'return_token_ids': True, 'logprobs': 5, 'stream': False}
        values = []
        for phase, length in [('prime', 1), ('reuse', 128)]:
            payload['max_tokens'] = length
            req = urllib.request.Request('http://172.16.12.33:29482/v1/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=600) as response:
                value = json.load(response)
            (out/f'wave-{i}-{phase}.json').write_text(json.dumps({'request': payload, 'response': value}, indent=2)+'\n')
            values.append(value)
        choice = values[-1]['choices'][0]
        ids = choice['token_ids']
        row = {'wave': i, 'reference_ids_equal': ids == fixture['reference_ids'], 'response_id': values[-1]['id'],
               'first_difference': next((j for j,(a,b) in enumerate(zip(ids,fixture['reference_ids'])) if a!=b), None),
               'text_prefix': choice['text'][:180], 'first_ids': ids[:20]}
        result['waves'].append(row)
        print(json.dumps(row), flush=True)
    result['completed'] = True
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)


if __name__ == '__main__':
    main()
