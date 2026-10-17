#!/usr/bin/env python3
"""Test fresh native local-prefix references for the failed long-input fixture."""
import argparse
import json
from pathlib import Path
import socket
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('native', 'delayed-pool', 'producer-reference', 'handoff-abba'), default='native')
    parser.add_argument('--run-id', default='native-long-repeat-v2')
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')
    fixture = json.loads((base/'baseline-c1-v1/fixture-4096-128.json').read_text())
    out = base/args.run_id
    out.mkdir(exist_ok=False)
    result = {'mode': args.mode, 'scope': 'Native fresh-prefix control or fixed one-second P-to-D quiescence diagnostic. No performance gain claim; all waiting is retained.', 'waves': []}
    count = 12 if args.mode == 'handoff-abba' else (3 if args.mode == 'producer-reference' else (4 if args.mode == 'native' else 6))
    for i in range(count):
        wait_seconds = 1 if args.mode == 'delayed-pool' or (args.mode == 'handoff-abba' and i//3 in (1, 2)) else 0
        started = time.monotonic()
        payload = {'model': 'glm53-w8a8c8-memcache', 'prompt': fixture['prompt'], 'max_tokens': 1,
                   'cache_salt': 'native-long-'+uuid.uuid4().hex, 'temperature': 0, 'seed': 1024,
                   'ignore_eos': True, 'return_token_ids': True, 'stream': False}
        values = []
        phases = [('prime', 1), ('reuse', 128)]
        if args.mode == 'producer-reference':
            phases.append(('producer-native', 128))
        for phase, length in phases:
            payload['max_tokens'] = length
            node = 32 if ((args.mode != 'native' and phase == 'prime') or phase == 'producer-native') else 33
            req = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/v1/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=600) as response:
                value = json.load(response)
            (out/f'wave-{i}-{phase}.json').write_text(json.dumps({'request': payload, 'response': value}, indent=2)+'\n')
            values.append(value)
            if wait_seconds and phase == 'prime':
                time.sleep(wait_seconds)
        choice = values[-1]['choices'][0]
        ids = choice['token_ids']
        row = {'wave': i, 'reference_ids_equal': ids == fixture['reference_ids'], 'response_id': values[-1]['id'],
               'handoff_wait_seconds': wait_seconds, 'functional_e2e_seconds_including_wait': time.monotonic()-started,
               'first_difference': next((j for j,(a,b) in enumerate(zip(ids,fixture['reference_ids'])) if a!=b), None),
               'text_prefix': choice['text'][:180], 'first_ids': ids[:20]}
        if args.mode == 'producer-reference':
            pool_ids = values[1]['choices'][0]['token_ids']
            row.update(producer_native_matches_pool=ids == pool_ids,
                       pool_matches_D_native=pool_ids == fixture['reference_ids'],
                       producer_vs_pool_first_difference=next((j for j,(a,b) in enumerate(zip(ids,pool_ids)) if a!=b), None))
        result['waves'].append(row)
        print(json.dumps(row), flush=True)
    result['completed'] = True
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)


if __name__ == '__main__':
    main()
