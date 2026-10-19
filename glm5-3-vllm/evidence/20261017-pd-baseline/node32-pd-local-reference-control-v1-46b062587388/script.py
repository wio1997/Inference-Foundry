#!/usr/bin/env python3
"""Distinguish transfer errors from cold-versus-local-prefix numerical behavior."""
import json
from pathlib import Path
import socket
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')
    source = base/'correctness-v3'
    out = base/'local-reference-control-v1'
    out.mkdir(exist_ok=False)
    records = {}
    for label, pattern in [('golden', '*boundary-255-cold-node33.json'),
                           ('producer', '*boundary-255-producer-node32.json'),
                           ('pool', '*boundary-255-pool-node33.json')]:
        records[label] = json.loads(next(source.glob(pattern)).read_text())
    results = []
    for label, original, node, cold in [('D-local-reference', 'golden', 33, False),
                                      ('P-local-reference', 'producer', 32, False),
                                      ('D-local-from-pool', 'pool', 33, False),
                                      ('D-cold-repeat', 'golden', 33, True),
                                      ('P-cold-repeat', 'producer', 32, True)]:
        payload = dict(records[original]['request'], logprobs=5)
        if cold:
            payload['cache_salt'] = 'fresh-'+uuid.uuid4().hex
        req = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node==32 else 29482}/v1/completions',
                                     data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            data = json.load(response)
        (out/(label+'.json')).write_text(json.dumps({'request': payload, 'response': data}, indent=2)+'\n')
        choice = data['choices'][0]
        results.append({'label': label, 'response_id': data['id'], 'text': choice['text'],
                        'token_ids': choice['token_ids'], 'first_logprobs': {
                            k: (v[:6] if isinstance(v, list) else v) for k, v in (choice.get('logprobs') or {}).items()}})
    report = {'original_cold_ids': records['golden']['response']['choices'][0]['token_ids'],
              'original_pool_ids': records['pool']['response']['choices'][0]['token_ids'],
              'results': results,
              'D_local_matches_pool': results[0]['token_ids'] == records['pool']['response']['choices'][0]['token_ids'],
              'D_cold_repeats_original': results[3]['token_ids'] == records['golden']['response']['choices'][0]['token_ids'],
              'P_local_matches_D_local': results[1]['token_ids'] == results[0]['token_ids'],
              'scope': 'Identical input/seed; salts isolate cold state only. Native local prefix and transferred prefix compared separately from cold full prefill. No precision criterion relaxed.'}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
