#!/usr/bin/env python3
"""Observe matched native/PD top probabilities at the first divergent prefix."""
import hashlib
import json
from pathlib import Path
import pickle
import socket
import time
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')
    fixture = json.loads((base/'baseline-c1-v1/fixture-4096-128.json').read_text())
    out = base/'numeric-boundary-control-v1'
    out.mkdir(exist_ok=False)
    report = {'completed': False, 'waves': [], 'scope': '4096 input, eight output tokens, top20 logprobs. This is a separate numerical diagnostic, not the bare128-token regression or a throughput benchmark. P observer remains enabled, D checksum budget exhausted; model/quantization unchanged.'}

    def infer(node, payload, label):
        request = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/v1/completions',
                                         data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=300) as response:
            value = json.load(response)
        (out/(label+'.json')).write_text(json.dumps({'request': payload, 'response': value}, indent=2)+'\n')
        return value

    try:
        for wave in range(6):
            payload = {'model': 'glm53-w8a8c8-memcache', 'prompt': fixture['prompt'], 'max_tokens': 1,
                       'temperature': 0, 'seed': 1024, 'ignore_eos': True,
                       'return_token_ids': True, 'stream': False, 'cache_salt': 'native-numeric-'+uuid.uuid4().hex}
            infer(33, payload, f'wave-{wave}-native-prime')
            payload.update(max_tokens=8, logprobs=20)
            native = infer(33, payload, f'wave-{wave}-native-reference')
            payload.update(max_tokens=1, cache_salt='pool-numeric-'+uuid.uuid4().hex)
            payload.pop('logprobs')
            producer = infer(32, payload, f'wave-{wave}-producer')
            digest = lambda value: hashlib.sha256(pickle.dumps(value, protocol=5)).digest()
            parent = digest('0')
            expected = set()
            for offset in range(0, len(fixture['prompt']), 128):
                parent = digest((parent, tuple(fixture['prompt'][offset:offset+128]),
                                 (payload['cache_salt'],) if offset == 0 else None))
                expected.add('GLM-5.3-w8a8c8@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@'+parent.hex())
            deadline = time.monotonic()+180
            while True:
                committed = set()
                for path in (base.parent/'pd-node32-v11/kv-control-audit').glob('buffers-*.jsonl'):
                    for line in path.read_text().splitlines():
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if row['phase'] == 'put_after' and row['key'] in expected:
                            committed.add(row['key'])
                if committed == expected:
                    break
                assert time.monotonic() < deadline, ('Source audit incomplete', len(committed), len(expected))
                time.sleep(.2)
            payload.update(max_tokens=8, logprobs=20)
            pool = infer(33, payload, f'wave-{wave}-pool')
            a, b = native['choices'][0], pool['choices'][0]
            first = next((i for i, (x, y) in enumerate(zip(a['token_ids'], b['token_ids'])) if x != y), None)
            positions = range(8) if first is None else range(first+1)
            row = {'wave': wave, 'native_id': native['id'], 'producer_id': producer['id'], 'consumer_id': pool['id'],
                   'pool_matches_current_native': a['token_ids'] == b['token_ids'], 'first_difference': first,
                   'verified_source_committed_keys': len(committed),
                   'native_ids': a['token_ids'], 'pool_ids': b['token_ids'],
                   'same_prefix_probabilities': [{'position': i, 'native_top20': a['logprobs']['top_logprobs'][i],
                                                  'pool_top20': b['logprobs']['top_logprobs'][i],
                                                  'native_chosen_logprob': a['logprobs']['token_logprobs'][i],
                                                  'pool_chosen_logprob': b['logprobs']['token_logprobs'][i]} for i in positions]}
            report['waves'].append(row)
            (out/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k: v for k, v in row.items() if k != 'same_prefix_probabilities'}), flush=True)
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
