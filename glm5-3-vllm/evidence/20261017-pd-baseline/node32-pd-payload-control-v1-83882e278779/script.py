#!/usr/bin/env python3
"""Observe real KV control checksums at matched native/PD cache boundaries."""
import hashlib
import json
from pathlib import Path
import pickle
import socket
import time
import urllib.request
import uuid

BASE = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')


def rpc(node, payload):
    request = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/v1/completions',
                                     data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def keys(tokens, salt):
    digest = lambda value: hashlib.sha256(pickle.dumps(value, protocol=5)).digest()
    parent = digest('vllm-none-hash')
    result = []
    for start in range(0, len(tokens), 128):
        parent = digest((parent, tuple(tokens[start:start+128]), (salt,) if start == 0 else None))
        result.append('GLM-5.3-w8a8c8@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@'+parent.hex())
    return result


def records():
    result = []
    for path in (BASE/'pd-node32-v11/kv-control-audit').glob('buffers-*.jsonl'):
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                pass  # A concurrently appended final line is retried.
    return result


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    out = BASE/'tests/payload-control-v1'
    out.mkdir(exist_ok=False)
    fixture = json.loads((BASE/'tests/baseline-c1-v1/fixture-4096-128.json').read_text())
    report = {'waves': [], 'completed': False,
              'scope': 'Four original4096/128 inputs; matched native no-logprob reference. Read-only NPU integer-moment observation and explicit source commit wait; not cryptographic proof, a production fix or performance benchmark.'}

    def infer(node, payload, name):
        value = rpc(node, payload)
        (out/(name+'.json')).write_text(json.dumps({'request': payload, 'response': value}, indent=2)+'\n')
        return value

    try:
        for wave in range(4):
            payload = {'model': 'glm53-w8a8c8-memcache', 'prompt': fixture['prompt'], 'max_tokens': 1,
                       'temperature': 0, 'seed': 1024, 'ignore_eos': True,
                       'return_token_ids': True, 'stream': False,
                       'cache_salt': 'native-'+uuid.uuid4().hex}
            infer(33, payload, f'wave-{wave}-native-prime')
            payload['max_tokens'] = 128
            native = infer(33, payload, f'wave-{wave}-native-reference')
            salt = 'pool-'+uuid.uuid4().hex
            payload.update(cache_salt=salt, max_tokens=1)
            started = time.monotonic()
            producer = infer(32, payload, f'wave-{wave}-producer')
            expected = keys(fixture['prompt'], salt)
            deadline = time.monotonic()+180
            while True:
                after = [r for r in records() if r['phase'] == 'put_after' and r['key'] in expected]
                committed = {r['key'] for r in after}
                if committed == set(expected):
                    break
                assert time.monotonic() < deadline, ('Source audit incomplete', len(committed), len(expected))
                time.sleep(.5)
            source_ready = time.monotonic()
            payload['max_tokens'] = 128
            pool = infer(33, payload, f'wave-{wave}-pool')
            row = {'wave': wave, 'producer_id': producer['id'], 'native_id': native['id'], 'consumer_id': pool['id'],
                   'source_keys_verified_committed': len(committed),
                   'pool_matches_current_native': pool['choices'][0]['token_ids'] == native['choices'][0]['token_ids'],
                   'native_matches_original': native['choices'][0]['token_ids'] == fixture['reference_ids'],
                   'producer_and_commit_wait_seconds': source_ready-started,
                   'functional_PD_seconds_including_commit_wait': time.monotonic()-started,
                   'expected_pool_keys': expected}
            report['waves'].append(row)
            (out/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k != 'expected_pool_keys'}), flush=True)
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
