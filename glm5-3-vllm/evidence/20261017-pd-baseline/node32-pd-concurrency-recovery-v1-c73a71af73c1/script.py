#!/usr/bin/env python3
"""Repeat native C4 independently of transfer, then gate the protected C1 scope."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import socket
import time
import urllib.request
import uuid

BASE = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')


def rpc(node, path, payload=None):
    request = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}'+path,
                                    data=None if payload is None else json.dumps(payload).encode(),
                                    headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=600)


def metrics(node):
    with rpc(node, '/metrics') as response:
        raw = response.read().decode()
    result = {}
    for line in raw.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        key, value = line.rsplit(' ', 1)
        key = key.split('{', 1)[0]
        result[key] = result.get(key, 0)+float(value)
    return result


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    source = BASE/'correctness-v4'
    out = BASE/'correctness-c1-gate-v1'
    out.mkdir(exist_ok=False)
    previous = json.loads((source/'result.json').read_text())
    assert previous['failure'] == 'AssertionError: C4 token mismatch'
    assert len(previous['cases']) == 12
    assert all(c['token_ids_equal'] for c in previous['cases'])
    refs = [json.loads(next(source.glob(f'*c4-local-reference-{i}-node33.json')).read_text()) for i in range(4)]
    report = {'scope': 'C1 strict transfer correctness plus cancellation/recovery. C4 remains excluded from accepted performance; native repeated C4 is a diagnostic, not a relaxed accuracy gate.',
              'source_result': str(source/'result.json'), 'native_c4_waves': [], 'passed': False}

    def complete(node, payload, name):
        with rpc(node, '/v1/completions', payload) as response:
            data = json.load(response)
        (out/(name+'.json')).write_text(json.dumps({'request': payload, 'response': data}, indent=2)+'\n')
        return data

    def ids(data):
        return data['choices'][0]['token_ids']

    try:
        for wave in range(3):
            with ThreadPoolExecutor(max_workers=4) as pool:
                values = list(pool.map(lambda i: complete(33, refs[i]['request'], f'native-c4-{wave}-{i}'), range(4)))
            record = {'wave': wave, 'equal_original_native_ids': [ids(a) == ids(b['response']) for a, b in zip(values, refs)],
                      'response_ids': [v['id'] for v in values]}
            report['native_c4_waves'].append(record)
            print(json.dumps(record), flush=True)
        tokens = json.loads(next(source.glob('*long-1024-cold-node33.json')).read_text())['request']['prompt']
        salt = 'cancel-'+uuid.uuid4().hex
        payload = dict(refs[0]['request'], prompt=tokens, max_tokens=1, cache_salt=salt)
        producer = complete(32, payload, 'cancel-producer')
        payload.update(max_tokens=2048, stream=True)
        payload.pop('logprobs', None)
        received = []
        active = False
        next_poll = time.monotonic()+1
        with rpc(33, '/v1/completions', payload) as stream:
            for line in stream:
                if line.startswith(b'data: ') and not line.startswith(b'data: [DONE]'):
                    chunk = json.loads(line[6:])
                    for choice in chunk.get('choices', []):
                        received.extend(choice.get('token_ids') or [])
                    if len(received) >= 8 and time.monotonic() >= next_poll:
                        next_poll = time.monotonic()+1
                        active = metrics(33).get('vllm:num_requests_running', 0) > 0
                        if active:
                            break
        assert active and len(received) < 2048
        started = time.monotonic()
        deadline = started+40
        keys = ('vllm:num_requests_running', 'vllm:num_requests_waiting',
                'vllm:ascend_store_delayed_release_requests', 'vllm:ascend_store_delayed_release_blocks')
        while True:
            values = {node: metrics(node) for node in (32, 33)}
            if all(all(v.get(k, 0) == 0 for k in keys) for v in values.values()):
                break
            assert time.monotonic() < deadline, values
            time.sleep(.2)
        idle_s = time.monotonic()-started
        original = json.loads(next(source.glob('*boundary-257-local-reference-node33.json')).read_text())
        recovery = complete(33, original['request'], 'after-cancel')
        assert ids(recovery) == ids(original['response']), 'Recovery changed protected C1 output'
        report.update(cancel_partial_tokens=len(received), active_before_cancel=True,
                      cancel_to_idle_s=idle_s, idle_metrics=values, recovery_id=recovery['id'],
                      recovery_exact_ids=True, producer_id=producer['id'], c4_validated=False,
                      cases=previous['cases'], passed=True)
        for path in source.glob('*-local-reference-node33.json'):
            shutil.copyfile(path, out/path.name)
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['passed'], report.get('failure')


if __name__ == '__main__':
    main()
