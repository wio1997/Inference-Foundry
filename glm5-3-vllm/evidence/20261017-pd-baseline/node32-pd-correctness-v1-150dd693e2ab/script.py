#!/usr/bin/env python3
"""Full-model cold-reference versus MemCache consumption and recovery checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import threading
import time
import urllib.request
import uuid

MODEL = 'glm53-w8a8c8-memcache'
URL = {32: 'http://172.16.12.32:29481', 33: 'http://172.16.12.33:29482'}


def http(node, path, payload=None, timeout=600):
    req = urllib.request.Request(URL[node]+path, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return raw.decode() if path == '/metrics' else (json.loads(raw) if raw else None)


def metrics(node):
    raw = http(node, '/metrics', timeout=10)
    values = {}
    for line in raw.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        key, value = line.rsplit(' ', 1)
        key = key.split('{', 1)[0]
        try:
            values[key] = values.get(key, 0)+float(value)
        except ValueError:
            pass
    return values


def reset(node):
    deadline = time.monotonic()+40
    while True:
        result = http(node, '/reset_prefix_cache?reset_external=false', {}, timeout=20)
        if result and result.get('success'):
            return
        assert time.monotonic() < deadline, ('Prefix reset still owns blocks', node, result)
        time.sleep(.2)


def idle():
    deadline = time.monotonic()+40
    while True:
        all_values = {node: metrics(node) for node in (32, 33)}
        wanted = ('vllm:num_requests_running', 'vllm:num_requests_waiting',
                  'vllm:ascend_store_delayed_release_requests', 'vllm:ascend_store_delayed_release_blocks')
        if all(all(m.get(k, 0) == 0 for k in wanted) for m in all_values.values()):
            return all_values
        assert time.monotonic() < deadline, all_values
        time.sleep(.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='correctness-v1')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.run_id)
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')/args.run_id
    root.mkdir(parents=True, exist_ok=False)
    report = {'started_host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Greedy token-ID integration comparisons to cold consumer prefill, not a production accuracy/SLO acceptance.',
              'cases': [], 'passed': False}
    nonce = uuid.uuid4().hex
    serial = 0
    serial_lock = threading.Lock()

    def save(name, obj):
        (root/name).write_text(json.dumps(obj, indent=2)+'\n')

    def tokenize(text, special=False):
        return http(32, '/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': special})['tokens']

    def prompt(length, case, a=2, b=3):
        header = tokenize('Reference '+nonce+' '+case+'. Read these validation notes.\n', True)
        body = tokenize('The notes preserve the order of the supplied facts. This is a deterministic memory transfer check.\n')
        end = tokenize(f'\nQuestion: What is {a} + {b}?\nAnswer: ')
        n = length-len(header)-len(end)
        assert n > 0 and body
        return header+(body*((n+len(body)-1)//len(body)))[:n]+end

    def complete(node, tokens, output, name, **extras):
        nonlocal serial
        with serial_lock:
            serial += 1
            this_serial = serial
        request = {'model': MODEL, 'prompt': tokens, 'max_tokens': output,
                   'temperature': 0, 'seed': 1024, 'ignore_eos': True,
                   'return_token_ids': True, 'stream': False, **extras}
        start = time.monotonic()
        response = http(node, '/v1/completions', request)
        result = {'node': node, 'request': request, 'response': response,
                  'elapsed_s_functional_only': time.monotonic()-start}
        save(f'{this_serial:03d}-{name}-node{node}.json', result)
        ids = response['choices'][0]['token_ids']
        assert isinstance(ids, list) and ids, response
        assert response['usage']['prompt_tokens'] == len(tokens), response['usage']
        return response

    def ids(response):
        return response['choices'][0]['token_ids']

    def wait_gets(before, positive):
        deadline = time.monotonic()+20
        while True:
            after = metrics(33)
            delta = after.get('vllm:ascend_store_load_get_keys_total', 0)-before
            if not positive or delta > 0:
                return delta, after
            if time.monotonic() >= deadline:
                raise AssertionError('No external KV get completed')
            time.sleep(.2)

    def wait_successes(target):
        deadline = time.monotonic()+25
        while True:
            values = metrics(33)
            if values.get('vllm:request_success_total', 0) >= target:
                return values
            assert time.monotonic() < deadline, ('Completion metrics did not advance', target, values)
            time.sleep(.2)

    def pair(case, tokens, output, **extras):
        reset(33)
        initial_metrics = metrics(33)
        before = initial_metrics.get('vllm:ascend_store_load_get_keys_total', 0)
        golden = complete(33, tokens, output, case+'-cold', **extras)
        reset(33)
        prepared = complete(32, tokens, output, case+'-producer', **extras)
        consumed = complete(33, tokens, output, case+'-pool', **extras)
        assert ids(golden) == ids(prepared) == ids(consumed), ('Token mismatch', case, ids(golden), ids(prepared), ids(consumed))
        assert golden['choices'][0]['text'] == consumed['choices'][0]['text'], ('Text mismatch', case)
        delta, _ = wait_gets(before, len(tokens) > 128)
        wait_successes(initial_metrics.get('vllm:request_success_total', 0)+2)
        item = {'case': case, 'input_tokens': len(tokens), 'output_tokens': len(ids(consumed)),
                'token_ids_equal': True, 'external_get_keys_delta': delta,
                'golden_id': golden['id'], 'producer_id': prepared['id'], 'consumer_id': consumed['id'],
                'finish_reason': consumed['choices'][0]['finish_reason']}
        report['cases'].append(item)
        save('progress.json', report)
        print(json.dumps(item), flush=True)
        return golden, consumed

    try:
        for node in (32, 33):
            assert any(m['id'] == MODEL for m in http(node, '/v1/models')['data'])
        for n in (127, 128, 129, 255, 256, 257):
            pair('boundary-'+str(n), prompt(n, 'boundary-'+str(n)), 32)
        pair('long-1024', prompt(1024, 'long-1024'), 64)
        pair('long-4096', prompt(4096, 'long-4096'), 128)
        golden, consumed = pair('repeat-prefix', prompt(513, 'repeat-prefix'), 64)
        before_repeat = metrics(33)
        repeat_tokens = json.loads(next(root.glob('*repeat-prefix-cold-node33.json')).read_text())['request']['prompt']
        repeated = complete(33, repeat_tokens, 64, 'repeat-local')
        assert ids(golden) == ids(repeated)
        after_repeat = wait_successes(before_repeat.get('vllm:request_success_total', 0)+1)
        repeat_gets = after_repeat.get('vllm:ascend_store_load_get_keys_total', 0)-before_repeat.get('vllm:ascend_store_load_get_keys_total', 0)
        assert repeat_gets == 0, 'Repeated local prefix unexpectedly reloaded external KV'
        report['cases'].append({'case': 'repeat-local', 'token_ids_equal': True, 'external_get_keys_delta': repeat_gets, 'consumer_id': repeated['id']})
        pair('stop-newline', prompt(513, 'stop-newline'), 64, stop=['\n'], ignore_eos=False)

        tokens_a = prompt(769, 'shared-prefix')
        tokens_b = tokens_a[:256]+prompt(513, 'distinct-tail', a=9, b=4)
        reset(33)
        golden_b = complete(33, tokens_b, 64, 'shared-reference')
        reset(33)
        complete(32, tokens_a, 1, 'shared-producer')
        before = metrics(33).get('vllm:ascend_store_load_get_keys_total', 0)
        consumed_b = complete(33, tokens_b, 64, 'shared-consumer')
        assert ids(golden_b) == ids(consumed_b), 'Shared prefix changed the uncached suffix output'
        delta, _ = wait_gets(before, True)
        assert delta == 32, ('Expected two shared blocks across sixteen ranks', delta)
        report['cases'].append({'case': 'shared-prefix-distinct-tail', 'common_prefix_tokens': 256,
                                'token_ids_equal': True, 'external_get_keys_delta': delta, 'consumer_id': consumed_b['id']})

        tokens4 = [prompt(1024, 'concurrent-'+str(i)) for i in range(4)]
        reset(33)
        with ThreadPoolExecutor(max_workers=4) as pool:
            refs = list(pool.map(lambda p: complete(33, p[1], 64, 'c4-ref-'+str(p[0])), enumerate(tokens4)))
        reset(33)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda p: complete(32, p[1], 1, 'c4-producer-'+str(p[0])), enumerate(tokens4)))
        before = metrics(33).get('vllm:ascend_store_load_get_keys_total', 0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(lambda p: complete(33, p[1], 64, 'c4-pool-'+str(p[0])), enumerate(tokens4)))
        assert all(ids(a) == ids(b) for a, b in zip(refs, outputs)), 'C4 token mismatch'
        delta, _ = wait_gets(before, True)
        report['cases'].append({'case': 'concurrent-4', 'requests': 4, 'token_ids_equal': True,
                                'external_get_keys_delta': delta, 'consumer_ids': [o['id'] for o in outputs]})

        cancel_tokens = prompt(1024, 'cancel')
        complete(32, cancel_tokens, 1, 'cancel-producer')
        payload = {'model': MODEL, 'prompt': cancel_tokens, 'max_tokens': 256, 'temperature': 0,
                   'seed': 1024, 'ignore_eos': True, 'return_token_ids': True, 'stream': True}
        request = urllib.request.Request(URL[33]+'/v1/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        received = []
        with urllib.request.urlopen(request, timeout=600) as stream:
            for line in stream:
                if line.startswith(b'data: ') and not line.startswith(b'data: [DONE]'):
                    chunk = json.loads(line[6:])
                    for choice in chunk.get('choices', []):
                        received.extend(choice.get('token_ids') or [])
                    if len(received) >= 8:
                        break
        assert 8 <= len(received) < 256
        reset(33)
        recovery = complete(33, prompt(257, 'recovery'), 16, 'after-cancel')
        report['cases'].append({'case': 'cancel-and-recover', 'partial_tokens': len(received), 'recovery_id': recovery['id'], 'reset_succeeded': True})
        report['idle_metrics'] = idle()
        report['passed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    finally:
        report['ended_host_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save('result.json', report)
        print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['passed'], report.get('failure')


if __name__ == '__main__':
    main()
