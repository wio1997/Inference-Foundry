#!/usr/bin/env python3
"""Observe cached KV only through native reuse requests after unobserved D128."""
import json
from pathlib import Path
import socket
import time
import urllib.request
import uuid


CASES = [{'label': 'post-D-content-zero-block18', 'cache_salt': 'post-pool-08bfa01a29824dd988b90fe7aad155c5', 'archive': 'node32-completed-pd-tests-post-consume-final-v1.tar.gz', 'member': 'tests/post-consume-cache-control-v1/wave-3-producer.json'}, {'label': 'final-PUT-output-mismatch-wave2', 'cache_salt': 'final-put-a0f67ad1200f4aa3a9091c8813eaa711', 'archive': 'node32-completed-pd-tests-final-put-rejected-v1.tar.gz', 'member': 'tests/final-put-immediate-control-v1/wave-2-producer.json'}, {'label': 'final-PUT-output-mismatch-wave9', 'cache_salt': 'final-put-85ace47f5a5c4ea0a3edf6777b0353e9', 'archive': 'node32-completed-pd-tests-final-put-rejected-v1.tar.gz', 'member': 'tests/final-put-immediate-control-v1/wave-9-producer.json'}]


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    out = base/'tests/post-pool-state-control-v1'
    out.mkdir(exist_ok=False)
    fixture = json.loads((base/'tests/baseline-c1-v1/fixture-4096-128.json').read_text())
    deadline = time.monotonic()+1500
    report = {'completed': False, 'waves': [], 'scope': 'Four initial unfenced immediate P1 to D128 paths, post-D actual cache and independent private-arena pool readback on D rank0; private SDK reads are separate from loader counters, then existing request_id-tagged native P128 and D-local8 controls. No KV observation before/during first PUT/GET. Original D golden retained. P FIFO drain is a diagnostic boundary guard, not an adopted repair. Integer moments are not cryptographic payload hashes. No throughput acceptance.'}

    def rpc(node, payload):
        headers = {'Content-Type': 'application/json'}
        if 'request_id' in payload:
            headers['X-Request-Id'] = payload['request_id']
        request = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/v1/completions',
                                         data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=min(300, max(1, deadline-time.monotonic()))) as response:
            return json.load(response)

    def save(label, payload, value):
        (out/(label+'.json')).write_text(json.dumps({'request': payload, 'response': value}, indent=2)+'\n')

    def metrics(node, label):
        with urllib.request.urlopen(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/metrics', timeout=10) as response:
            raw = response.read().decode()
        (out/(label+'.txt')).write_text(raw)
        values = {}
        for line in raw.splitlines():
            if not line or line.startswith('#'):
                continue
            key, value = line.rsplit(' ', 1)
            key = key.split('{', 1)[0]
            values[key] = values.get(key, 0)+float(value)
        return values

    def source_controls(request_id):
        rows = []
        for path in (base/'pd-node32-v15/kv-control-audit').glob('buffers-*.jsonl'):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        return [row for row in rows if row['phase']=='native_prefix' and row['request_id'].startswith(request_id+'-')]

    try:
        ready_deadline = time.monotonic()+600
        while True:
            ready = []
            for node, port in [(32, 29481), (33, 29482)]:
                try:
                    with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/v1/models', timeout=2) as response:
                        value = json.load(response)
                    ready.append(any(x['id'] == 'glm53-w8a8c8-memcache' for x in value['data']))
                except Exception:
                    ready.append(False)
            if all(ready):
                break
            assert time.monotonic() < ready_deadline, 'Bounded API readiness wait exhausted'
            time.sleep(5)
        print('Both API ready; starting post-D private pool-state controls', flush=True)
        native_payload = {'model': 'glm53-w8a8c8-memcache', 'prompt': fixture['prompt'], 'max_tokens': 1,
                          'temperature': 0, 'seed': 1024, 'ignore_eos': True, 'return_token_ids': True,
                          'stream': False, 'cache_salt': 'post-native-'+uuid.uuid4().hex}
        value = rpc(33, native_payload)
        save('D-native-prime', native_payload, value)
        native_payload['max_tokens'] = 128
        native = rpc(33, native_payload)
        save('D-native-reference', native_payload, native)
        report['native_D_matches_original'] = native['choices'][0]['token_ids'] == fixture['reference_ids']
        old = json.loads((base/'tests/source-ready-abba-v1/D-native-reference.json').read_text())['response']
        report['new_D_native_matches_previous_D13_native'] = native['choices'][0]['token_ids'] == old['choices'][0]['token_ids']
        for wave in range(4):
            assert deadline-time.monotonic() > 90, 'Insufficient bounded lifetime for another complete pair'
            case = CASES[wave] if wave < len(CASES) else {'label':'new-fixed-shape-key','cache_salt':'pool-state-new-'+uuid.uuid4().hex}
            payload = dict(native_payload, max_tokens=1, cache_salt=case['cache_salt'],
                           request_id='glm53-source-control-'+uuid.uuid4().hex)
            producer = rpc(32, payload)
            consumer_payload = dict(payload, max_tokens=128)
            consumer_payload.pop('request_id',None)
            pool = rpc(33, consumer_payload)
            save(f'wave-{wave}-producer', payload, producer)
            save(f'wave-{wave}-pool', consumer_payload, pool)
            p_before = metrics(32, f'wave-{wave}-P-before-native')
            p_payload = dict(consumer_payload, request_id='glm53-postkv-source-'+uuid.uuid4().hex)
            p_native = rpc(32, p_payload)
            save(f'wave-{wave}-P-native-reference', p_payload, p_native)
            p_after = metrics(32, f'wave-{wave}-P-after-native')
            source = source_controls(p_native['id'])
            assert len(source) == 31*16, ('Post-consume source controls incomplete', len(source))
            assert {row['tp_rank'] for row in source} == set(range(16))
            d_before = metrics(33, f'wave-{wave}-D-before-local')
            d_payload = dict(consumer_payload, max_tokens=8, logprobs=20,
                             request_id='glm53-postkv-target-'+uuid.uuid4().hex)
            d_local = rpc(33, d_payload)
            save(f'wave-{wave}-D-pool-local', d_payload, d_local)
            d_after = metrics(33, f'wave-{wave}-D-after-local')
            row = {'wave': wave, 'source_fence_selected':False, 'case':case, 'producer_id': producer['id'], 'consumer_id': pool['id'],
                   'P_native_id': p_native['id'], 'D_local_id': d_local['id'],
                   'pool_matches_P_native': pool['choices'][0]['token_ids'] == p_native['choices'][0]['token_ids'],
                   'pool_matches_original_D_reference': pool['choices'][0]['token_ids'] == fixture['reference_ids'],
                   'pool_matches_current_D_native': pool['choices'][0]['token_ids'] == native['choices'][0]['token_ids'],
                   'source_control_records': len(source),
                   'P_local_prefix_hits_delta': p_after.get('vllm:prefix_cache_hits_total', 0)-p_before.get('vllm:prefix_cache_hits_total', 0),
                   'D_local_prefix_hits_delta': d_after.get('vllm:prefix_cache_hits_total', 0)-d_before.get('vllm:prefix_cache_hits_total', 0),
                   'D_local_model_loader_get_keys_delta': d_after.get('vllm:ascend_store_load_get_keys_total', 0)-d_before.get('vllm:ascend_store_load_get_keys_total', 0)}
            report['waves'].append(row)
            (out/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            assert row['P_local_prefix_hits_delta'] == row['D_local_prefix_hits_delta'] == 3968
            assert row['D_local_model_loader_get_keys_delta'] == 0
        reference_payload = dict(native_payload, max_tokens=8, logprobs=20,
                                 request_id='glm53-postkv-D-native-'+uuid.uuid4().hex)
        value = rpc(33, reference_payload)
        save('D-native-local-control', reference_payload, value)
        report['D_native_control_id'] = value['id']
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
