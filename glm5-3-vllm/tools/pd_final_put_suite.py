#!/usr/bin/env python3
"""Validate immediate handoff after final-PUT drain, retaining both native references."""
import json
from pathlib import Path
import socket
import time
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')
    fixture = json.loads((base/'baseline-c1-v1/fixture-4096-128.json').read_text())
    out = base/'final-put-immediate-control-v1'
    out.mkdir(exist_ok=False)
    report = {'completed': False, 'waves': [], 'scope': 'No payload observer. Immediate P1 HTTP completion to D128 submission; no sleep or key polling. P-native128 reference runs after D, preserving the handoff test. Current D-native and original protected D reference remain separate. This is correctness/completion evidence, not throughput acceptance or complete precision proof.'}

    def rpc(node, payload):
        request = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/v1/completions',
                                         data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)

    def save(label, payload, value):
        (out/(label+'.json')).write_text(json.dumps({'request': payload, 'response': value}, indent=2)+'\n')

    payload = {'model': 'glm53-w8a8c8-memcache', 'prompt': fixture['prompt'], 'max_tokens': 1,
               'temperature': 0, 'seed': 1024, 'ignore_eos': True, 'return_token_ids': True,
               'stream': False, 'cache_salt': 'D-native-final-'+uuid.uuid4().hex}
    try:
        prime = rpc(33, payload)
        save('D-native-prime', payload, prime)
        payload['max_tokens'] = 128
        native_d = rpc(33, payload)
        save('D-native-reference', payload, native_d)
        report['native_D_reference_id'] = native_d['id']
        report['native_D_matches_original'] = native_d['choices'][0]['token_ids'] == fixture['reference_ids']
        for wave in range(12):
            producer_payload = dict(payload, max_tokens=1, cache_salt='final-put-'+uuid.uuid4().hex)
            started = time.monotonic()
            producer = rpc(32, producer_payload)
            producer_done = time.monotonic()
            consumer_payload = dict(producer_payload, max_tokens=128)
            consumer_submit = time.monotonic()
            consumer = rpc(33, consumer_payload)
            consumer_done = time.monotonic()
            # Persist after handoff so evidence I/O cannot insert a key-publication wait.
            save(f'wave-{wave}-producer', producer_payload, producer)
            save(f'wave-{wave}-pool', consumer_payload, consumer)
            native_p = rpc(32, consumer_payload)
            save(f'wave-{wave}-P-native-reference', consumer_payload, native_p)
            ids = consumer['choices'][0]['token_ids']
            p_ids = native_p['choices'][0]['token_ids']
            row = {'wave': wave, 'producer_id': producer['id'], 'consumer_id': consumer['id'],
                   'P_native_id': native_p['id'], 'pool_matches_P_native': ids == p_ids,
                   'pool_matches_current_D_native': ids == native_d['choices'][0]['token_ids'],
                   'pool_matches_original_D_reference': ids == fixture['reference_ids'],
                   'P_native_matches_original_D_reference': p_ids == fixture['reference_ids'],
                   'producer_HTTP_seconds': producer_done-started,
                   'handoff_client_gap_seconds': consumer_submit-producer_done,
                   'functional_PD_seconds': consumer_done-started,
                   'P_vs_pool_first_difference': next((i for i, (x, y) in enumerate(zip(ids, p_ids)) if x != y), None)}
            if wave < 2:
                scores = {}
                score_payload = dict(consumer_payload, max_tokens=8, logprobs=20)
                for label, node in [('P-native-local-scores', 32), ('D-pool-local-scores', 33)]:
                    value = rpc(node, score_payload)
                    save(f'wave-{wave}-{label}', score_payload, value)
                    scores[label] = value['choices'][0]
                a, b = scores.values()
                differences = [abs(x[key]-y[key]) for x, y in zip(a['logprobs']['top_logprobs'], b['logprobs']['top_logprobs']) for key in set(x)&set(y)]
                row['sampled_local_top20_max_logprob_delta'] = max(differences)
                row['sampled_local_token_ids_equal'] = a['token_ids'] == b['token_ids']
            report['waves'].append(row)
            (out/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(row), flush=True)
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
