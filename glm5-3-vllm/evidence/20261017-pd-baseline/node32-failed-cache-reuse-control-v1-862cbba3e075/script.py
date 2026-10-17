#!/usr/bin/env python3
"""Reuse failed transferred prefixes after handoff, without loading KV again."""
import json
from pathlib import Path
import socket
import urllib.request


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')
    source = base/'final-put-immediate-control-v1'
    original = json.loads((source/'result.json').read_text())
    assert original['completed']
    out = base/'failed-cache-reuse-control-v1'
    out.mkdir(exist_ok=False)

    def metrics(label):
        with urllib.request.urlopen('http://172.16.12.33:29482/metrics', timeout=10) as response:
            raw = response.read().decode()
        (out/(label+'.txt')).write_text(raw)
        result = {}
        for line in raw.splitlines():
            if not line or line.startswith('#'):
                continue
            key, value = line.rsplit(' ', 1)
            key = key.split('{', 1)[0]
            result[key] = result.get(key, 0)+float(value)
        return result

    def infer(node, payload, label):
        request = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/v1/completions',
                                         data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=300) as response:
            value = json.load(response)
        (out/(label+'.json')).write_text(json.dumps({'request': payload, 'response': value}, indent=2)+'\n')
        return value

    report = {'completed': False, 'waves': [], 'scope': 'Same failed salts and existing P/D cached prefixes. Numerical eight-token queries and bare128 D-local repeat; no model load or precision change. Verify zero additional SDK GET and actual local-prefix hit counters before interpreting persistence.'}
    try:
        for wave in original['waves']:
            if wave['pool_matches_P_native']:
                continue
            i = wave['wave']
            stored_pool = json.loads((source/f'wave-{i}-pool.json').read_text())
            stored_p = json.loads((source/f'wave-{i}-P-native-reference.json').read_text())
            payload = stored_pool['request']
            before = metrics(f'wave-{i}-before')
            scores = {}
            for label, node in [('P-local', 32), ('D-pool-local', 33)]:
                scores[label] = infer(node, dict(payload, max_tokens=8, logprobs=20), f'wave-{i}-{label}-scores')['choices'][0]
            repeated = infer(33, payload, f'wave-{i}-D-pool-local-repeat')
            after = metrics(f'wave-{i}-after')
            a, b = scores.values()
            differences = [abs(x[key]-y[key]) for x, y in zip(a['logprobs']['top_logprobs'], b['logprobs']['top_logprobs']) for key in set(x)&set(y)]
            ids = repeated['choices'][0]['token_ids']
            row = {'wave': i, 'repeat_id': repeated['id'], 'repeat_matches_original_pool': ids == stored_pool['response']['choices'][0]['token_ids'],
                   'repeat_matches_stored_P_native': ids == stored_p['response']['choices'][0]['token_ids'],
                   'local_P_vs_D_sampled_ids_equal': a['token_ids'] == b['token_ids'],
                   'local_P_vs_D_top20_max_delta': max(differences),
                   'D_sdk_get_keys_delta': after.get('vllm:ascend_store_load_get_keys_total', 0)-before.get('vllm:ascend_store_load_get_keys_total', 0),
                   'D_local_prefix_hits_delta': after.get('vllm:prefix_cache_hits_total', 0)-before.get('vllm:prefix_cache_hits_total', 0)}
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
