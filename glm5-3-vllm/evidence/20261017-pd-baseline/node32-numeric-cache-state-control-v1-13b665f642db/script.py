#!/usr/bin/env python3
"""Compare saved numerical controls against existing P/D local cache reuse."""
import json
from pathlib import Path
import socket
import urllib.request


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests')
    source = base/'numeric-boundary-control-v1'
    out = base/'numeric-cache-state-control-v1'
    out.mkdir(exist_ok=False)
    records = {name: json.loads((source/f'wave-5-{name}.json').read_text())
               for name in ['native-reference', 'producer', 'pool']}
    values = {'original-D-native': records['native-reference']['response'],
              'original-D-pool': records['pool']['response']}
    for label, original, node in [('P-local', 'producer', 32), ('D-native-local', 'native-reference', 33), ('D-pool-local', 'pool', 33)]:
        payload = dict(records[original]['request'], max_tokens=8, logprobs=20)
        request = urllib.request.Request(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/v1/completions',
                                         data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=300) as response:
            value = json.load(response)
        (out/(label+'.json')).write_text(json.dumps({'request': payload, 'response': value}, indent=2)+'\n')
        values[label] = value
    comparisons = []
    labels = list(values)
    for i, left in enumerate(labels):
        for right in labels[i+1:]:
            a, b = values[left]['choices'][0], values[right]['choices'][0]
            positions = []
            for j, (x, y) in enumerate(zip(a['logprobs']['top_logprobs'], b['logprobs']['top_logprobs'])):
                common = set(x)&set(y)
                positions.append({'position': j, 'max_common_top20_delta': max(abs(x[k]-y[k]) for k in common),
                                  'chosen_logprob_delta': abs(a['logprobs']['token_logprobs'][j]-b['logprobs']['token_logprobs'][j])})
            comparisons.append({'left': left, 'right': right, 'token_ids_equal': a['token_ids'] == b['token_ids'], 'positions': positions})
    report = {'responses': {label: value['id'] for label, value in values.items()}, 'comparisons': comparisons,
              'scope': 'Reuse exactly the last numerical-control salts and existing cache states, eight outputs/top20. No new model load or model/precision change. Original pool versus subsequent local reuse distinguishes transient load/resume state from persistent transferred-cache state; not complete-logit or payload equality proof.'}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
