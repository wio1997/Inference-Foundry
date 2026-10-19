#!/usr/bin/env python3
"""Two fixed arithmetic prompts across a one-token inert-note length change."""
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import time
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
    root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33')
    assert not (root / 'divergence-probe/skip-unused-single-token-draft').exists()
    originals = json.loads((root / 'unused-draft-cases-v1.json').read_text())['cases']
    out = root / 'prefill-boundary-crossover-v1'
    out.mkdir(exist_ok=False)
    report = {'completed': False, 'cases': [], 'rows': [], 'scope': 'P native local completion, same loaded original P/K1/precision/operators. Two existing arithmetic cases each at256/257 tokens by one inert-note edit, math suffix unchanged. Content/positions also change; consistent boundary evidence alone is not an operator/root-cause proof or an optimization result.'}

    def detokenize(ids):
        req = urllib.request.Request('http://172.16.12.32:29481/detokenize', json.dumps({'model': 'glm5', 'tokens': ids}).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)['prompt']

    def one(case, phase):
        rid = 'prefill-boundary-' + uuid.uuid4().hex
        body = {'model': 'glm5', 'prompt': case['prompt_ids'], 'max_tokens': 1, 'min_tokens': 1,
                'ignore_eos': True, 'temperature': 0, 'top_p': 1, 'seed': 1024,
                'return_token_ids': True, 'cache_salt': rid, 'stream': True,
                'stream_options': {'include_usage': True}}
        req = urllib.request.Request('http://172.16.12.32:29882/v1/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json', 'X-GLM53-Route': 'p-complete', 'X-Request-Id': rid})
        row = {'case_id': case['case_id'], 'N': case['N'], 'phase': phase, 'request': body,
               'request_id': rid, 'chunks': [], 'token_ids': [], 'arrival_ns': []}
        report['rows'].append(row)
        row['sent_ns'] = time.monotonic_ns()
        with urllib.request.urlopen(req, timeout=120) as response:
            assert response.status == 200 and response.headers.get('X-GLM53-Route') == 'p-local-complete'
            for line in response:
                if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                    continue
                chunk = json.loads(line[6:])
                assert not chunk.get('error')
                stamp = time.monotonic_ns()
                row['chunks'].append(chunk)
                for choice in chunk.get('choices', []):
                    tokens = choice.get('token_ids') or []
                    row['token_ids'].extend(tokens)
                    row['arrival_ns'].extend([stamp] * len(tokens))
        row['finished_ns'] = time.monotonic_ns()
        row['text'] = ''.join(c.get('text', '') for chunk in row['chunks'] for c in chunk.get('choices', []))
        row['usage'] = next(c['usage'] for c in reversed(row['chunks']) if c.get('usage'))
        assert row['token_ids'] == [case['expected_token_id']] and row['text'].strip() == case['answer']
        assert row['usage']['prompt_tokens'] == case['N'] and row['usage']['completion_tokens'] == 1
        row['TTFT_ms'] = (row['arrival_ns'][0] - row['sent_ns']) / 1e6
        row['HTTP_finish_ms'] = (row['finished_ns'] - row['sent_ns']) / 1e6
        row['complete_pass'] = True

    try:
        for original in originals:
            ids = original['prompt_ids'][:-2]
            assert len(ids) in [256, 257]
            text = detokenize(ids)
            if len(ids) == 256:
                variant = ids[:23] + [198] + ids[23:]
                edit = {'kind': 'one extra inert blank line', 'index': 23, 'token': 198}
            else:
                index = next(i for i in range(23, 60) if ids[i] == 25)
                variant = ids[:index] + ids[index + 1:]
                edit = {'kind': 'remove one filler-note colon', 'index': index, 'token': 25}
            for current, edit_info in [(ids, None), (variant, edit)]:
                decoded = text if edit_info is None else detokenize(current)
                assert current[-32:] == ids[-32:] and decoded.splitlines()[-1] == text.splitlines()[-1]
                report['cases'].append({'case_id': original['case_id'], 'N': len(current), 'prompt_ids': current,
                                        'prompt_sha256': hashlib.sha256(json.dumps(current, separators=(',', ':')).encode()).hexdigest(),
                                        'decoded': decoded, 'edit': edit_info, 'math_suffix_ids_unchanged': True,
                                        'expected_token_id': original['prompt_ids'][-2],
                                        'answer': {'sum-three-4': '84', 'sum-three-1': '79'}[original['case_id']]})
        (out / 'cases.json').write_text(json.dumps({'cases': report['cases']}, indent=2) + '\n')
        for case in report['cases']:
            one(case, 'warmup')
        for original in originals:
            cases = {c['N']: c for c in report['cases'] if c['case_id'] == original['case_id']}
            for n in [256, 257, 257, 256] * 3:
                one(cases[n], 'formal')
            (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
        summary = []
        for original in originals:
            item = {'case_id': original['case_id'], 'lengths': {}}
            for n in [256, 257]:
                rows = [r for r in report['rows'] if r['case_id'] == original['case_id'] and r['N'] == n and r['phase'] == 'formal']
                assert len(rows) == 6
                item['lengths'][str(n)] = {'median_TTFT_ms': statistics.median(r['TTFT_ms'] for r in rows), 'samples_ms': [r['TTFT_ms'] for r in rows]}
            item['257_over_256_TTFT'] = item['lengths']['257']['median_TTFT_ms'] / item['lengths']['256']['median_TTFT_ms']
            summary.append(item)
        report['summary'] = {'cases': summary, 'ratio_equal_case_geomean': math.exp(statistics.mean(math.log(c['257_over_256_TTFT']) for c in summary)), 'formal_requests': 24, 'warmup_requests': 4}
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
