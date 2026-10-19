#!/usr/bin/env python3
"""Current-instance matched native/fallback, retaining old-reference failures."""
import json
from pathlib import Path
import socket
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
    root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33')
    assert not (root / 'divergence-probe/skip-unused-single-token-draft').exists()
    cases = json.loads((root / 'local-complete-protected-reference-v1.json').read_text())['cases']
    out = root / 'local-complete-protected-matched-v1'
    out.mkdir(exist_ok=False)
    report = {'completed': False, 'rows': [], 'groups': [], 'scope': 'Same loaded native engines and same experimental CPU proxy, native/local-header native fallback ABBA. Independent cold salts. Distinguish native variation and selected-fallback difference; no timing.'}
    try:
        for case in cases:
            for maximum in [2, 64]:
                rows = []
                for mode in ['native', 'p-complete', 'p-complete', 'native']:
                    rid = 'local-fallback-' + uuid.uuid4().hex
                    body = {'model': 'glm5', 'prompt': case['prompt_ids'], 'max_tokens': maximum, 'min_tokens': 1,
                            'ignore_eos': True, 'temperature': 0, 'top_p': 1, 'seed': 1024,
                            'return_token_ids': True, 'cache_salt': rid, 'stream': True,
                            'stream_options': {'include_usage': True}}
                    req = urllib.request.Request('http://172.16.12.32:29882/v1/completions', json.dumps(body).encode(),
                                                 {'Content-Type': 'application/json', 'X-Request-Id': rid, 'X-GLM53-Route': mode})
                    row = {'case_id': case['case_id'], 'mode': mode, 'max_tokens': maximum,
                           'request': body, 'request_id': rid, 'chunks': [], 'token_ids': []}
                    report['rows'].append(row)
                    rows.append(row)
                    with urllib.request.urlopen(req, timeout=120) as response:
                        row['HTTP_status'] = response.status
                        row['route_header'] = response.headers.get('X-GLM53-Route')
                        for line in response:
                            if line.startswith(b'data: ') and not line.startswith(b'data: [DONE]'):
                                chunk = json.loads(line[6:])
                                assert not chunk.get('error')
                                row['chunks'].append(chunk)
                                for choice in chunk.get('choices', []):
                                    row['token_ids'].extend(choice.get('token_ids') or [])
                    row['text'] = ''.join(c.get('text', '') for chunk in row['chunks'] for c in chunk.get('choices', []))
                    row['usage'] = next(c['usage'] for c in reversed(row['chunks']) if c.get('usage'))
                    assert row['HTTP_status'] == 200 and row['route_header'] is None
                    assert len(row['token_ids']) == maximum and row['token_ids'][:2] == case['expected_ids'][:2]
                    assert row['usage']['completion_tokens'] == maximum and row['usage']['prompt_tokens'] == len(case['prompt_ids'])
                    row['old_reference_equal'] = row['token_ids'] == case['expected_ids'][:maximum]
                report['groups'].append({'case_id': case['case_id'], 'max_tokens': maximum,
                                         'native_repeat_equal': rows[0]['token_ids'] == rows[3]['token_ids'],
                                         'all_matched_IDs_text_usage_equal': all((r['token_ids'], r['text'], r['usage']) == (rows[0]['token_ids'], rows[0]['text'], rows[0]['usage']) for r in rows),
                                         'old_reference_equal_count': sum(r['old_reference_equal'] for r in rows)})
                (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
        report['completed'] = True
        report['all_matched_equal'] = all(g['all_matched_IDs_text_usage_equal'] for g in report['groups'])
    except Exception as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
