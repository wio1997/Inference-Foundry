#!/usr/bin/env python3
"""Make real model requests and verify API text, usage and eight-rank token evidence."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:18110'


def request(path, payload=None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + path, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as reply:
        return json.load(reply)


def main():
    output = ROOT / 'runtime-20261021/verification-latest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'status': 'VERIFYING', 'updated_at': datetime.now(timezone(timedelta(hours=8))).isoformat()}, indent=2) + '\n')
    assert request('/healthz').get('mock') is False, 'API must use the real backend'
    ready = request('/readyz')
    assert ready.get('ready') is True and ready.get('ranks') == 8
    results = []
    cases = [
        ('arithmetic', '只回答数字：1+1等于几？', 32),
        ('chinese_reasoning', '小明有3盒铅笔，每盒12支，送出8支后还剩多少支？请用中文写出计算过程和答案。', 256),
    ]
    for name, prompt, maximum in cases:
        started = time.time()
        response = request('/v1/chat/completions', {
            'model': 'DeepSeek-V4.1-Flash-1017',
            'messages': [{'role': 'user', 'content': prompt}],
            'thinking': {'type': 'disabled'}, 'temperature': 0, 'stream': False, 'max_tokens': maximum,
        })
        text = response['choices'][0]['message']['content']
        if name == 'arithmetic':
            assert text.strip() == '2', 'Unexpected arithmetic answer: ' + repr(text)
        else:
            assert '28' in text, 'Expected a calculation ending in 28 pencils: ' + repr(text)
        jobs = sorted((ROOT / 'runtime-20261021/generations').glob('job-*.json'), key=lambda p: p.stat().st_mtime)
        assert jobs and jobs[-1].stat().st_mtime >= started - 2, 'No fresh generation evidence'
        evidence = json.loads(jobs[-1].read_text())
        ranks = evidence['rank_results']
        assert len(ranks) == 8 and {x['rank'] for x in ranks} == set(range(8))
        token_hash = hashlib.sha256(json.dumps(evidence['token_ids']).encode()).hexdigest()
        assert {x['token_sha256'] for x in ranks} == {token_hash}, 'Rank output tokens differ'
        assert response['usage']['completion_tokens'] == len(evidence['token_ids']) > 0
        results.append({'name': name, 'text': text, 'usage': response['usage'],
                        'all_eight_rank_tokens_equal': True, 'evidence': jobs[-1].name})
        print('[PASS]', name, text.strip(), flush=True)
        print('[PASS] all eight rank token hashes and output usage match', flush=True)
    report = {'verified_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'status': 'DEPLOYMENT_VERIFIED', 'results': results,
              'scope': 'DP8/EP8/CP1 mirrored, non-streaming functional deployment; not a performance benchmark'}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('DEPLOYMENT_VERIFIED', str(output), flush=True)


if __name__ == '__main__':
    main()
