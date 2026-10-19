#!/usr/bin/env python3
"""Native-vs-local whole-answer checks, occupancy and unsupported fallback."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.request
import uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--protected-reference', type=Path, required=True)
    a = p.parse_args()
    assert a.execute and socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
    root = a.run_dir.resolve()
    assert str(root) == '/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33'
    cid = json.loads((root / 'launch.json').read_text())['container_id']
    assert cid == '6584fed1fd2712610c6840a6cc631914e6e4549958055a5b2175f22455c7426a'
    assert json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]['State']['Running']
    assert not (root / 'divergence-probe/skip-unused-single-token-draft').exists()
    cases = json.loads(a.cases.read_text())['cases']
    assert len(cases) == 10
    refs = json.loads(a.protected_reference.read_text())['cases']
    out = root / 'local-complete-functional-v1'
    out.mkdir(exist_ok=False)
    report = {'completed': False, 'rows': [], 'scope': 'Independent CPU29882 strict local-complete route vs its unchanged native fallback. Selected8 short whole numeric cases+2 long cases, stream/nonstream/C4, cache idle and D completion counters. No general accuracy or performance claim.'}

    def state(label):
        result = {}
        for node, port in [(32, 29481), (33, 29482)]:
            with urllib.request.urlopen('http://172.16.12.%s:%s/metrics' % (node, port), timeout=5) as response:
                raw = response.read().decode()
            (out / ('metrics-' + label + '-' + str(node) + '.txt')).write_text(raw)
            counts = {}
            for kind in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc', 'request_success_total']:
                values = re.findall(r'^vllm:' + kind + r'(?:\{[^\n]*\})?\s+(\S+)', raw, re.M)
                assert values, (node, kind)
                counts[kind] = sum(float(v) for v in values)
            result[str(node)] = counts
        with urllib.request.urlopen('http://172.16.12.32:29882/healthcheck', timeout=5) as response:
            result['proxy'] = json.load(response)
        return result

    def idle(label):
        deadline = time.monotonic() + 20
        while True:
            result = state(label)
            if all(result[str(n)][k] == 0 for n in [32, 33] for k in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc']) and result['proxy']['request_num'] == 0:
                return result
            assert time.monotonic() < deadline, result
            time.sleep(.2)

    def one(case, mode, stream, cohort, barrier=None, maximum=1):
        rid = 'local-complete-' + uuid.uuid4().hex
        body = {'model': 'glm5', 'prompt': case['prompt_ids'], 'max_tokens': maximum, 'min_tokens': 1,
                'ignore_eos': True, 'temperature': 0, 'top_p': 1, 'seed': 1024,
                'return_token_ids': True, 'cache_salt': rid, 'stream': stream}
        if stream:
            body['stream_options'] = {'include_usage': True}
        req = urllib.request.Request('http://172.16.12.32:29882/v1/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json', 'X-Request-Id': rid, 'X-GLM53-Route': mode})
        row = {'case_id': case['case_id'], 'N': len(case['prompt_ids']), 'mode': mode, 'cohort': cohort,
               'stream': stream, 'max_tokens': maximum, 'request': body, 'request_id': rid,
               'token_ids': [], 'chunks': []}
        report['rows'].append(row)
        if barrier:
            barrier.wait()
        with urllib.request.urlopen(req, timeout=120) as response:
            row['HTTP_status'] = response.status
            row['route_header'] = response.headers.get('X-GLM53-Route')
            if stream:
                for line in response:
                    if line.startswith(b'data: ') and not line.startswith(b'data: [DONE]'):
                        row['chunks'].append(json.loads(line[6:]))
            else:
                row['chunks'].append(json.load(response))
        assert all(not chunk.get('error') for chunk in row['chunks'])
        for chunk in row['chunks']:
            for choice in chunk.get('choices', []):
                row['token_ids'].extend(choice.get('token_ids') or [])
        row['text'] = ''.join(c.get('text', '') for chunk in row['chunks'] for c in chunk.get('choices', []))
        row['usage'] = next(chunk['usage'] for chunk in reversed(row['chunks']) if chunk.get('usage'))
        assert row['HTTP_status'] == 200 and len(row['token_ids']) == maximum
        assert row['usage']['completion_tokens'] == maximum and row['usage']['prompt_tokens'] == row['N']
        if maximum == 1:
            assert row['token_ids'] == [case['expected_token_id']] and row['text'].strip() == case['expected']
            assert row['route_header'] == ('p-local-complete' if mode == 'p-complete' else None)
            if mode == 'p-complete' and not stream:
                assert not row['chunks'][0].get('kv_transfer_params'), 'Remote KV metadata from local completion'
        else:
            assert row['route_header'] is None
            assert row['token_ids'] == case['expected_ids'][:maximum]
        row['complete_pass'] = True

    try:
        report['before'] = idle('before')
        for mode in ['native', 'p-complete']:
            if mode == 'p-complete':
                report['before_local'] = idle('before-local')
            for case in cases:
                for stream in [False, True]:
                    one(case, mode, stream, 'C1')
                (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
            for long in [False, True]:
                picked = [c for c in cases if (len(c['prompt_ids']) > 1000) == long][:2]
                barrier = threading.Barrier(4)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda i: one(picked[i % 2], mode, True, 'C4-long' if long else 'C4-short', barrier), range(4)))
            if mode == 'p-complete':
                report['after_local'] = idle('after-local')
                report['D_local_success_delta'] = report['after_local']['33']['request_success_total'] - report['before_local']['33']['request_success_total']
                assert report['D_local_success_delta'] == 0
        for case in refs:
            for maximum in [2, 64]:
                one(case, 'p-complete', True, 'protected-native-fallback', maximum=maximum)
        report['after'] = idle('after')
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
