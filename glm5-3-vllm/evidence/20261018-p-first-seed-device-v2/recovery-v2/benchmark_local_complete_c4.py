#!/usr/bin/env python3
"""Fixed C4 closed-loop point on the original official PD proxy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid


def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--functional-only', action='store_true')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args()
    assert args.execute and socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
    root = args.run_dir.resolve()
    assert str(root) == '/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33'
    plan = json.loads((root / 'plan.json').read_text())
    cid = json.loads((root / 'launch.json').read_text())['container_id']
    actual = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    assert cid == '6584fed1fd2712610c6840a6cc631914e6e4549958055a5b2175f22455c7426a'
    assert actual['State']['Running'] and actual['Image'] == plan['image']['id']
    assert actual['Name'] == '/' + plan['container']['name']
    probe = root / 'divergence-probe'
    marker = probe / 'skip-unused-single-token-draft'
    tensor_disabled = probe / 'disabled'
    policy_disabled = probe / 'overlap-probe-disabled'
    unused_disabled = probe / 'unused-draft-trace-disabled'
    host_disabled = root / 'adopt-trace/disabled'
    assert all(p.exists() for p in [tensor_disabled, policy_disabled, unused_disabled, host_disabled])
    assert not (probe / 'shared-overlap-disabled').exists()
    cases = json.loads(args.cases.read_text())['cases']
    assert [c['case_id'] for c in cases] == ['sum-three-4', 'sum-three-1']
    answers = {'sum-three-4': '84', 'sum-three-1': '79'}
    out = root / ('single-token-C4-functional-v1' if args.functional_only else 'local-complete-performance-c4-v1')
    out.mkdir(exist_ok=False)
    report = {'completed': False, 'blocks': [], 'formal_order': ['pd-skip', 'p-complete', 'p-complete', 'pd-skip'] * 3,
              'scope': 'Fixed C4 closed-loop point, 4 slots x16 complete requests/block, equal N256/257, cold independent salts, same patched official CPU29882; native PD with validated D max1 skip vs P native local completion, both full P/D resources retained. No rate/parameter sweep, no general64-output or maximum-SLO-capacity claim. TTFT=last for complete single-token answers, TPOT undefined.'}

    def idle():
        deadline = time.monotonic() + 15
        while True:
            counts = {}
            for node, port in [(32, 29481), (33, 29482)]:
                with urllib.request.urlopen('http://172.16.12.%s:%s/metrics' % (node, port), timeout=5) as response:
                    text = response.read().decode()
                for key in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc']:
                    values = re.findall(r'^vllm:' + key + r'(?:\{[^\n]*\})?\s+(\S+)', text, re.M)
                    assert values, (node, key)
                    counts[str(node) + ':' + key] = sum(float(v) for v in values)
            with urllib.request.urlopen('http://172.16.12.32:29882/healthcheck', timeout=5) as response:
                counts['proxy:requests'] = json.load(response)['request_num']
            if all(v == 0 for v in counts.values()):
                return counts
            assert time.monotonic() < deadline, counts
            time.sleep(.1)

    def one(case, mode, slot, index):
        ids = case['prompt_ids'][:-2]
        rid = 'single-C4-' + uuid.uuid4().hex
        body = {'model': 'glm5', 'prompt': ids, 'max_tokens': 1, 'min_tokens': 1,
                'ignore_eos': True, 'temperature': 0, 'top_p': 1, 'seed': 1024,
                'return_token_ids': True, 'cache_salt': rid, 'stream': True,
                'stream_options': {'include_usage': True}}
        req = urllib.request.Request('http://172.16.12.32:29882/v1/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json', 'X-Request-Id': rid, 'X-GLM53-Route': 'p-complete' if mode == 'p-complete' else 'native'})
        row = {'case_id': case['case_id'], 'N': len(ids), 'mode': mode, 'slot': slot, 'index': index,
               'request_id': rid, 'cache_salt': rid, 'token_ids': [], 'chunks': [], 'arrival_ns': []}
        row['sent_ns'] = time.monotonic_ns()
        with urllib.request.urlopen(req, timeout=120) as response:
            row['HTTP_status'] = response.status
            row['route_header'] = response.headers.get('X-GLM53-Route')
            assert row['route_header'] == ('p-local-complete' if mode == 'p-complete' else None)
            for line in response:
                if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                    continue
                chunk = json.loads(line[6:])
                assert not chunk.get('error'), chunk
                stamp = time.monotonic_ns()
                row['chunks'].append(chunk)
                for choice in chunk.get('choices', []):
                    tokens = choice.get('token_ids') or []
                    row['token_ids'].extend(tokens)
                    row['arrival_ns'].extend([stamp] * len(tokens))
        row['finished_ns'] = time.monotonic_ns()
        row['usage'] = next(c['usage'] for c in reversed(row['chunks']) if c.get('usage'))
        row['text'] = ''.join(c.get('text', '') for chunk in row['chunks'] for c in chunk.get('choices', []))
        assert row['HTTP_status'] == 200 and row['token_ids'] == [case['prompt_ids'][-2]]
        assert row['text'].strip() == answers[case['case_id']]
        assert row['usage']['prompt_tokens'] == len(ids) and row['usage']['completion_tokens'] == 1
        row['TTFT_ms'] = (row['arrival_ns'][0] - row['sent_ns']) / 1e6
        row['HTTP_finish_ms'] = (row['finished_ns'] - row['sent_ns']) / 1e6
        row['complete_answer_pass'] = True
        return row

    def block(mode, phase, iterations):
        before = idle()
        marker.touch()  # Best prior PD max1 skip enabled for both routes.
        item = {'mode': mode, 'phase': phase, 'requests_per_slot': iterations, 'idle_before': before, 'rows': []}
        report['blocks'].append(item)
        barrier = threading.Barrier(4, action=lambda: item.update(release_ns=time.monotonic_ns()))

        def worker(slot):
            rows = []
            barrier.wait()
            for i in range(iterations):
                rows.append(one(cases[(slot + i) % 2], mode, slot, i))
            return rows

        with ThreadPoolExecutor(max_workers=4) as pool:
            for rows in pool.map(worker, range(4)):
                item['rows'].extend(rows)
        item['last_finish_ns'] = max(r['finished_ns'] for r in item['rows'])
        item['duration_s'] = (item['last_finish_ns'] - item['release_ns']) / 1e9
        item['complete_answers_per_s'] = len(item['rows']) / item['duration_s']
        item['TTFT_ms'] = {label: percentile([r['TTFT_ms'] for r in item['rows']], q) for label, q in [('median', .5), ('p95', .95)]}
        item['idle_after'] = idle()
        (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')

    try:
        if args.functional_only:
            unused_disabled.unlink()
            host_disabled.unlink()
            for mode in ['pd-skip', 'p-complete']:
                block(mode, 'functional', 1)
        else:
            for mode in ['pd-skip', 'p-complete']:
                block(mode, 'warmup', 2)
            for mode in report['formal_order']:
                block(mode, 'formal', 16)
            modes = {}
            for mode in ['pd-skip', 'p-complete']:
                blocks = [b for b in report['blocks'] if b['phase'] == 'formal' and b['mode'] == mode]
                rows = [r for b in blocks for r in b['rows']]
                assert len(blocks) == 6 and len(rows) == 384 and all(r['complete_answer_pass'] for r in rows)
                modes[mode] = {'blocks': 6, 'complete_answers': len(rows),
                               'aggregate_complete_answers_per_s': len(rows) / sum(b['duration_s'] for b in blocks),
                               'median_block_complete_answers_per_s': statistics.median(b['complete_answers_per_s'] for b in blocks),
                               'block_complete_answers_per_s': [b['complete_answers_per_s'] for b in blocks],
                               'TTFT_ms': {label: percentile([r['TTFT_ms'] for r in rows], q) for label, q in [('median', .5), ('p95', .95)]}}
            report['summary'] = {'modes': modes, 'local_over_PD_skip': {
                'aggregate_complete_answers_per_s': modes['p-complete']['aggregate_complete_answers_per_s'] / modes['pd-skip']['aggregate_complete_answers_per_s'],
                'median_TTFT': modes['p-complete']['TTFT_ms']['median'] / modes['pd-skip']['TTFT_ms']['median'],
                'p95_TTFT': modes['p-complete']['TTFT_ms']['p95'] / modes['pd-skip']['TTFT_ms']['p95']},
                'formal_requests': 768, 'warmup_requests': 16, 'TPOT': None}
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        marker.unlink(missing_ok=True)
        for path in [unused_disabled, host_disabled]:
            path.touch()
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
