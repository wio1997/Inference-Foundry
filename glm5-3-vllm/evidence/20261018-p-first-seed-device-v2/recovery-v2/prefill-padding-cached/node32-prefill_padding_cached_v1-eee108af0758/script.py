#!/usr/bin/env python3
"""Finite ordinary multi-token PD checks of native P padding versus native eager."""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import statistics
import subprocess
import time
import urllib.request
import uuid

P_ROOT = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefill-padding-live1-20261018/node32')
P_CID = '28e687e363317e918ed636a4615f918dda3ea80839626e357e82d77ea87faf56'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--phase', choices=['functional', 'quality', 'cached', 'c1', 'c4'], required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--run-tag', default='v2')
    args = parser.parse_args()
    assert args.execute and socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    plan = json.loads((P_ROOT / 'plan.json').read_text())
    actual = json.loads(subprocess.check_output(['docker', 'inspect', P_CID]))[0]
    assert actual['State']['Running'] and actual['Image'] == plan['image']['id']
    assert actual['Name'] == '/' + plan['container']['name']
    assert plan['env']['GLM53_PREFILL_PADDING_PROBE'] == '1'
    fixtures = json.loads(args.cases.read_text())
    assert re.fullmatch(r'[a-z0-9-]+', args.run_tag)
    out = P_ROOT / ('prefill-padding-' + args.phase + '-' + args.run_tag)
    out.mkdir(exist_ok=False)
    marker = P_ROOT / 'prefill-padding-probe/enable'
    trace_disabled = P_ROOT / 'prefill-padding-probe/disabled'
    if args.phase in ['c1', 'c4']:
        assert trace_disabled.exists() and (P_ROOT / 'adopt-trace/disabled').exists()
    else:
        assert not trace_disabled.exists()
    report = {'completed': False, 'rows': [], 'blocks': [], 'phase': args.phase,
              'P_CID': P_CID, 'fixture_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              'route': 'ordinary native PD; no seed/adoption or local completion',
              'scope': 'Fixed finite workloads and numeric tasks; full sequences retained, no general quality/capacity acceptance.'}
    cache_keys = {case['case_id']: 'padding-cache-' + uuid.uuid4().hex for case in fixtures['mixed']}

    def get(url):
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read().decode()

    def idle():
        for attempt in range(61):
            selected = {}
            for node, port in [(32, 29481), (33, 29482)]:
                data = get('http://172.16.12.%d:%d/metrics' % (node, port))
                rows = [line for line in data.splitlines() if re.match(r'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)\{', line)]
                assert len(rows) >= 3
                selected[str(node)] = rows
            if all(float(line.rsplit(' ', 1)[1]) == 0 for rows in selected.values() for line in rows):
                report.setdefault('idle_checks', []).append({'polls': attempt + 1, 'metrics': selected})
                return
            time.sleep(0.25)
        raise AssertionError(selected)

    def select(mode):
        idle()
        if mode == 'adaptive':
            marker.touch()
        else:
            marker.unlink(missing_ok=True)

    def post(node, port, body, rid):
        request = urllib.request.Request('http://172.16.12.%d:%d/v1/completions' % (node, port),
                                        json.dumps(body).encode(), {'Content-Type': 'application/json', 'X-Request-Id': rid})
        return urllib.request.urlopen(request, timeout=180)

    def one(case, mode, phase, block=None):
        ids = case['prompt_token_ids']
        assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == case['prompt_sha256']
        budget = case.get('output_tokens', 64)
        rid = 'padding-' + uuid.uuid4().hex
        body = {'model': 'glm5', 'prompt': ids, 'max_tokens': budget, 'min_tokens': 1,
                'ignore_eos': True, 'temperature': 0, 'top_p': 1, 'seed': 1024,
                'return_token_ids': True, 'cache_salt': rid, 'stream': True,
                'stream_options': {'include_usage': True}}
        if args.phase == 'cached':
            body['cache_salt'] = cache_keys[case['case_id']]
        row = {'case_id': case['case_id'], 'N': len(ids), 'mode': mode, 'phase': phase,
               'block': block, 'header_request_id': rid, 'request': body,
               'token_ids': [], 'arrival_ns': [], 'chunks': [], 'expected': case['expected']}
        report['rows'].append(row)
        row['sent_ns'] = time.monotonic_ns()
        if args.phase in ['functional', 'quality', 'cached']:
            producer = {k: v for k, v in body.items() if k != 'stream_options'}
            producer.update(max_tokens=1, stream=False, kv_transfer_params={
                'do_remote_decode': True, 'do_remote_prefill': False,
                'remote_engine_id': None, 'remote_block_ids': None, 'remote_host': None, 'remote_port': None})
            row['P_request'] = producer
            with post(32, 29481, producer, rid) as response:
                row['P_response'] = json.load(response)
            row['P_received_ns'] = time.monotonic_ns()
            params = row['P_response']['kv_transfer_params']
            assert params['remote_num_prompt_tokens'] == len(ids)
            body['kv_transfer_params'] = params
            # Always consume native D after P, including when later quality checks fail.
            response = post(33, 29482, body, rid)
        else:
            response = post(32, 29880, body, rid)
        with response:
            row['HTTP_status'] = response.status
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
        row['text'] = ''.join(choice.get('text', '') for chunk in row['chunks'] for choice in chunk.get('choices', []))
        row['usage'] = next(chunk['usage'] for chunk in reversed(row['chunks']) if chunk.get('usage'))
        match = re.search(r'[-+]?\d+(?:\.\d+)?', row['text'])
        row['numeric_pass'] = bool(match and float(match.group()) == case['expected'])
        row['historical64_exact'] = row['token_ids'][:64] == case.get('expected_token_ids') if 'expected_token_ids' in case else None
        assert row['HTTP_status'] == 200 and len(row['token_ids']) == budget
        assert row['usage']['completion_tokens'] == budget and row['usage']['prompt_tokens'] == len(ids)
        row['TTFT_ms'] = (row['arrival_ns'][0] - row['sent_ns']) / 1e6
        row['last_token_ms'] = (row['arrival_ns'][-1] - row['sent_ns']) / 1e6
        row['mean_TPOT_ms'] = (row['arrival_ns'][-1] - row['arrival_ns'][0]) / 1e6 / (budget - 1)
        if 'P_received_ns' in row:
            row['P_ms'] = (row['P_received_ns'] - row['sent_ns']) / 1e6
        row['functional_pass'] = True
        return row

    def save():
        (out / 'progress.json').write_text(json.dumps(report) + '\n')

    try:
        idle()
        if args.phase in ['functional', 'quality', 'cached', 'c1']:
            cases = fixtures['quality'] if args.phase == 'quality' else fixtures['mixed' if args.phase in ['functional', 'cached'] else 'fixed']
            for case in cases:
                order = ['native', 'adaptive'] if args.phase == 'quality' else ['native', 'adaptive', 'adaptive', 'native']
                if args.phase == 'c1':
                    for mode in ['native', 'adaptive']:
                        select(mode); one(case, mode, 'warmup'); save()
                    order *= 3
                for mode in order:
                    select(mode); one(case, mode, 'formal'); save()
        else:
            cases = fixtures['mixed']
            for mode in ['native', 'adaptive']:
                select(mode)
                for case in cases:
                    one(case, mode, 'warmup')
                save()
            for index, mode in enumerate(['native', 'adaptive', 'adaptive', 'native'] * 2):
                select(mode)
                block = {'index': index, 'mode': mode, 'started_ns': time.monotonic_ns()}
                report['blocks'].append(block)
                def slot(case):
                    return [one(case, mode, 'formal', index) for _ in range(8)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    for result in pool.map(slot, cases):
                        assert len(result) == 8
                block['finished_ns'] = time.monotonic_ns()
                block['requests'] = 32
                block['seconds'] = (block['finished_ns'] - block['started_ns']) / 1e9
                block['output_tokens'] = sum(row['usage']['completion_tokens'] for row in report['rows'] if row['block'] == index)
                block['output_tokens_per_s'] = block['output_tokens'] / block['seconds']
                save()
        idle()
        report['completed'] = True
        report['numeric_all_pass'] = all(row['numeric_pass'] for row in report['rows'])
        report['summary'] = {}
        for mode in ['native', 'adaptive']:
            rows = [row for row in report['rows'] if row['phase'] == 'formal' and row['mode'] == mode]
            report['summary'][mode] = {'requests': len(rows), 'numeric_pass': sum(row['numeric_pass'] for row in rows),
                                       'historical64_exact': sum(row['historical64_exact'] is True for row in rows),
                                       'cases': {}}
            for cid in sorted(set(row['case_id'] for row in rows)):
                chosen = [row for row in rows if row['case_id'] == cid]
                report['summary'][mode]['cases'][cid] = {metric: statistics.median(row[metric] for row in chosen) for metric in ['TTFT_ms', 'last_token_ms', 'mean_TPOT_ms'] + (['P_ms'] if 'P_ms' in chosen[0] else [])}
            blocks = [block for block in report['blocks'] if block['mode'] == mode]
            if blocks:
                report['summary'][mode]['output_tokens_per_s'] = sum(block['output_tokens'] for block in blocks) / sum(block['seconds'] for block in blocks)
        by_case = {}
        for row in report['rows']:
            by_case.setdefault(row['case_id'], []).append(row)
        report['sequence_comparisons'] = {cid: {'native_unique': len({tuple(row['token_ids']) for row in rows if row['mode'] == 'native'}),
                                                'adaptive_unique': len({tuple(row['token_ids']) for row in rows if row['mode'] == 'adaptive'}),
                                                'all_same': len({tuple(row['token_ids']) for row in rows}) == 1}
                                          for cid, rows in by_case.items()}
    except Exception as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        marker.unlink(missing_ok=True)
    (out / 'result.json').write_text(json.dumps(report) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
