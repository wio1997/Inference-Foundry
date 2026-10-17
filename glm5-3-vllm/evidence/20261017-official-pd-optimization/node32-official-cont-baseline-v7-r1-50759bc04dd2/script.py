#!/usr/bin/env python3
"""Measure four continuously active clients under frozen experimental latency bounds."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
import hashlib
import math
from pathlib import Path
import re
import socket
import threading
import time
import urllib.request
import uuid

BASE = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
URL = {'P': 'http://172.16.12.32:29481', 'D': 'http://172.16.12.33:29482',
       'proxy': 'http://172.16.12.32:29880'}


def http(role, path, payload=None, timeout=180):
    request = urllib.request.Request(URL[role] + path,
                                    data=None if payload is None else json.dumps(payload).encode(),
                                    headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=timeout)


def distribution(values):
    values = sorted(values)
    def percentile(q):
        x = (len(values) - 1) * q
        i, j = math.floor(x), math.ceil(x)
        return values[i] + (values[j] - values[i]) * (x - i)
    return {'n': len(values), 'p50': percentile(.5), 'p95': percentile(.95),
            'min': values[0], 'max': values[-1]} if values else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=('baseline', 'optimized'), required=True)
    parser.add_argument('--revision', type=int, required=True)
    parser.add_argument('--prefill-revision', type=int, required=True)
    parser.add_argument('--bounds-b64', required=True)
    parser.add_argument('--after-run')
    parser.add_argument('--duration', type=int, default=180)
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    assert re.fullmatch(r'[a-z0-9-]+', args.run_id) and args.duration in (120, 180, 300)
    bounds = json.loads(base64.b64decode(args.bounds_b64))
    assert set(bounds) == {'256', '1024'}
    root = BASE / 'tests' / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'passed': False, 'state': 'waiting_ready', 'run_id': args.run_id,
              'stage': args.stage, 'revision': args.revision, 'prefill_revision': args.prefill_revision,
              'bounds': bounds, 'cases': [], 'scope': 'Four continuous closed-loop clients, cold unique '
              'prefixes, fixed64 output tokens, frozen common experimental TTFT/request-average TPOT '
              'bounds. Initial and per-shape warmups excluded. No production SLO/global capacity claim.'}

    def save():
        (root / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')

    def invoke(prompt):
        payload = {'model': 'glm5', 'prompt': prompt, 'max_tokens': 64, 'temperature': 0,
                   'top_p': 1, 'seed': 1024, 'ignore_eos': True, 'return_token_ids': True,
                   'cache_salt': 'continuous-' + uuid.uuid4().hex, 'stream': True,
                   'stream_options': {'include_usage': True}}
        row = {'start_ns': time.perf_counter_ns(), 'token_ids': [], 'arrivals_ns': [], 'text': ''}
        row['cache_salt'] = payload['cache_salt']
        try:
            with http('proxy', '/v1/completions', payload) as response:
                row['http_status'] = response.status
                for line in response:
                    if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                        continue
                    now = time.perf_counter_ns()
                    value = json.loads(line[6:])
                    row['completion_id'] = value.get('id')
                    if value.get('error'):
                        raise RuntimeError(str(value['error']))
                    for choice in value.get('choices', []):
                        ids = choice.get('token_ids') or []
                        row['token_ids'].extend(ids)
                        row['arrivals_ns'].extend([now] * len(ids))
                        row['text'] += choice.get('text', '')
                    if value.get('usage'):
                        row['usage'] = value['usage']
            number = re.match(r'\s*([-+]?\d+(?:\.\d+)?)', row['text'])
            row['success'] = (row['http_status'] == 200 and len(row['token_ids']) == 64
                              and bool(number and Decimal(number.group(1)) == 5))
        except Exception as exc:
            row.update(success=False, error=type(exc).__name__ + ': ' + str(exc))
        row['closed_ns'] = time.perf_counter_ns()
        if row['arrivals_ns']:
            arrivals = row['arrivals_ns']
            row['ttft_ms'] = (arrivals[0] - row['start_ns']) / 1e6
            row['tpot_ms'] = (arrivals[-1] - arrivals[0]) / 1e6 / max(1, len(arrivals) - 1)
        return row

    try:
        save()
        deadline = time.monotonic() + 1200
        if args.after_run:
            assert re.fullmatch(r'[a-z0-9-]+', args.after_run)
            report['state'] = 'waiting_predecessor'
            save()
            previous = BASE / 'tests' / args.after_run / 'result.json'
            while not previous.exists():
                assert time.monotonic() < deadline, 'Predecessor timeout'
                time.sleep(5)
            prior = json.loads(previous.read_text())
            assert prior['completed'] and prior['passed']
            deadline = time.monotonic() + 1200
            report['state'] = 'waiting_ready'
            save()
        while True:
            ready = []
            for role in ('P', 'D'):
                try:
                    with http(role, '/v1/models', timeout=2) as response:
                        ready.append(any(x['id'] == 'glm5' for x in json.load(response)['data']))
                except Exception:
                    ready.append(False)
            if all(ready):
                break
            assert time.monotonic() < deadline, 'Pair readiness timeout'
            time.sleep(5)
        prefill = BASE / ('prefill-node32-v' + str(args.prefill_revision))
        assert (prefill / 'launch.json').is_file()
        assert not any(p.is_file() for p in (prefill / 'profile').rglob('*'))
        source = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests/native-eager-c1-v1')
        fixtures = {n: json.loads((source / ('cold-eager-reference-' + str(n) + '.json')).read_text())['request']['prompt']
                    for n in (256, 1024)}
        assert all(len(prompt) == n for n, prompt in fixtures.items())
        report['fixtures_sha256'] = hashlib.sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest()
        report['state'] = 'initial_connection_compile_warmup'
        save()
        report['initial_warmups'] = [invoke(fixtures[n]) for n in (256, 1024)]
        assert all(x['success'] for x in report['initial_warmups'])

        def metrics(label):
            result = {}
            for role in ('P', 'D'):
                with http(role, '/metrics', timeout=10) as response:
                    raw = response.read().decode()
                (root / (label + '-' + role + '.txt')).write_text(raw)
                value = {}
                for line in raw.splitlines():
                    if not line or line.startswith('#'):
                        continue
                    name, number = line.rsplit(' ', 1)
                    name = name.split('{', 1)[0]
                    value[name] = value.get(name, 0) + float(number)
                result[role] = value
            return result

        for n in (256, 1024):
            with ThreadPoolExecutor(max_workers=4) as pool:
                warm = list(pool.map(invoke, [fixtures[n]] * 4))
                (root / (str(n) + '-warmup.json')).write_text(json.dumps(warm, indent=2) + '\n')
                assert all(x['success'] for x in warm)
                before = metrics(str(n) + '-before')
                rows = []
                lock = threading.Lock()
                stop = threading.Event()
                barrier = threading.Barrier(5)
                start = [None]
                report['state'] = 'continuous-' + str(n)
                report['current_completed_requests'] = 0
                save()

                def client(client_id):
                    barrier.wait(timeout=30)
                    while not stop.is_set() and time.perf_counter_ns() < start[0] + args.duration * 10**9:
                        row = invoke(fixtures[n])
                        row['client_id'] = client_id
                        with lock:
                            rows.append(row)
                            if not row['success']:
                                stop.set()
                            if len(rows) % 20 == 0:
                                report['current_completed_requests'] = len(rows)
                                save()

                futures = [pool.submit(client, i) for i in range(4)]
                start[0] = time.perf_counter_ns()
                barrier.wait(timeout=30)
                for future in futures:
                    future.result()
                elapsed = (max(x['closed_ns'] for x in rows) - start[0]) / 1e9
                after = metrics(str(n) + '-after')
            (root / (str(n) + '-requests.json')).write_text(json.dumps(rows, indent=2) + '\n')
            good = [x for x in rows if x['success']]
            timely = [x for x in good if x['ttft_ms'] <= bounds[str(n)]['ttft_ms']
                      and x['tpot_ms'] <= bounds[str(n)]['request_average_tpot_ms']]
            gaps = [(b - a) / 1e6 for x in good for a, b in zip(x['arrivals_ns'], x['arrivals_ns'][1:])]
            case = {'input_tokens': n, 'output_tokens': 64, 'concurrency': 4, 'requests': len(rows),
                    'valid_requests': len(good), 'timely_valid_requests': len(timely), 'elapsed_s': elapsed,
                    'valid_output_tok_s': len(good) * 64 / elapsed,
                    'timely_valid_output_tok_s': len(timely) * 64 / elapsed,
                    'ttft_ms': distribution([x['ttft_ms'] for x in good]),
                    'tpot_ms': distribution([x['tpot_ms'] for x in good]),
                    'inter_token_gap_ms': distribution(gaps),
                    'zero_gap_fraction': sum(x == 0 for x in gaps) / len(gaps) if gaps else None,
                    'prefix_hit_delta': {role: after[role].get('vllm:prefix_cache_hits_total', 0)
                                         - before[role].get('vllm:prefix_cache_hits_total', 0) for role in ('P', 'D')},
                    'rolling_30s_valid_tok_s': [sum(64 for x in good if i * 30 <= (x['closed_ns'] - start[0]) / 1e9 < (i + 1) * 30) / 30
                                              for i in range(args.duration // 30)]}
            report['cases'].append(case)
            save()
            assert len(good) == len(rows) and len(rows) >= 50, 'Formal HTTP/length/numerical failure'
            assert all(x == 0 for x in case['prefix_hit_delta'].values()), 'Cold-prefix hit detected'
        report.update(completed=True, passed=True, state='continuous_collected_compare_common_bounds')
    except Exception as exc:
        report.update(state='needs_specific_followup', failure=type(exc).__name__ + ': ' + str(exc))
    (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    print(json.dumps(report, indent=2))
    assert report['completed'] and report['passed'], report.get('failure')


if __name__ == '__main__':
    main()
