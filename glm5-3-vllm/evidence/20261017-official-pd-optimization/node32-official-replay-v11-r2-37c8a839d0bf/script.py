#!/usr/bin/env python3
"""Replay every original baseline arrival to compare latency at the same offered load."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from decimal import Decimal
import errno
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import threading
import time
import urllib.request
import uuid

BASE = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
URL = 'http://172.16.12.32:29880/v1/completions'


class Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]


def absolute_wait(target_ns):
    # Python3.9 sleep uses select on this host, whose timeout slack scales
    # with the long look-ahead interval. An absolute monotonic deadline avoids
    # that drift while leaving the worker asleep and the GIL released.
    deadline = Timespec(*divmod(target_ns, 10**9))
    while True:
        rc = LIBC.clock_nanosleep(time.CLOCK_MONOTONIC, 1, ctypes.byref(deadline), None)
        if rc == 0:
            return
        if rc != errno.EINTR:
            raise OSError(rc, 'clock_nanosleep')


def distribution(values):
    values = sorted(values)
    def pct(q):
        x = (len(values) - 1) * q
        i, j = math.floor(x), math.ceil(x)
        return values[i] + (values[j] - values[i]) * (x - i)
    return {'n': len(values), 'p50': pct(.5), 'p95': pct(.95),
            'min': values[0], 'max': values[-1]} if values else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--after-run', required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    global LIBC
    LIBC = ctypes.CDLL(None, use_errno=True)
    assert LIBC.prctl(29, 1, 0, 0, 0) == 0
    assert time.get_clock_info('perf_counter').implementation == 'clock_gettime(CLOCK_MONOTONIC)'
    assert all(re.fullmatch(r'[a-z0-9-]+', v) for v in (args.run_id, args.baseline_run, args.after_run))
    root = BASE / 'tests' / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    report = {'run_id': args.run_id, 'completed': False, 'passed': False,
              'state': 'waiting_predecessor', 'cases': [],
              'scope': 'Replay all original baseline actual send offsets, same token IDs/64 outputs, '
                       'cold fresh salts. TTFT includes dispatch lateness. Compare same offered arrivals, '
                       'not only closed-loop saturated rates. No production distribution claim.'}
    def save():
        (root / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    previous = BASE / 'tests' / args.after_run / 'result.json'
    deadline = time.monotonic() + 1800
    while not previous.exists():
        assert time.monotonic() < deadline
        time.sleep(5)
    prior = json.loads(previous.read_text())
    assert prior['completed'] and prior['passed']
    report['revision'] = prior['revision']
    report['prefill_revision'] = prior.get('prefill_revision', prior['revision'])
    source = BASE / 'tests' / args.baseline_run
    baseline = json.loads((source / 'result.json').read_text())
    assert baseline['completed'] and baseline['passed'] and baseline['stage'] == 'baseline'
    report['baseline_run'] = args.baseline_run
    report['baseline_result_sha256'] = hashlib.sha256((source / 'result.json').read_bytes()).hexdigest()
    fixture_root = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests/native-eager-c1-v1')
    fixtures = {n: json.loads((fixture_root / ('cold-eager-reference-' + str(n) + '.json')).read_text())['request']['prompt']
                for n in (256, 1024)}
    assert all(len(p) == n for n, p in fixtures.items())
    assert hashlib.sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest() == baseline['fixtures_sha256']
    def invoke(prompt, scheduled_ns=None):
        payload = {'model': 'glm5', 'prompt': prompt, 'max_tokens': 64, 'temperature': 0,
                   'top_p': 1, 'seed': 1024, 'ignore_eos': True, 'return_token_ids': True,
                   'cache_salt': 'replay-' + uuid.uuid4().hex, 'stream': True,
                   'stream_options': {'include_usage': True}}
        if scheduled_ns is not None:
            absolute_wait(scheduled_ns)
        dispatched = time.perf_counter_ns()
        row = {'start_ns': dispatched if scheduled_ns is None else scheduled_ns,
               'dispatch_ns': dispatched, 'token_ids': [], 'arrivals_ns': [], 'text': ''}
        try:
            request = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                            headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=90) as response:
                row['http_status'] = response.status
                for line in response:
                    if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                        continue
                    now = time.perf_counter_ns(); value = json.loads(line[6:])
                    if value.get('error'):
                        raise RuntimeError(str(value['error']))
                    row['completion_id'] = value.get('id')
                    for choice in value.get('choices', []):
                        ids = choice.get('token_ids') or []
                        row['token_ids'].extend(ids); row['arrivals_ns'].extend([now] * len(ids))
                        row['text'] += choice.get('text', '')
            match = re.match(r'\s*([-+]?\d+(?:\.\d+)?)', row['text'])
            row['success'] = (row['http_status'] == 200 and len(row['token_ids']) == 64
                              and bool(match and Decimal(match.group(1)) == 5))
        except Exception as exc:
            row.update(success=False, error=type(exc).__name__ + ': ' + str(exc))
        row['closed_ns'] = time.perf_counter_ns()
        row['dispatch_lag_ms'] = (dispatched - row['start_ns']) / 1e6
        if row['arrivals_ns']:
            row['ttft_ms'] = (row['arrivals_ns'][0] - row['start_ns']) / 1e6
            row['tpot_ms'] = (row['arrivals_ns'][-1] - row['arrivals_ns'][0]) / 1e6 / 63
        return row
    try:
        report['state'] = 'warmup'; save()
        report['initial_warmups'] = [invoke(fixtures[n]) for n in (256, 1024)]
        save()
        assert all(x['success'] for x in report['initial_warmups'])
        for n in (256, 1024):
            raw = (source / (str(n) + '-requests.json')).read_bytes()
            old = sorted(json.loads(raw), key=lambda x: x['start_ns'])
            assert old and all(x['success'] and len(x['token_ids']) == 64 for x in old)
            origin = min(x['start_ns'] for x in old)
            offsets = [x['start_ns'] - origin for x in old]
            assert max(offsets) <= 181 * 10**9
            with ThreadPoolExecutor(max_workers=32) as pool:
                barrier = threading.Barrier(33)
                def ready_worker():
                    barrier.wait(timeout=10)
                ready = [pool.submit(ready_worker) for _ in range(32)]
                barrier.wait(timeout=10)
                for future in ready:
                    future.result()
                control_start = time.perf_counter_ns() + 10**9
                def check_deadline(index):
                    target = control_start + index * 5_000_000
                    absolute_wait(target)
                    return (time.perf_counter_ns() - target) / 1e6
                control = list(pool.map(check_deadline, range(32)))
                report.setdefault('driver_controls', []).append({
                    'input_tokens': n, 'absolute_monotonic_wait': True,
                    'timerslack_ns': Path('/proc/self/timerslack_ns').read_text().strip(),
                    'dispatch_lag_ms': distribution(control)})
                save()
                assert distribution(control)['p95'] <= 10 and max(control) <= 50
                warm = list(pool.map(invoke, [fixtures[n]] * 4))
                (root / (str(n) + '-warmup.json')).write_text(json.dumps(warm, indent=2) + '\n')
                assert all(x['success'] for x in warm)
                report['state'] = 'replaying-' + str(n); save()
                start = time.perf_counter_ns() + 2 * 10**9
                futures = [pool.submit(invoke, fixtures[n], start + offset) for offset in offsets]
                rows = [future.result() for future in futures]
            for index, row in enumerate(rows):
                row['source_order'] = index
                row['source_start_ns'] = old[index]['start_ns']
            (root / (str(n) + '-requests.json')).write_text(json.dumps(rows, indent=2) + '\n')
            good = [x for x in rows if x['success']]
            case = {'input_tokens': n, 'output_tokens': 64, 'source_requests_sha256': hashlib.sha256(raw).hexdigest(),
                    'source_requests': len(old), 'requests': len(rows), 'valid_requests': len(good),
                    'source_send_span_s': max(offsets) / 1e9,
                    'baseline_ttft_ms': distribution([x['ttft_ms'] for x in old]),
                    'candidate_ttft_ms': distribution([x['ttft_ms'] for x in good]),
                    'baseline_tpot_ms': distribution([x['tpot_ms'] for x in old]),
                    'candidate_tpot_ms': distribution([x['tpot_ms'] for x in good]),
                    'dispatch_lag_ms': distribution([x['dispatch_lag_ms'] for x in rows])}
            if len(good) == len(old):
                case['paired_ttft_difference_ms'] = distribution([new['ttft_ms'] - prior['ttft_ms'] for new, prior in zip(rows, old)])
            report['cases'].append(case); save()
            assert len(good) == len(old), 'Replay functional/HTTP/length failure'
            assert case['dispatch_lag_ms']['p95'] <= 10 and case['dispatch_lag_ms']['max'] <= 50, 'Driver did not follow frozen arrival schedule'
        report.update(completed=True, passed=True, state='same_arrival_latency_collected')
    except Exception as exc:
        report.update(state='needs_specific_followup', failure=type(exc).__name__ + ': ' + str(exc))
    (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n'); save()
    print(json.dumps(report, indent=2))
    assert report['completed'] and report['passed'], report.get('failure')


if __name__ == '__main__':
    main()
