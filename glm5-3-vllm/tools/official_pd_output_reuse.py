#!/usr/bin/env python3
"""Check first-output completion/cancellation followed by immediate request reuse."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from pathlib import Path
import re
import socket
import time
import urllib.request
import uuid

BASE = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
PROXY = 'http://172.16.12.32:29880'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--after-tasks', required=True)
    parser.add_argument('--transitions', action='store_true')
    parser.add_argument('--transition-waves', type=int, choices=range(1, 17), default=4)
    parser.add_argument('--ignore-eos', action='store_true',
                        help='Exercise the narrow producer-first prototype eligibility')
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    assert all(re.fullmatch(r'[a-z0-9-]+', v) for v in (args.run_id, args.after_tasks))
    root = BASE / 'tests' / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    report = {'run_id': args.run_id, 'completed': False, 'passed': False,
              'state': 'waiting_protected_tasks', 'pairs': [],
              'scope': 'C4 max1 completion and first-token cancellation, each followed immediately '
                       'by a different42-answer request. Functional/lifetime checks, outside timing.'}
    def save():
        (root / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    previous = BASE / 'tests' / args.after_tasks / 'result.json'
    deadline = time.monotonic() + 1200
    while not previous.exists():
        assert time.monotonic() < deadline
        time.sleep(5)
    tasks = json.loads(previous.read_text())
    assert tasks['completed'] and tasks['scores'] == {'proxy': 24, 'D': 24}
    report['revision'] = tasks['decode_revision']
    report['prefill_revision'] = tasks['prefill_revision']
    fixtures = {}
    source = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests/native-eager-c1-v1')
    for n in (256, 1024):
        fixtures[n] = json.loads((source / ('cold-eager-reference-' + str(n) + '.json')).read_text())['request']['prompt']
    def post(value):
        request = urllib.request.Request(PROXY + '/v1/completions', data=json.dumps(value).encode(),
                                        headers={'Content-Type': 'application/json',
                                                 'X-Request-Id': 'reuse-' + uuid.uuid4().hex})
        return urllib.request.urlopen(request, timeout=90)
    def numeric(text, expected):
        match = re.match(r'\s*([-+]?\d+(?:\.\d+)?)', text)
        return bool(match and Decimal(match.group(1)) == expected)
    def first_only(mode, n):
        value = {'model': 'glm5', 'prompt': fixtures[n], 'max_tokens': 1 if mode == 'max1' else 64,
                 'temperature': 0, 'top_p': 1, 'seed': 1024, 'return_token_ids': True,
                 'cache_salt': 'reuse-' + uuid.uuid4().hex, 'stream': mode == 'cancel'}
        if args.ignore_eos:
            value['ignore_eos'] = True
        row = {'mode': mode, 'input_tokens': n}
        with post(value) as response:
            row['first_http_status'] = response.status
            if mode == 'max1':
                first = json.load(response)
                row['first_response'] = first
                choice = first['choices'][0]
                row['first_pass'] = (response.status == 200 and numeric(choice['text'], 5)
                                     and len(choice.get('token_ids') or []) == 1)
            else:
                chunks = []
                first_ids = []
                for line in response:
                    if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                        continue
                    chunk = json.loads(line[6:]); chunks.append(chunk)
                    first_ids.extend(token for c in chunk.get('choices', []) for token in (c.get('token_ids') or []))
                    if first_ids:
                        break
                row.update(first_chunks=chunks, cancelled_after_tokens=len(first_ids),
                           first_pass=response.status == 200 and bool(first_ids))
        return row

    def replacement(unused=None):
        row = {}
        # No grace period: this must be safe while the previous draft can be live.
        next_value = {'model': 'glm5', 'prompt': 'Complete arithmetic expressions. Return only the final number.\n1+1=2\n3*4=12\n17+25=',
                      'max_tokens': 16, 'temperature': 0, 'top_p': 1, 'seed': 1024,
                      'return_token_ids': True, 'stop': ['\n'], 'stream': False,
                      'cache_salt': 'replacement-' + uuid.uuid4().hex}
        with post(next_value) as response:
            row['replacement_http_status'] = response.status
            row['replacement_response'] = json.load(response)
        row['replacement_pass'] = (row['replacement_http_status'] == 200
                                   and numeric(row['replacement_response']['choices'][0]['text'], 42))
        return row

    def pair(mode, n):
        row = first_only(mode, n)
        row.update(replacement())
        return row
    try:
        report['state'] = 'checking_immediate_reuse'; save()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for mode in ('max1', 'cancel'):
                for wave in range(2):
                    rows = list(pool.map(lambda n: pair(mode, n), [256, 1024, 256, 1024]))
                    for row in rows:
                        row['wave'] = wave
                    report['pairs'].extend(rows); save()
                    assert all(x['first_pass'] and x['replacement_pass'] for x in rows)
        if args.transitions:
            report['state'] = 'checking_singleton_c4_transitions'
            report['transitions'] = []
            report['transition_scope'] = (
                'HTTP concurrency1->4 and4->1 immediately after max1/first-token cancellation. '
                'Checks observable buffer/request ownership transitions; physical KV block IDs '
                'are not inspected, so this is not proof of recycling a specific block.')
            save()
            with ThreadPoolExecutor(max_workers=4) as pool:
                for wave in range(args.transition_waves):
                    mode = 'max1' if wave % 2 == 0 else 'cancel'
                    first = first_only(mode, 256 if wave % 2 == 0 else 1024)
                    replacements = list(pool.map(replacement, range(4)))
                    row = {'wave': wave, 'direction': '1_to_4', 'mode': mode,
                           'first': first, 'replacements': replacements}
                    report['transitions'].append(row); save()
                    assert first['first_pass'] and all(x['replacement_pass'] for x in replacements)
                    firsts = list(pool.map(lambda n: first_only(mode, n), [256, 1024, 256, 1024]))
                    last = replacement()
                    row = {'wave': wave, 'direction': '4_to_1', 'mode': mode,
                           'firsts': firsts, 'replacement': last}
                    report['transitions'].append(row); save()
                    assert all(x['first_pass'] for x in firsts) and last['replacement_pass']
        report.update(completed=True, passed=True, state='immediate_reuse_functional_checks_passed')
    except Exception as exc:
        report.update(state='needs_specific_followup', failure=type(exc).__name__ + ': ' + str(exc))
    (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n'); save()
    print(json.dumps(report, indent=2))
    assert report['completed'] and report['passed'], report.get('failure')


if __name__ == '__main__':
    main()
