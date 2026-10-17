#!/usr/bin/env python3
"""Run bounded numerical/extraction tasks through official PD and native D after timing."""
import argparse
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import socket
import time
import urllib.request
import uuid

BASE = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
URL = {'P': 'http://172.16.12.32:29481', 'D': 'http://172.16.12.33:29482',
       'proxy': 'http://172.16.12.32:29880'}


def post(role, path, value):
    request = urllib.request.Request(URL[role] + path, data=json.dumps(value).encode(),
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--after-run', required=True)
    parser.add_argument('--wait-seconds', type=int, default=1800)
    parser.add_argument('--concurrency', type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    assert all(re.fullmatch(r'[a-z0-9-]+', x) for x in (args.run_id, args.after_run))
    root = BASE / 'tests' / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'state': 'waiting_formal_timing', 'tasks': [],
              'concurrency': args.concurrency,
              'after_run': args.after_run, 'run_id': args.run_id,
              'scope': '24 fixed task oracles, same token IDs on stock PD/native D, recorded concurrency and '
                       'outside formal timing. Full model accuracy and exact-token equality are not claimed.'}

    def save():
        (root / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    previous = BASE / 'tests' / args.after_run / 'result.json'
    deadline = time.monotonic() + args.wait_seconds
    while not previous.exists():
        assert time.monotonic() < deadline, 'Formal-run wait expired'
        time.sleep(5)
    formal = json.loads(previous.read_text())
    assert formal['completed'] and formal['passed']
    report['prefill_revision'] = formal.get('prefill_revision', formal['revision'])
    report['decode_revision'] = formal['revision']

    def tokens(text):
        return post('P', '/tokenize', {'model': 'glm5', 'prompt': text,
                                      'add_special_tokens': False})['tokens']

    fixtures = []
    for i in range(8):
        expected = 731942 + i * 113
        key = ['alpha', 'beta', 'gamma', 'delta'][i % 4]
        values = {k: expected + (j + 1) * 71 for j, k in enumerate(['alpha', 'beta', 'gamma', 'delta'])}
        values[key] = expected
        text = ('Read the record and return only the requested numeric value.\n'
                'Record: alpha=14; beta=29. Key: beta. Answer: 29\n'
                'Record: alpha=38; beta=57. Key: alpha. Answer: 38\n'
                'Record: ' + '; '.join(k + '=' + str(v) for k, v in values.items()) +
                '. Key: ' + key + '. Answer:')
        fixtures.append({'id': 'record-' + str(i), 'expected': expected, 'prompt': tokens(text)})
    for i, (a, b, c) in enumerate([(17, 25, 9), (38, 14, 27), (41, 12, 18), (63, 16, 8),
                                  (12, 29, 43), (28, 34, 17), (51, 23, 19), (39, 15, 26)]):
        text = ('Complete arithmetic expressions. Return only the final number.\n'
                '1+2+3=6\n4+5+6=15\n' + str(a) + '+' + str(b) + '+' + str(c) + '=')
        fixtures.append({'id': 'sum-three-' + str(i), 'expected': a + b + c, 'prompt': tokens(text)})
    header = tokens('Read the archive. Return only the numeric value of the requested record.\nArchive:\n')
    filler = tokens('Unrelated archive note: this entry has no requested record.\n')
    for n in (2048, 4096):
        for fraction in (.25, .75):
            for variant in (0, 1):
                key = 'QZ' + str(n) + str(int(fraction * 100)) + str(variant)
                expected = 846273 + variant * 239 + int(fraction * 100)
                fact = tokens('\nRecord ' + key + ' has numeric value ' + str(expected) + '.\n')
                query = tokens('\nEnd archive. What is the numeric value of record ' + key + '? Answer:')
                length = n - len(header) - len(fact) - len(query)
                padding = (filler * (length // len(filler) + 1))[:length]
                split = int(length * fraction)
                prompt = header + padding[:split] + fact + padding[split:] + query
                assert len(prompt) == n
                fixtures.append({'id': 'retrieve-' + key, 'expected': expected, 'prompt': prompt})
    assert len(fixtures) == 24
    raw = json.dumps(fixtures, sort_keys=True).encode()
    (root / 'fixtures.json').write_bytes(raw)
    report['fixtures_sha256'] = hashlib.sha256(raw).hexdigest()
    report['state'] = 'checking_tasks'
    def check(fixture, role):
            payload = {'model': 'glm5', 'prompt': fixture['prompt'], 'max_tokens': 32,
                       'temperature': 0, 'top_p': 1, 'seed': 1024, 'return_token_ids': True,
                       'stream': False, 'stop': ['\n'], 'cache_salt': 'protect-' + uuid.uuid4().hex}
            response = post(role, '/v1/completions', payload)
            text = response['choices'][0]['text']
            match = re.search(r'[-+]?\d+(?:\.\d+)?', text)
            return {'response': response,
                         'numeric_pass': bool(match and Decimal(match.group()) == fixture['expected']),
                         'exact_format_pass': text.strip() == str(fixture['expected'])}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for offset in range(0, len(fixtures), args.concurrency):
            batch = fixtures[offset:offset + args.concurrency]
            rows = [{**{k: fixture[k] for k in ('id', 'expected')},
                     'input_tokens': len(fixture['prompt'])} for fixture in batch]
            for role in ('proxy', 'D'):
                values = list(pool.map(lambda fixture: check(fixture, role), batch))
                for row, value in zip(rows, values):
                    row[role] = value
            report['tasks'].extend(rows)
            save()
    report['scores'] = {role: sum(row[role]['numeric_pass'] for row in report['tasks'])
                        for role in ('proxy', 'D')}
    report['completed'] = True
    report['state'] = 'tasks_collected_compare_protected_set'
    save()
    (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
