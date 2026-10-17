#!/usr/bin/env python3
"""Compare producer-known tokens with native decode on existing protected fixtures."""
import argparse
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
URL = {'P': 'http://172.16.12.32:29481', 'D': 'http://172.16.12.33:29482'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--fixture-run', required=True)
    parser.add_argument('--after-profile', required=True)
    parser.add_argument('--revision', type=int)
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    assert all(re.fullmatch(r'[a-z0-9-]+', x)
               for x in (args.run_id, args.fixture_run, args.after_profile))
    prior = json.loads((BASE/'tests'/args.after_profile/'result.json').read_text())
    assert prior['completed'] and all(p['stopped'] for p in prior['profiles'])
    fixture_raw = (BASE/'tests'/args.fixture_run/'fixtures.json').read_bytes()
    fixtures = json.loads(fixture_raw)
    assert len(fixtures) == 24
    root = BASE/'tests'/args.run_id
    root.mkdir(parents=True, exist_ok=False)
    report = {'run_id': args.run_id, 'completed': False, 'passed': False,
              'prefill_revision': prior['prefill_revision'],
              'revision': args.revision if args.revision is not None else prior['decode_revision'],
              'fixtures_sha256': hashlib.sha256(fixture_raw).hexdigest(), 'rows': [],
              'scope': 'Original native P/D request protocol, existing 24 protected fixtures, '
                       'C1 and C4. First-token agreement only; no substitution or speedup implemented.'}

    def post(role, payload, request_id):
        request = urllib.request.Request(URL[role]+'/v1/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
                                                      'X-Request-Id': request_id})
        return urllib.request.urlopen(request, timeout=120)

    def check(fixture):
        rid = 'p-first-compat-'+uuid.uuid4().hex
        payload = {'model': 'glm5', 'prompt': fixture['prompt'], 'max_tokens': 32,
                   'temperature': 0, 'top_p': 1, 'seed': 1024, 'ignore_eos': True,
                   'return_token_ids': True, 'stream': False, 'cache_salt': rid}
        prefill = dict(payload, max_tokens=1, min_tokens=1,
            kv_transfer_params={'do_remote_decode': True, 'do_remote_prefill': False,
                                'remote_engine_id': None, 'remote_block_ids': None,
                                'remote_host': None, 'remote_port': None})
        with post('P', prefill, rid) as response:
            p = json.load(response)
        known = time.perf_counter_ns()
        params = p['kv_transfer_params']
        payload.update(kv_transfer_params=params, stream=True)
        ids, chunks, first = [], [], None
        with post('D', payload, rid) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                    continue
                now = time.perf_counter_ns()
                chunk = json.loads(line[6:]); chunks.append(chunk)
                assert not chunk.get('error'), chunk
                for choice in chunk.get('choices', []):
                    new = choice.get('token_ids') or []
                    if new and first is None:
                        first = now
                    ids.extend(new)
        assert len(ids) == 32
        return {'id': fixture['id'], 'input_tokens': len(fixture['prompt']),
                'request_id': rid, 'P_response': p, 'D_chunks': chunks,
                'P_first_token_id': params['last_token_id'], 'D_first_token_id': ids[0],
                'matches': params['last_token_id'] == ids[0],
                'known_to_D_first_ms': (first-known)/1e6}

    try:
        for concurrency in (1, 4):
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for offset in range(0, len(fixtures), concurrency):
                    rows = list(pool.map(check, fixtures[offset:offset+concurrency]))
                    for row in rows:
                        row['concurrency'] = concurrency
                    report['rows'].extend(rows)
                    (root/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        report.update(completed=True, passed=all(row['matches'] for row in report['rows']),
                      agreement_count=sum(row['matches'] for row in report['rows']))
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
