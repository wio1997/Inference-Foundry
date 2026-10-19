#!/usr/bin/env python3
"""Diagnose known P token availability with the original native P/D request protocol."""
import argparse
from decimal import Decimal
import json
import math
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
    parser.add_argument('--after-profile', required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    assert all(re.fullmatch(r'[a-z0-9-]+', x) for x in (args.run_id, args.after_profile))
    prior = json.loads((BASE/'tests'/args.after_profile/'result.json').read_text())
    assert prior['completed'] and all(p.get('stopped') for p in prior['profiles'])
    root = BASE/'tests'/args.run_id
    root.mkdir(parents=True, exist_ok=False)
    report = {'run_id': args.run_id, 'completed': False, 'passed': False,
              'prefill_revision': prior['prefill_revision'], 'revision': prior['decode_revision'],
              'scope': 'Single client, unchanged native request metadata copied from P to D as in '
                       'official proxy. Both availability timestamps use this client monotonic clock. '
                       'Held-known-token interval is diagnostic, not an implemented speedup or a '
                       'claim that every P/D first token is interchangeable.', 'rows': []}
    source = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests/native-eager-c1-v1')
    def post(role, payload, request_id):
        q = urllib.request.Request(URL[role]+'/v1/completions', data=json.dumps(payload).encode(),
                                   headers={'Content-Type': 'application/json', 'X-Request-Id': request_id})
        return urllib.request.urlopen(q, timeout=90)
    try:
        for n in (256, 1024):
            prompt = json.loads((source/f'cold-eager-reference-{n}.json').read_text())['request']['prompt']
            assert len(prompt) == n
            for repetition in range(4):
                request_id = 'known-first-'+uuid.uuid4().hex
                payload = {'model': 'glm5', 'prompt': prompt, 'max_tokens': 64,
                           'temperature': 0, 'top_p': 1, 'seed': 1024, 'ignore_eos': True,
                           'return_token_ids': True, 'cache_salt': request_id, 'stream': True}
                prefill = dict(payload)
                prefill.update(max_tokens=1, min_tokens=1, stream=False,
                    kv_transfer_params={'do_remote_decode': True, 'do_remote_prefill': False,
                        'remote_engine_id': None, 'remote_block_ids': None,
                        'remote_host': None, 'remote_port': None})
                started = time.perf_counter_ns()
                with post('P', prefill, request_id) as response:
                    p = json.load(response)
                known = time.perf_counter_ns()
                params = p['kv_transfer_params']
                token = params['last_token_id']
                assert isinstance(token, int) and token >= 0 and params['do_remote_prefill']
                payload['kv_transfer_params'] = params
                ids, arrivals, text, chunks = [], [], '', []
                sent = time.perf_counter_ns()
                with post('D', payload, request_id) as response:
                    status = response.status
                    for line in response:
                        if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                            continue
                        now = time.perf_counter_ns()
                        value = json.loads(line[6:]); chunks.append(value)
                        if value.get('error'):
                            raise RuntimeError(str(value['error']))
                        for choice in value.get('choices', []):
                            new = choice.get('token_ids') or []
                            ids.extend(new); arrivals.extend([now]*len(new))
                            text += choice.get('text', '')
                match = re.match(r'\s*([-+]?\d+(?:\.\d+)?)', text)
                valid = status == 200 and len(ids) == 64 and bool(match and Decimal(match.group(1)) == 5)
                row = {'input_tokens': n, 'repetition': repetition, 'warmup': repetition == 0,
                       'request_id': request_id, 'P_response': p, 'D_chunks': chunks,
                       'P_last_token_id': token, 'D_first_token_id': ids[0] if ids else None,
                       'first_id_matches': bool(ids and ids[0] == token), 'functional_pass': valid,
                       'P_elapsed_ms': (known-started)/1e6, 'P_known_to_D_send_ms': (sent-known)/1e6,
                       'D_send_to_first_ms': (arrivals[0]-sent)/1e6 if arrivals else None,
                       'P_known_to_D_first_ms': (arrivals[0]-known)/1e6 if arrivals else None}
                report['rows'].append(row)
                (root/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
                assert valid
        report.update(completed=True, passed=True)
        report['first_id_match_count'] = sum(r['first_id_matches'] for r in report['rows'])
        report['observations'] = []
        for n in (256, 1024):
            rows = [r for r in report['rows'] if r['input_tokens'] == n and not r['warmup']]
            vals = sorted(r['P_known_to_D_first_ms'] for r in rows)
            report['observations'].append({'input_tokens': n, 'n': len(rows),
                'known_to_D_first_p50_ms': vals[len(vals)//2], 'min_ms': min(vals), 'max_ms': max(vals),
                'first_id_matches': sum(r['first_id_matches'] for r in rows)})
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    assert report['completed'] and report['passed'], report.get('failure')


if __name__ == '__main__':
    main()
