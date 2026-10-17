#!/usr/bin/env python3
"""Capture one real official-PD request after the functional smoke, outside timing."""
import argparse
import datetime
import json
from pathlib import Path
import socket
import re
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='path-profile-v1')
    parser.add_argument('--after-run')
    parser.add_argument('--prefill-revision', type=int, default=1)
    parser.add_argument('--decode-revision', type=int, default=1)
    parser.add_argument('--output-tokens', type=int, choices=(16, 32), default=16)
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.run_id)
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
    if args.after_run:
        assert re.fullmatch(r'[a-z0-9-]+', args.after_run)
        formal = json.loads((base/'tests'/args.after_run/'result.json').read_text())
        assert formal['completed'] and formal['passed'], 'Formal timing must finish before profiling'
        assert formal['revision'] == args.decode_revision
        assert formal.get('prefill_revision', formal['revision']) == args.prefill_revision
    else:
        assert args.prefill_revision == args.decode_revision == 1
        assert json.loads((base/'consumer-audits/smoke-validation-v1/result.json').read_text())['completed']
    out = base/'tests'/args.run_id; out.mkdir(parents=True, exist_ok=False)
    urls = {'P': 'http://172.16.12.32:29481', 'D': 'http://172.16.12.33:29482', 'proxy': 'http://172.16.12.32:29880'}
    source = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests/native-eager-c1-v1/cold-eager-reference-256.json')
    prompt = json.loads(source.read_text())['request']['prompt']; assert len(prompt) == 256
    report = {'completed': False, 'scope': 'Path evidence from a real stock-proxy request; profiling and dummy steps are excluded from all performance measurements.',
              'started_host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'requests': [], 'profiles': [],
              'after_formal_run': args.after_run, 'prefill_revision': args.prefill_revision,
              'decode_revision': args.decode_revision}
    def post(role, path, value, timeout=180):
        q = urllib.request.Request(urls[role]+path, data=json.dumps(value).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(q, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw) if raw else None
    def request(role, count):
        payload = {'model': 'glm5', 'prompt': prompt, 'max_tokens': count, 'temperature': 0,
                   'top_p': 1, 'seed': 1024, 'ignore_eos': True, 'cache_salt': 'path-'+uuid.uuid4().hex, 'stream': False}
        value = post(role, '/v1/completions', payload)
        report['requests'].append({'role': role, 'request': payload, 'response': value})
        assert value.get('choices') and value['choices'][0]['text']
    started = []
    try:
        for role in ('P', 'D'):
            post(role, '/start_profile', {}); started.append(role)
            report['profiles'].append({'role': role, 'started': True})
        # The configured schedule has delay2/active1. Two one-step native requests
        # arm the active window before the actual remote-KV request arrives.
        for role in ('P', 'D'):
            for unused in range(2): request(role, 1)
        request('proxy', args.output_tokens)
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    finally:
        for role in reversed(started):
            try:
                post(role, '/stop_profile', {}, timeout=300)
                next(p for p in report['profiles'] if p['role'] == role)['stopped'] = True
            except Exception as exc:
                report['stop_error_'+role] = str(exc)
    report['ended_host_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['completed'] and all(p.get('stopped') for p in report['profiles']), report.get('failure')


if __name__ == '__main__':
    main()
