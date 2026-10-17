#!/usr/bin/env python3
"""Validate fresh-D external KV consumption, deterministic output and collect one profile."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import time
import urllib.error
import urllib.request

ROOT = Path('/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-functional-v1')
MODEL = 'glm53-w8a8c8-memcache'
URL = {32: 'http://172.16.12.32:29481', 33: 'http://172.16.12.33:29482'}


def http(node, path, data=None, timeout=240):
    request = urllib.request.Request(URL[node]+path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return raw.decode() if path == '/metrics' else (json.loads(raw) if raw else None)


def get_keys(metrics):
    return sum(float(s.rsplit(' ', 1)[1]) for s in metrics.splitlines()
               if s.startswith('vllm:ascend_store_load_get_keys_total{'))


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    ROOT.mkdir(parents=True, exist_ok=False)
    report = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'One synthetic same-TP full-model KV-pool consumption check; no performance or accuracy acceptance.',
              'profiles': [], 'completed': False}
    started = []
    try:
        for node in (32, 33):
            assert any(m['id'] == MODEL for m in http(node, '/v1/models')['data'])
        def tokenize(text, special=False):
            return http(32, '/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': special})['tokens']
        header = tokenize('Reference MF20261017-'+report['started_utc']+'. Read the following notes.\n', True)
        paragraph = tokenize('These notes describe a memory transfer validation. The supplied facts and their order remain the same.\n')
        question = tokenize('\nQuestion: What is 2 + 3? Answer with the result:')
        count = 480
        body_count = count-len(header)-len(question)
        assert paragraph and body_count > 0
        tokens = header+(paragraph*((body_count+len(paragraph)-1)//len(paragraph)))[:body_count]+question
        assert len(tokens) == count
        (ROOT/'prompt-token-ids.json').write_text(json.dumps(tokens)+'\n')
        report['prompt_tokens'] = count
        report['prompt_ids_sha256'] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        before = http(33, '/metrics')
        (ROOT/'decode-metrics-before.txt').write_text(before)
        report['decode_get_keys_before'] = get_keys(before)
        for node in (32, 33):
            http(node, '/start_profile', {})
            started.append(node)
            report['profiles'].append({'node': node, 'started': True})
        payload = {'model': MODEL, 'prompt': tokens, 'max_tokens': 16, 'temperature': 0,
                   'top_p': 1, 'seed': 1024, 'stream': False}
        report['prefill'] = http(32, '/v1/completions', payload)
        (ROOT/'prefill-response.json').write_text(json.dumps(report['prefill'], indent=2)+'\n')
        # Save completion is asynchronous; do not send D until P's pending block
        # releases have drained, which certifies that the save no longer owns them.
        deadline = time.monotonic()+45
        while True:
            metrics = http(32, '/metrics')
            pending = [float(s.rsplit(' ', 1)[1]) for s in metrics.splitlines()
                       if s.startswith('vllm:ascend_store_delayed_release_requests{')]
            if pending and all(v == 0 for v in pending):
                break
            assert time.monotonic() < deadline, 'P asynchronous save did not drain'
            time.sleep(1)
        (ROOT/'prefill-metrics-after-save.txt').write_text(metrics)
        report['decode'] = http(33, '/v1/completions', payload)
        (ROOT/'decode-response.json').write_text(json.dumps(report['decode'], indent=2)+'\n')
        # Metrics export runs on the normal engine logging interval.
        deadline = time.monotonic()+30
        while True:
            after = http(33, '/metrics')
            delta = get_keys(after)-report['decode_get_keys_before']
            if delta > 0 or time.monotonic() >= deadline:
                break
            time.sleep(1)
        (ROOT/'decode-metrics-after.txt').write_text(after)
        report['decode_get_keys_delta'] = delta
        report['outputs_equal'] = report['prefill']['choices'][0]['text'] == report['decode']['choices'][0]['text']
        report['nonempty_output'] = bool(report['decode']['choices'][0]['text'])
        report['decode_cached_tokens'] = (report['decode']['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
        assert report['outputs_equal'] and report['nonempty_output'], 'P/D deterministic outputs differ'
        assert delta > 0, 'No completed external KV get observed on D'
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    finally:
        for node in reversed(started):
            try:
                http(node, '/stop_profile', {}, timeout=300)
                next(p for p in report['profiles'] if p['node'] == node)['stopped'] = True
            except Exception as exc:
                report['profile_stop_error_'+str(node)] = str(exc)
        report['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (ROOT/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
    assert report['completed'], report.get('failure')
    assert all(p.get('stopped') for p in report['profiles']), 'Profiler did not stop cleanly'


if __name__ == '__main__':
    main()
