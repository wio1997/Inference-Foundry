#!/usr/bin/env python3
"""Exercise the unchanged official PD proxy and native transfer metrics."""
import datetime
import json
from pathlib import Path
import socket
import time
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    root = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd/tests/smoke-v1')
    root.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'started_host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Stock official V2 connector and proxy deployment smoke, not full accuracy or performance acceptance.', 'requests': []}
    urls = {'P': 'http://172.16.12.32:29481', 'D': 'http://172.16.12.33:29482', 'proxy': 'http://172.16.12.32:29880'}
    def get(url, path):
        with urllib.request.urlopen(url+path, timeout=5) as response:
            return response.read()
    def metrics(label):
        raw = get(urls['D'], '/metrics'); (root/(label+'.txt')).write_bytes(raw); values = {}
        for line in raw.decode().splitlines():
            if line.startswith('#') or not line: continue
            k, v = line.rsplit(' ', 1); k = k.split('{', 1)[0]; values[k] = values.get(k, 0)+float(v)
        return values
    def complete(destination, payload, label):
        request_id = 'official-pd-'+uuid.uuid4().hex
        q = urllib.request.Request(urls[destination]+'/v1/completions', data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'X-Request-Id': request_id})
        start = time.monotonic()
        with urllib.request.urlopen(q, timeout=180) as response:
            raw = response.read(); status = response.status
        value = json.loads(raw)
        record = {'label': label, 'destination': destination, 'request_id': request_id,
                  'request': payload, 'response': value, 'http_status': status, 'elapsed_s_diagnostic': time.monotonic()-start}
        (root/(label+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        report['requests'].append(record)
        assert status == 200 and value.get('choices') and value['choices'][0].get('text'), 'Empty official response'
        return value
    try:
        deadline = time.monotonic()+1200
        while True:
            ready = []
            for role in ('P', 'D'):
                try: ready.append(any(x['id'] == 'glm5' for x in json.loads(get(urls[role], '/v1/models'))['data']))
                except Exception: ready.append(False)
            if all(ready): break
            assert time.monotonic() < deadline, 'Official backend readiness timeout'
            time.sleep(5)
        report['both_backends_ready_host_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report['proxy_health'] = json.loads(get(urls['proxy'], '/healthcheck'))
        before = metrics('metrics-before')
        for count in (256, 1024):
            # Exact prompt IDs come from the already frozen task fixture, no server modifications.
            # Read the completed native request as data, never import or execute its script.
            original = json.loads((Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline/tests/native-eager-c1-v1')/f'cold-eager-reference-{count}.json').read_text())['request']
            assert len(original['prompt']) == count
            payload = {**original, 'model': 'glm5', 'max_tokens': 16, 'stream': False, 'cache_salt': 'official-'+uuid.uuid4().hex}
            native = complete('D', payload, f'native-D-{count}')
            payload['cache_salt'] = 'official-'+uuid.uuid4().hex
            via_proxy = complete('proxy', payload, f'official-proxy-{count}')
            report.setdefault('checks', []).append({'input_tokens': count,
                'native_text': native['choices'][0]['text'], 'official_pd_text': via_proxy['choices'][0]['text'],
                'exact_text_equal': native['choices'][0]['text'] == via_proxy['choices'][0]['text'],
                'arithmetic_smoke_begins_with_5': via_proxy['choices'][0]['text'].lstrip().startswith('5')})
        end = time.monotonic()+20
        while True:
            after = metrics('metrics-after')
            count = after.get('vllm:mooncake_bytes_transferred_count', 0)-before.get('vllm:mooncake_bytes_transferred_count', 0)
            if count > 0 or time.monotonic() >= end: break
            time.sleep(1)
        report['transfer_metric_deltas'] = {k: after.get(k, 0)-before.get(k, 0) for k in after
            if k.startswith('vllm:mooncake') and (k.endswith('_sum') or k.endswith('_count') or k.endswith('_total'))}
        assert count > 0, 'No completed official Mooncake D2D transfer counted'
        assert report['transfer_metric_deltas'].get('vllm:mooncake_num_failed_transfers_total', 0) == 0
        assert all(c['arithmetic_smoke_begins_with_5'] for c in report['checks']), 'Arithmetic functional smoke failed'
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    report['ended_host_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
