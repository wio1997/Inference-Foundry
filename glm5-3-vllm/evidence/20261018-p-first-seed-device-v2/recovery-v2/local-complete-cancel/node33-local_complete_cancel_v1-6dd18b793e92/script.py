#!/usr/bin/env python3
"""Client close and immediate reuse, without claiming unobserved engine aborts."""
import json
from pathlib import Path
import re
import socket
import time
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B5-5U-Atlas900RCKA3-33'
    root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33')
    assert not (root / 'divergence-probe/skip-unused-single-token-draft').exists()
    cases = json.loads((root / 'unused-draft-cases-v1.json').read_text())['cases']
    out = root / 'local-complete-cancel-reuse-v1'
    out.mkdir(exist_ok=False)
    report = {'completed': False, 'rows': [], 'scope': 'Eight TCP client closes at body-sent or response-headers and immediate complete max1 reuse. Actual engine abort counters separate; local requests must leave KV/proxy counts idle and D completion unchanged.'}

    def state():
        result = {}
        for node, port in [(32, 29481), (33, 29482)]:
            with urllib.request.urlopen('http://172.16.12.%s:%s/metrics' % (node, port), timeout=5) as response:
                raw = response.read().decode()
            result[str(node)] = {}
            for name in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc', 'request_success_total']:
                values = re.findall(r'^vllm:' + name + r'(?:\{[^\n]*\})?\s+(\S+)', raw, re.M)
                assert values
                result[str(node)][name] = sum(float(v) for v in values)
            result[str(node)]['abort'] = sum(float(line.rsplit(' ', 1)[1]) for line in raw.splitlines() if line.startswith('vllm:request_success_total') and 'abort' in line)
        with urllib.request.urlopen('http://172.16.12.32:29882/healthcheck', timeout=5) as response:
            result['proxy'] = json.load(response)
        return result

    def payload(case, stream):
        body = {'model': 'glm5', 'prompt': case['prompt_ids'][:-2], 'max_tokens': 1, 'min_tokens': 1,
                'ignore_eos': True, 'temperature': 0, 'top_p': 1, 'seed': 1024,
                'return_token_ids': True, 'cache_salt': 'local-cancel-' + uuid.uuid4().hex, 'stream': stream}
        if stream:
            body['stream_options'] = {'include_usage': True}
        return body

    try:
        report['before'] = state()
        for case in cases:
            for point in ['body-sent', 'response-headers']:
                for _ in range(2):
                    body = payload(case, True)
                    data = json.dumps(body).encode()
                    row = {'N': len(body['prompt']), 'close_point': point, 'request': body}
                    report['rows'].append(row)
                    headers = ('POST /v1/completions HTTP/1.1\r\nHost: 172.16.12.32:29882\r\nContent-Type: application/json\r\nX-GLM53-Route: p-complete\r\nX-Request-Id: ' + body['cache_salt'] + '\r\nContent-Length: ' + str(len(data)) + '\r\nConnection: close\r\n\r\n').encode()
                    with socket.create_connection(('172.16.12.32', 29882), timeout=10) as sock:
                        sock.sendall(headers + data)
                        if point == 'response-headers':
                            received = b''
                            while b'\r\n\r\n' not in received:
                                chunk = sock.recv(4096)
                                assert chunk
                                received += chunk
                                assert len(received) < 128000
                            assert received.startswith(b'HTTP/1.1 200')
                            row['bytes_before_close'] = len(received)
                    row['client_closed'] = True
                    reuse = payload(case, False)
                    req = urllib.request.Request('http://172.16.12.32:29882/v1/completions', json.dumps(reuse).encode(),
                                                 {'Content-Type': 'application/json', 'X-GLM53-Route': 'p-complete', 'X-Request-Id': reuse['cache_salt']})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        assert response.headers.get('X-GLM53-Route') == 'p-local-complete'
                        result = json.load(response)
                    assert result['choices'][0]['token_ids'] == [case['prompt_ids'][-2]]
                    assert result['usage']['completion_tokens'] == 1 and not result.get('kv_transfer_params')
                    row['reuse_response'] = result
                    row['reuse_pass'] = True
        deadline = time.monotonic() + 20
        while True:
            after = state()
            if all(after[str(n)][k] == 0 for n in [32, 33] for k in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc']) and after['proxy']['request_num'] == 0:
                break
            assert time.monotonic() < deadline, after
            time.sleep(.2)
        report['after'] = after
        report['P_abort_delta'] = after['32']['abort'] - report['before']['32']['abort']
        report['D_completion_delta'] = after['33']['request_success_total'] - report['before']['33']['request_success_total']
        assert report['D_completion_delta'] == 0
        report['completed'] = True
    except Exception as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert report['completed'], report.get('failure')


if __name__ == '__main__':
    main()
