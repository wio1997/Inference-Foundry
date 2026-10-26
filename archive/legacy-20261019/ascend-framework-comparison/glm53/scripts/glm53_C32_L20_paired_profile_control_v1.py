"""L20 DRAFT native profiler control; no inference, SDK or framework imports.

SGLang: one existing control scope (API36) because start_profile fans out to all
DP ranks. vLLM: two necessary scopes (API43/API45) because each hybrid-LB client
owns only its eight local EngineCores. One start and one bounded stop per scope.
Admission is native 16-DP with at least one running request; otherwise no start.
"""
import argparse
import concurrent.futures
import hashlib
import json
import re
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

CAP = 8 * 1024 ** 2


def gauge_occupancy(framework, node, text):
    names = ({'sglang:num_running_reqs': 'running', 'sglang:num_queue_reqs': 'waiting'}
             if framework == 'sglang' else
             {'vllm:num_requests_running': 'running',
              'vllm:num_requests_waiting': 'waiting'})
    rank_label = 'dp_rank' if framework == 'sglang' else 'engine'
    result = {}
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        match = re.fullmatch(r'([^ {]+)\{(.*)\}\s+(\S+)(?:\s+\S+)?', line)
        if not match or match.group(1) not in names:
            continue
        labels = match.group(2)
        if framework == 'sglang' and 'engine_type="decode"' not in labels:
            continue
        rank = re.search(r'(?:^|,)' + rank_label + r'="(\d+)"(?:,|$)', labels)
        assert rank is not None, 'Native per-DP label missing'
        dp = int(rank.group(1))
        value = float(match.group(3))
        assert value.is_integer() and value >= 0
        row = result.setdefault(dp, {'node': node, 'native_dp': dp})
        kind = names[match.group(1)]
        if kind in row:
            assert row[kind] == int(value), 'Duplicate native gauges disagree'
        row[kind] = int(value)
    return result


def request(url, body=None, timeout=10, allowed_prefixes=(), method=None):
    assert any(url.startswith(prefix) for prefix in allowed_prefixes)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = None if body is None else json.dumps(body).encode()
    if method is None:
        method = 'GET' if body is None else 'POST'
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'},
                                 method=method)
    started = time.monotonic_ns()
    try:
        with opener.open(req, timeout=timeout) as response:
            status = response.status
            raw = response.read(CAP + 1)
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read(CAP + 1)
    assert len(raw) <= CAP
    text = raw.decode()
    result = {'url': url, 'method': method, 'status': status, 'bytes': len(raw),
              'sha256': hashlib.sha256(raw).hexdigest(),
              'dispatch_monotonic_ns': started,
              'terminal_monotonic_ns': time.monotonic_ns(), 'body': text}
    if url.endswith('/metrics'):
        result['body'] = '\n'.join(line for line in text.splitlines()
                                   if line.startswith(('sglang:num_running_reqs{',
                                                       'sglang:num_queue_reqs{',
                                                       'vllm:num_requests_running{',
                                                       'vllm:num_requests_waiting{')))
    return result


def control(settings, request_fn=request, pause=time.sleep,
            monotonic=time.monotonic, save=lambda value: None):
    assert settings['project_id'] == 'glm53'
    framework = settings['framework']
    assert framework in ('sglang', 'vllm')
    assert settings['diagnostic_only'] is True
    assert settings['execution_authorized'] is True
    endpoints = settings['metrics_endpoints']
    expected = {int(node): tuple(bounds) for node, bounds in settings['dp_expectations'].items()}
    allowed = tuple(settings['allowed_prefixes'])
    result = {'project_id': 'glm53', 'framework': framework, 'run_id': settings['run_id'],
              'diagnostic_only': True, 'model_requests': 0, 'SDK_init_calls': 0,
              'admission_deadline_seconds': settings['admission_deadline_seconds'],
              'profile_window_seconds': settings['profile_window_seconds'],
              'profile_stop_deadline_seconds': settings['profile_stop_deadline_seconds'],
              'profile_start_attempts': {}, 'profile_stop_attempts': {},
              'events': [], 'pass_control': False, 'profile_completion': 'NOT_STARTED'}

    def record(kind, value):
        result['events'].append({'kind': kind, 'monotonic_seconds': monotonic(),
                                 'value': value})
        save(result)

    def get(url, body=None, timeout=10, method=None):
        value = request_fn(url, body, timeout, allowed, method)
        record('HTTP', value)
        return value

    admitted = False
    attempted = []
    start_ok = False

    def scope_calls(operation):
        # Disjoint hybrid-DP scopes share EP collectives: issue their controls
        # together rather than block one scope before starting the other.
        replies = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = {}
            for node in settings['profile_endpoints']:
                key = str(node)
                if operation == 'start':
                    attempted.append(key)
                    result['profile_start_attempts'][key] = 1
                else:
                    if key not in attempted:
                        continue
                    result['profile_stop_attempts'][key] = 1
                pending[key] = pool.submit(
                    request_fn, f'http://172.16.12.{node}:30400/{operation}_profile',
                    {}, 90 if operation == 'start' else 180, allowed, 'POST')
            # Keep progress writes on this one thread, including failures.
            for key, future in pending.items():
                try:
                    reply = future.result()
                    record('HTTP', reply)
                except Exception as error:
                    reply = {'error': repr(error), 'scope': key}
                    record('HTTP_EXCEPTION', reply)
                replies[key] = reply
        result[operation + '_responses'] = replies
        return bool(replies) and all(v.get('status') == 200 for v in replies.values())
    try:
        if framework == 'sglang':
            info = get(settings['profile_endpoint'] + '/get_server_info')
            assert info['status'] == 200
            parsed = json.loads(info['body'])
            selected = {key: parsed[key] for key in settings['server_info_keys']}
            result['server_info_selected'] = selected
            result['server_info_pending_keys'] = settings['server_info_pending_keys']
            for key, value in settings['server_info_expectations'].items():
                assert selected[key] == value, (key, selected[key], value)

        deadline = monotonic() + settings['admission_deadline_seconds']
        last = None
        while monotonic() < deadline:
            native = [get(endpoint, None, 10) for endpoint in endpoints]
            assert all(row['status'] == 200 for row in native)
            counts = {}
            for row in native:
                node = int(re.search(r'172\.16\.12\.(\d+)', row['url']).group(1))
                local = gauge_occupancy(framework, node, row['body'])
                low, high = expected[node]
                assert set(local) == set(range(low, high)), (node, sorted(local))
                for dp, value in local.items():
                    assert dp not in counts, 'Native global DP reported twice'
                    counts[dp] = value
            assert set(counts) == set(range(16))
            all_active = all(value['running'] >= 1 for value in counts.values())
            last = {'counts': counts, 'all16_running': all_active,
                    'monotonic_seconds': monotonic()}
            record('D16_admission_sample', last)
            if all_active:
                admitted = True
                break
            pause(settings['admission_poll_seconds'])
        if not admitted:
            result['admission_counts'] = last
            result['profile_completion'] = 'NOT_STARTED_ADMISSION_TIMEOUT'
            return result

        if framework == 'sglang':
            attempted.append('36')
            result['profile_start_attempts']['36'] = 1
            start = get(settings['profile_endpoint'] + '/start_profile',
                        settings['profile_body'], 90)
            result['start_response'] = start
            start_ok = start['status'] == 200
        else:
            start_ok = scope_calls('start')

        if not start_ok:
            result['profile_completion'] = 'START_FAILED_PARTIAL_STATE_PRESERVED'
            return result

        pause(settings['profile_window_seconds'])
        result['profile_completion'] = 'PENDING_NATIVE_TRACE_AND_ITERATION_AUDIT'
    except Exception as error:
        result['error'] = repr(error)
        result['profile_completion'] = 'FAILED_PRESERVED'
    finally:
        # A failed/timeout start may already have reached some workers. Stop
        # each attempted scope once, even if a response/other control failed.
        stop_ok = False
        if attempted:
            try:
                if framework == 'sglang':
                    result['profile_stop_attempts']['36'] = 1
                    reply = get(settings['profile_endpoint'] + '/stop_profile', {}, 180)
                    result['stop_response'] = reply
                    result['stop_acknowledged'] = reply['status'] == 200
                    result['stop_inactive_duplicate'] = bool(
                        reply['status'] == 500 and 'not in progress' in reply['body'].lower())
                    # Preserve raw HTTP500; native completion is a separate gate.
                    stop_ok = reply['status'] == 200
                else:
                    stop_ok = scope_calls('stop')
            except Exception as error:
                result['stop_error'] = repr(error)
        result['pass_control'] = bool(admitted and start_ok and stop_ok and not result.get('error'))
        save(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == args.source_sha256
    raw = Path(args.settings).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.sha256
    settings = json.loads(raw)
    assert socket.gethostname() == settings['expected_hostname']
    root = Path('/data/wio/inference-foundry/glm53/runs') / settings['run_id']
    assert root == Path(args.settings).parent
    with (root / 'L20-profile-control.once').open('x') as file:
        file.write(str(__import__('os').getpid()))

    def save(value):
        path = root / 'L20-profile-control-progress.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)

    result = control(settings, save=save)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
