#!/usr/bin/env python3
"""Measure a real streaming PD front door, then compare first-token ownership."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid

MODEL = 'glm53-w8a8c8-memcache'
URL = {32: 'http://172.16.12.32:29481', 33: 'http://172.16.12.33:29482'}
BASE = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')


def request(url, payload=None, timeout=600):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: '+exc.read().decode(errors='replace')[:4000]) from exc


def rpc(node, path, payload=None):
    with request(URL[node]+path, payload) as response:
        raw = response.read()
    return raw.decode() if path == '/metrics' else (json.loads(raw) if raw else None)


def backend_stream(payload):
    with request(URL[33]+'/v1/completions', payload) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                continue
            chunk = json.loads(line[6:])
            if chunk.get('error'):
                raise RuntimeError(str(chunk['error']))
            yield chunk


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def distribution(values):
    return {'n': len(values), 'p50': percentile(values, .5), 'p90': percentile(values, .9),
            'p95': percentile(values, .95), 'min': min(values) if values else None,
            'max': max(values) if values else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('baseline', 'handoff'), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--reference-run', default='correctness-v4')
    parser.add_argument('--scope', choices=('all', 'c1'), default='all')
    args = parser.parse_args()
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    assert all(re.fullmatch(r'[a-z0-9-]+', v) for v in (args.run_id, args.reference_run))
    reference_root = BASE/'tests'/args.reference_run
    gate = json.loads((reference_root/'result.json').read_text())
    assert gate['passed'], 'Do not accept timing before the protected correctness suite passes'
    root = BASE/'tests'/args.run_id
    root.mkdir(exist_ok=False)
    lock = threading.Lock()
    server_records = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def log_message(self, *_):
            pass

        def do_POST(self):
            started = time.perf_counter_ns()
            record = {'proxy_id': 'pd-'+uuid.uuid4().hex}
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                mode = payload.pop('_pd_mode')
                assert mode in ('naive', 'append_first')
                assert payload['temperature'] == 0 and payload['ignore_eos'] is True
                assert isinstance(payload['prompt'], list) and payload['max_tokens'] >= 2
                record.update(mode=mode, original_prompt_tokens=len(payload['prompt']),
                              requested_output_tokens=payload['max_tokens'], cache_salt=payload['cache_salt'])
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()

                def emit(ids, text, finish=None):
                    obj = {'id': record['proxy_id'], 'object': 'text_completion', 'model': MODEL,
                           'choices': [{'index': 0, 'text': text, 'token_ids': ids, 'finish_reason': finish}]}
                    self.wfile.write(('data: '+json.dumps(obj)+'\n\n').encode())
                    self.wfile.flush()

                p_payload = dict(payload, max_tokens=1, stream=False)
                p_payload.pop('stream_options', None)
                prefill = rpc(32, '/v1/completions', p_payload)
                first = prefill['choices'][0]
                assert len(first['token_ids']) == 1
                record['producer_id'] = prefill['id']
                record['producer_complete_ms'] = (time.perf_counter_ns()-started)/1e6
                record['producer_first_token_id'] = first['token_ids'][0]
                d_payload = dict(payload, stream=True, stream_options={'include_usage': True})
                if mode == 'append_first':
                    emit(first['token_ids'], first['text'])
                    d_payload['prompt'] = payload['prompt']+first['token_ids']
                    d_payload['max_tokens'] = payload['max_tokens']-1
                record['decode_prompt_tokens'] = len(d_payload['prompt'])
                record['decode_requested_output_tokens'] = d_payload['max_tokens']
                for chunk in backend_stream(d_payload):
                    record['consumer_id'] = chunk['id']
                    for choice in chunk.get('choices', []):
                        ids = choice.get('token_ids') or []
                        if ids and 'decode_first_ms' not in record:
                            record['decode_first_ms'] = (time.perf_counter_ns()-started)/1e6
                        if ids or choice.get('text') or choice.get('finish_reason'):
                            emit(ids, choice.get('text', ''), choice.get('finish_reason'))
                    if chunk.get('usage'):
                        record['decode_native_usage'] = chunk['usage']
                record['proxy_complete_ms'] = (time.perf_counter_ns()-started)/1e6
                record['success'] = True
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except Exception as exc:
                record['success'] = False
                record['error'] = type(exc).__name__+': '+str(exc)
                try:
                    self.wfile.write(('data: '+json.dumps({'error': record['error']})+'\n\n').encode())
                    self.wfile.flush()
                except Exception:
                    pass
            finally:
                self.close_connection = True
                with lock:
                    server_records.append(record)
                    with (root/'proxy-events.jsonl').open('a') as out:
                        out.write(json.dumps(record)+'\n')

    server = ThreadingHTTPServer(('127.0.0.1', 29500), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    report = {'phase': args.phase, 'started_host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Temporary representative greedy workloads; closed-loop clients; no production distribution or SLO claim.',
            'correctness_gate': str(reference_root/'result.json'), 'blocks': [], 'passed': False,
              'instrumentation': 'No profiler; token IDs returned for exact output checks. Client arrival times measured at the real HTTP streaming front door on node32. Cache salts change cache identity, not model input IDs.'}
    rows = []
    fixtures = {}

    def fixture(n, output):
        for path in sorted(reference_root.glob('*-local-reference-node33.json')):
            item = json.loads(path.read_text())
            if len(item['request']['prompt']) == n:
                break
        else:
            raise AssertionError(f'No protected native-prefix reference input of length {n}')
        prompt = item['request']['prompt']
        payload = {'model': MODEL, 'prompt': prompt, 'max_tokens': output, 'temperature': 0,
                   'seed': 1024, 'top_p': 1, 'ignore_eos': True, 'return_token_ids': True,
                   'cache_salt': 'reference-'+uuid.uuid4().hex, 'stream': False}
        if len(item['response']['choices'][0]['token_ids']) == output:
            reference = item['response']
        else:
            # Establish the exact same native block-cache boundary as a KV
            # transfer. Cold full prefill can have different quantized logits.
            rpc(33, '/v1/completions', dict(payload, max_tokens=1))
            reference = rpc(33, '/v1/completions', payload)
        data = {'prompt': prompt, 'reference_ids': reference['choices'][0]['token_ids'],
                'reference_text': reference['choices'][0]['text'],
                'prompt_sha256': hashlib.sha256(json.dumps(prompt).encode()).hexdigest()}
        assert len(data['reference_ids']) == output
        fixtures[(n, output)] = data
        (root/f'fixture-{n}-{output}.json').write_text(json.dumps(data, indent=2)+'\n')

    def invoke(n, output, mode, case, round_index, warmup):
        data = fixtures[(n, output)]
        payload = {'model': MODEL, 'prompt': data['prompt'], 'max_tokens': output,
                   'temperature': 0, 'top_p': 1, 'seed': 1024, 'ignore_eos': True,
                   'return_token_ids': True, 'cache_salt': uuid.uuid4().hex, '_pd_mode': mode,
                   'stream': True, 'stream_options': {'include_usage': True}}
        start = time.perf_counter_ns()
        item = {'case': case, 'mode': mode, 'round': round_index, 'warmup': warmup,
                'cache_salt': payload['cache_salt'], 'start_ns': start, 'input_tokens': n,
                'requested_output_tokens': output, 'token_arrival_ns': [], 'token_ids': [], 'text': ''}
        try:
            with request('http://127.0.0.1:29500/v1/completions', payload) as response:
                for line in response:
                    if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get('error'):
                        raise RuntimeError(str(chunk['error']))
                    item['proxy_id'] = chunk['id']
                    for choice in chunk.get('choices', []):
                        now = time.perf_counter_ns()
                        ids = choice.get('token_ids') or []
                        item['token_ids'].extend(ids)
                        item['token_arrival_ns'].extend([now]*len(ids))
                        text = choice.get('text', '')
                        item['text'] += text
                        if text.strip() and 'first_visible_ns' not in item:
                            item['first_visible_ns'] = now
            item['success'] = item['token_ids'] == data['reference_ids'] and item['text'] == data['reference_text']
            if not item['success']:
                item['error'] = 'Token IDs or text differ from the protected native-prefix reference'
        except Exception as exc:
            item['success'] = False
            item['error'] = type(exc).__name__+': '+str(exc)
        item['closed_ns'] = time.perf_counter_ns()
        times = item['token_arrival_ns']
        if times:
            item['ttft_ms'] = (times[0]-start)/1e6
            item['last_token_ms'] = (times[-1]-start)/1e6
            item['itl_ms'] = [(b-a)/1e6 for a, b in zip(times, times[1:])]
            item['tpot_ms'] = (times[-1]-times[0])/1e6/max(1, len(times)-1)
            item['first_gap_ms'] = (times[1]-times[0])/1e6 if len(times)>1 else None
        if 'first_visible_ns' in item:
            item['ttf_visible_ms'] = (item['first_visible_ns']-start)/1e6
        with lock:
            rows.append(item)
            with (root/'client-events.jsonl').open('a') as out:
                out.write(json.dumps(item)+'\n')
        return item

    def block(n, output, concurrency, count, mode, case, round_index, warmup=False):
        barrier = threading.Barrier(concurrency)

        def worker(worker_id):
            barrier.wait()
            return [invoke(n, output, mode, case, round_index, warmup)
                    for _ in range(worker_id, count, concurrency)]

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            batch = [item for group in pool.map(worker, range(concurrency)) for item in group]
        duration = (max(i['closed_ns'] for i in batch)-min(i['start_ns'] for i in batch))/1e9
        result = {'case': case, 'mode': mode, 'round': round_index, 'warmup': warmup,
                  'input_tokens': n, 'output_tokens': output, 'concurrency': concurrency,
                  'requests': len(batch), 'successes': sum(i['success'] for i in batch),
                  'duration_s': duration,
                  'valid_output_tokens': sum(len(i['token_ids']) for i in batch if i['success'])}
        report['blocks'].append(result)
        (root/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        assert all(i['success'] for i in batch), 'Do not accept timing for mismatched/failed output'

    try:
        for node in (32, 33):
            (root/f'node{node}-metrics-before.txt').write_text(rpc(node, '/metrics'))
        specifications = [(256, 64, 1, 6, 'short-c1'), (1024, 64, 1, 6, 'medium-c1'),
                          (1024, 64, 4, 8, 'medium-c4'), (4096, 128, 1, 6, 'long-c1')]
        if args.scope == 'c1':
            specifications = [s for s in specifications if s[2] == 1]
        report['load_scope'] = args.scope
        for n, output, _, _, _ in specifications:
            if (n, output) not in fixtures:
                fixture(n, output)
        if args.phase == 'baseline':
            for r in range(2):
                for n, output, concurrency, count, case in (specifications if r == 0 else reversed(specifications)):
                    block(n, output, concurrency, concurrency, 'naive', case, r, True)
                    block(n, output, concurrency, count, 'naive', case, r)
        else:
            # Equal input IDs, output IDs and load. Salts isolate all cache
            # identities, so neither side gets an accidental warm-P advantage.
            for n, output, concurrency, count, case in [s for s in specifications if s[0] == 1024]:
                for mode in ('naive', 'append_first'):
                    block(n, output, concurrency, concurrency, mode, case, -1, True)
                for r, mode in enumerate(('naive', 'append_first', 'append_first', 'naive')):
                    block(n, output, concurrency, max(concurrency, 4), mode, case, r)
        # This is outside all reported block windows; allow stats to publish.
        time.sleep(12)
        for node in (32, 33):
            (root/f'node{node}-metrics-after.txt').write_text(rpc(node, '/metrics'))
        summaries = []
        keys = sorted({(i['case'], i['mode']) for i in rows if not i['warmup']})
        for case, mode in keys:
            selected = [i for i in rows if i['case'] == case and i['mode'] == mode and not i['warmup']]
            blocks = [b for b in report['blocks'] if b['case'] == case and b['mode'] == mode and not b['warmup']]
            summary = {'case': case, 'mode': mode, 'requests': len(selected),
                       'success_rate': sum(i['success'] for i in selected)/len(selected),
                       'valid_output_tokens_per_second': sum(b['valid_output_tokens'] for b in blocks)/sum(b['duration_s'] for b in blocks),
                       'round_output_tokens_per_second': [b['valid_output_tokens']/b['duration_s'] for b in blocks]}
            for key in ('ttft_ms', 'ttf_visible_ms', 'tpot_ms', 'last_token_ms', 'first_gap_ms'):
                summary[key] = distribution([i[key] for i in selected if i.get(key) is not None])
            summary['itl_ms'] = distribution([v for i in selected for v in i.get('itl_ms', [])])
            summaries.append(summary)
        report['summaries'] = summaries
        report['all_outputs_match_native_prefix_reference'] = all(i['success'] for i in rows)
        report['external_KV_validation'] = 'PENDING post-run consumer INFO/SDK counter audit; do not promote this report alone as a no-recompute baseline.'
        report['passed'] = (report['all_outputs_match_native_prefix_reference']
                            and len(server_records) == len(rows)
                            and all(i['success'] for i in server_records))
    except Exception as exc:
        report['failure'] = type(exc).__name__+': '+str(exc)
    finally:
        server.shutdown()
        server.server_close()
        report['ended_host_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report['client_requests'] = len(rows)
        report['proxy_records'] = server_records
        (root/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert report['passed'], report.get('failure')


if __name__ == '__main__':
    main()
