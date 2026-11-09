"""HTTP request/event transport for the normal original DeepInfer Engine."""
import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sys
import threading
import time
import uuid


@dataclass
class Request:
    token_ids: list
    max_tokens: int
    ignore_eos: bool
    id: str = field(default_factory=lambda: 'cmpl-'+uuid.uuid4().hex)
    events: queue.Queue = field(default_factory=queue.Queue)
    ttft_trace: object = None
    wire: dict = field(default_factory=dict)
    arrival_time: float = field(default_factory=time.perf_counter)
    directory: object = None
    cancelled: bool = False


class Frontend:
    def __init__(self, args):
        import deepinfer
        if not Path(deepinfer.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1]):
            raise RuntimeError('original Engine frontend source was shadowed')
        from deepinfer.hugging_face import get_hf_tokenizer
        from deepinfer.engine.tokenization_cache import TokenizationCache
        self.args = args
        self.tokenizer = get_hf_tokenizer(args.model_dir, trust_remote_code=False)
        self.eos = self.tokenizer.eos_token_id
        self.token_cache = TokenizationCache(self.tokenizer.encode, token_budget=args.tokenizer_cache_tokens)
        self.pending, self.deferred = queue.Queue(), deque()
        self.active_requests = {}
        self.failed = False
        self.peers = None
        self.lock = threading.Lock()
        self.counts = dict(requests=0, prompt_tokens=0, cached_tokens=0, output_tokens=0,
            failed_requests=0, tokenizer_cache_hits=0, tokenizer_cache_hit_tokens=0)

    def ready(self):
        from deepinfer.entrypoints.file_transport import read_ready, ready_health, source_identity
        if self.failed:
            return False
        try:
            if self.peers is None:
                self.peers = read_ready(self.args.control_dir, world_size=self.args.world_size, source=source_identity())
            return ready_health(self.args.control_dir, self.peers, dp_size=self.args.dp_size)
        except (OSError, ValueError, KeyError):
            return False

    def cancel(self, request):
        from deepinfer.entrypoints.file_transport import atomic_json
        with self.lock:
            request.cancelled = True
            if request.directory is not None:
                name = 'cancel-'+hashlib.sha256(request.id.encode()).hexdigest()+'.json'
                atomic_json(request.directory/name, dict(request_id=request.id))

    def wire_request(self, request, directory):
        from deepinfer.entrypoints.file_transport import atomic_json
        with self.lock:
            request.directory = directory
            if request.cancelled:
                name = 'cancel-'+hashlib.sha256(request.id.encode()).hexdigest()+'.json'
                atomic_json(directory/name, dict(request_id=request.id))
            return dict(request.wire, id=request.id, token_ids=request.token_ids,
                max_tokens=request.max_tokens, ignore_eos=request.ignore_eos, arrival_time=request.arrival_time)

    def retire_requests(self, requests):
        with self.lock:
            for request in requests:
                if self.active_requests.get(request.id) is request:
                    self.active_requests.pop(request.id)

    def fail_deployment(self, job_id, error):
        from deepinfer.entrypoints.file_transport import atomic_json
        with self.lock:
            self.failed = True
        atomic_json(self.args.control_dir/'fatal-frontend.json', dict(job_id=job_id, error=str(error)))
        waiting = list(self.deferred)
        self.deferred.clear()
        while True:
            try:
                waiting.append(self.pending.get_nowait())
            except queue.Empty:
                break
        for request in waiting:
            request.events.put(dict(type='error', message=str(error), terminal=True))
        self.retire_requests(waiting)

    def scheduler(self):
        # This only transports intake batches. The original Engine selects
        # requests, physical cache actions and model steps on every DP driver.
        from deepinfer.engine.continuous_frontend910c import serve
        from deepinfer.entrypoints.file_transport import PROTOCOL, atomic_json
        while not self.failed:
            first = self.deferred.popleft() if self.deferred else self.pending.get()
            batch = [first]
            deadline = time.monotonic()+self.args.batch_wait_ms/1000
            while len(batch) < 8:
                try:
                    batch.append(self.pending.get(timeout=max(0, deadline-time.monotonic())))
                except queue.Empty:
                    break
            job_id = uuid.uuid4().hex
            directory = self.args.control_dir/'jobs'/job_id
            serving = False
            try:
                if not self.ready():
                    raise RuntimeError('original Engine deployment is unavailable')
                if (self.args.control_dir/'current.json').exists():
                    raise RuntimeError('previous Engine transport job is still owned')
                directory.mkdir(parents=True)
                body = dict(protocol=PROTOCOL, eos_token_id=self.eos, continuous_admission=True,
                    request_timeout=self.args.request_timeout,
                    requests=[self.wire_request(request, directory) for request in batch])
                atomic_json(directory/'request.json', body)
                atomic_json(self.args.control_dir/'current.json', dict(id=job_id))
                serving = True
                serve(self, directory, batch, body)
                current = self.args.control_dir/'current.json'
                if json.loads(current.read_text()) != dict(id=job_id):
                    raise RuntimeError('Engine transport ownership changed')
                current.unlink()
            except Exception as exc:
                if not serving:
                    for request in batch:
                        request.events.put(dict(type='error', message=str(exc), terminal=True))
                self.retire_requests(batch)
                self.fail_deployment(job_id, exc)
                return


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deepinfer.entrypoints.engine_job import completion_request
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--control-dir', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8172)
    p.add_argument('--batch-wait-ms', type=int, default=5)
    p.add_argument('--tokenizer-cache-tokens', type=int, default=0)
    p.add_argument('--request-timeout', type=int, default=3600)
    p.add_argument('--world-size', type=int, default=16)
    p.add_argument('--dp-size', type=int, default=2)
    args = p.parse_args()
    frontend = Frontend(args)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def send_json(self, value, status=200):
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == '/health':
                available = frontend.ready()
                self.send_json(dict(status='ok' if available else 'unavailable', backend='DeepInfer-original'), 200 if available else 503)
            elif self.path == '/v1/models':
                self.send_json(dict(object='list', data=[dict(id='dsv41', object='model', owned_by='DeepInfer')]))
            elif self.path == '/metrics':
                with frontend.lock:
                    text = ''.join(f'deepinfer_{key}_total {value}\n' for key,value in frontend.counts.items()).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', str(len(text)))
                self.end_headers()
                self.wfile.write(text)
            else:
                self.send_json(dict(error='unknown endpoint'), 404)

        def do_POST(self):
            arrived = time.perf_counter()
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32*1024*1024:
                    raise ValueError('invalid request body size')
                body = json.loads(self.rfile.read(length))
                if self.path == '/v1/abort':
                    with frontend.lock:
                        request = frontend.active_requests.get(body['request_id'])
                    if request is None:
                        self.send_json(dict(error='unknown active request'), 404)
                    else:
                        frontend.cancel(request)
                        self.send_json(dict(request_id=request.id, cancellation_requested=True))
                    return
                if self.path != '/v1/completions':
                    self.send_json(dict(error='use /v1/completions'), 404)
                    return
                if not frontend.ready():
                    self.send_json(dict(error='Engine unavailable'), 503)
                    return
                if body.get('model', 'dsv41') != 'dsv41' or body.get('n', 1) != 1:
                    raise ValueError('model=dsv41 and n=1 are required')
                token_hit = False
                def tokenize(text):
                    nonlocal token_hit
                    ids, token_hit = frontend.token_cache.get(text)
                    return ids
                wire = completion_request(body, tokenize, max_model_len=frontend.peers[0]['max_model_len'])
                mm = (wire.get('extra') or {}).get('multi_modal_data', wire.get('multi_modal_data'))
                if mm and mm.get('image_bytes') and not frontend.peers[0]['capabilities']['vision']:
                    raise ValueError('this deployment does not accept image inputs')
                request = Request(wire['token_ids'], wire['max_tokens'], wire['ignore_eos'],
                    id=body.get('request_id') or 'cmpl-'+uuid.uuid4().hex, wire=wire, arrival_time=arrived)
                if not isinstance(request.id, str) or not request.id:
                    raise ValueError('request_id must be a nonempty string')
                with frontend.lock:
                    if frontend.failed:
                        self.send_json(dict(error='Engine unavailable'), 503)
                        return
                    if request.id in frontend.active_requests:
                        raise ValueError('request_id is already active')
                    frontend.active_requests[request.id] = request
                    if token_hit:
                        frontend.counts['tokenizer_cache_hits'] += 1
                        frontend.counts['tokenizer_cache_hit_tokens'] += len(request.token_ids)
                    frontend.pending.put(request)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(dict(error=str(exc)), 400)
                return
            streaming = bool(body.get('stream', False))
            created = int(time.time())
            generated, raw_scores, pending_scores, published = [], [], [], ''
            hit, prompt_tokens = 0, len(request.token_ids)
            wants_logprobs = bool(body.get('logprobs') or wire['sampling_params']['top_logprobs'] or wire['sampling_params']['prompt_logprobs'])
            def scores(rows):
                if not wants_logprobs:
                    return None
                return dict(tokens=[frontend.tokenizer.decode([row['token_id']], skip_special_tokens=False) for row in rows],
                    token_ids=[row['token_id'] for row in rows], token_logprobs=[row['token_logprobs'] for row in rows],
                    top_logprobs=[{frontend.tokenizer.decode([token], skip_special_tokens=False):prob for token,prob in zip(row['top_token_ids'],row['top_logprobs'])} for row in rows],
                    top_token_ids=[row['top_token_ids'] for row in rows],
                    prompt_logprobs=[row['prompt_logprobs'] for row in rows if row['prompt_logprobs']])
            def send_chunk(text='', finish=None, usage=None, rows=()):
                message = dict(id=request.id, object='text_completion', created=created, model='dsv41',
                    choices=[dict(index=0, text=text, logprobs=scores(rows), finish_reason=finish)])
                if usage is not None:
                    message['usage'] = usage
                self.wfile.write(('data: '+json.dumps(message, ensure_ascii=False, allow_nan=False)+'\n\n').encode())
                self.wfile.flush()
            try:
                if streaming:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                while True:
                    event = request.events.get(timeout=args.request_timeout)
                    if event['type'] == 'error':
                        if streaming:
                            self.wfile.write(('data: '+json.dumps(dict(error=event['message']))+'\n\n').encode())
                            self.wfile.flush()
                        else:
                            self.send_json(dict(error=event['message']), 400)
                        return
                    if event['type'] == 'start':
                        hit, prompt_tokens = event['cached_tokens'], event['prompt_tokens']
                    elif event['type'] == 'token':
                        generated.append(event['token_id'])
                        raw_scores.append(event)
                        pending_scores.append(event)
                        decoded = frontend.tokenizer.decode(generated, skip_special_tokens=False)
                        if not decoded.endswith('\ufffd') and decoded.startswith(published):
                            delta, published = decoded[len(published):], decoded
                            if streaming:
                                send_chunk(delta, rows=pending_scores)
                                pending_scores.clear()
                    elif event['type'] == 'done':
                        decoded = frontend.tokenizer.decode(generated, skip_special_tokens=False)
                        usage = dict(prompt_tokens=prompt_tokens, completion_tokens=len(generated),
                            total_tokens=prompt_tokens+len(generated), prompt_tokens_details=dict(cached_tokens=hit))
                        if streaming:
                            send_chunk(decoded[len(published):] if decoded.startswith(published) else '', event['finish_reason'], usage, rows=pending_scores)
                            self.wfile.write(b'data: [DONE]\n\n')
                            self.wfile.flush()
                        else:
                            self.send_json(dict(id=request.id, object='text_completion', created=created, model='dsv41',
                                choices=[dict(index=0,text=decoded,logprobs=scores(raw_scores),finish_reason=event['finish_reason'])],usage=usage))
                        audit = dict(id=request.id,input_tokens=prompt_tokens,cached_tokens=hit,
                            generated_token_ids=generated,text=decoded,finish_reason=event['finish_reason'],
                            sampling_params=wire['sampling_params'],tokenizer_cache_hit=token_hit)
                        with frontend.lock:
                            with (args.control_dir/'api-results.jsonl').open('a') as output:
                                output.write(json.dumps(audit, ensure_ascii=False, allow_nan=False)+'\n')
                        return
            except (BrokenPipeError, ConnectionResetError, queue.Empty):
                frontend.cancel(request)
                return

        def log_message(self, fmt, *values):
            print(fmt % values, flush=True)

    threading.Thread(target=frontend.scheduler, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps(dict(stage='http_listening', host=args.host, port=args.port, backend='DeepInfer original')), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
