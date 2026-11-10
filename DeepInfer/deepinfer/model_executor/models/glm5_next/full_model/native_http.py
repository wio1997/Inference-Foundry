# SPDX-License-Identifier: Apache-2.0
"""OpenAI completion SSE transport for the native DeepInfer engine.

HTTP threads only validate and queue requests. The rank-zero calling thread owns
all engine/model operations. This initial transport serializes requests because
the current model bridge has one cache owner; it does not claim dynamic batching.
"""
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


@dataclass
class Pending:
    body: dict
    events: queue.Queue = field(default_factory=queue.Queue)
    disconnected: threading.Event = field(default_factory=threading.Event)


def serve_native_engine(llm, worker, args, out):
    import numpy as np
    from transformers import AutoTokenizer
    from deepinfer._C import engine
    from deepinfer.sampling_params import SamplingParams
    from deepinfer.model_executor.models.glm5_next.scheduler_eos import configure_scheduler_eos

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation = json.loads((Path(args.model) / 'generation_config.json').read_text())
    configure_scheduler_eos(llm.scheduler, generation,
        tokenizer_eos_id=tokenizer.eos_token_id,
        vocab_size=worker._glm_worker_runtime.backend.vocab_size)
    waiting = queue.Queue(maxsize=64)
    stopped = threading.Event()
    model_name = args.served_model_name

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/health':
                self.reply(200, {'status': 'ready', 'engine': 'deepinfer',
                                'max_active_requests': 1})
            elif self.path == '/v1/models':
                self.reply(200, {'object': 'list', 'data': [
                    {'id': model_name, 'object': 'model', 'owned_by': 'deepinfer'}]})
            else:
                self.reply(404, {'error': 'unknown endpoint'})

        def do_POST(self):
            if self.path != '/v1/completions':
                self.reply(404, {'error': 'only /v1/completions supported'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1048576:
                    raise ValueError('bounded Content-Length required')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('JSON object required')
                if body.get('model', model_name) != model_name:
                    raise ValueError('unknown model')
                if body.get('stream') is not True:
                    raise ValueError('stream=true required')
                if body.get('n', 1) != 1 or body.get('best_of', 1) != 1:
                    raise ValueError('n=best_of=1 required')
                if body.get('temperature', 0) != 0:
                    raise ValueError('this native bridge currently supports greedy temperature=0')
                for name, default in [('top_p', 1), ('presence_penalty', 0),
                                      ('frequency_penalty', 0), ('repetition_penalty', 1)]:
                    if body.get(name, default) != default:
                        raise ValueError(name + ' is unsupported by this greedy bridge')
                if body.get('stop') or body.get('logprobs') or body.get('echo'):
                    raise ValueError('stop/logprobs/echo unsupported')
                cap = body.get('max_tokens', args.max_new_tokens)
                if type(body.get('ignore_eos', False)) is not bool:
                    raise ValueError('ignore_eos must be boolean')
                if not isinstance(body.get('stream_options', {}), dict):
                    raise ValueError('stream_options must be an object')
                if type(cap) is not int or not 1 <= cap <= args.max_new_tokens:
                    raise ValueError('max_tokens outside configured bound')
                prompt = body.get('prompt')
                if not isinstance(prompt, str) and not (
                    isinstance(prompt, list) and prompt and
                    all(type(i) is int and 0 <= i < worker._glm_worker_runtime.backend.vocab_size
                        for i in prompt)):
                    raise ValueError('prompt must be text or a single token-id vector')
                request = Pending(body)
                waiting.put_nowait(request)
            except (ValueError, TypeError, queue.Full) as error:
                self.reply(400, {'error': str(error)})
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    event = request.events.get()
                    if event is None:
                        self.wfile.write(b'data: [DONE]\n\n')
                        self.wfile.flush()
                        break
                    data = json.dumps(event, ensure_ascii=False).encode()
                    self.wfile.write(b'data: ' + data + b'\n\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                request.disconnected.set()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    (out / 'service-ready.json').write_text(json.dumps({
        'host': args.host, 'port': args.port, 'model': model_name,
        'engine': 'DeepInfer LLMEngine + native scheduler/Worker',
        'max_active_requests': 1, 'http_pending_limit': 64,
        'continuous_batching': False, 'world_size': args.world_size}) + '\n')
    try:
        with (out / 'service-requests.jsonl').open('a') as log:
            completed = 0
            while not stopped.is_set():
                request = waiting.get()
                if request.disconnected.is_set():
                    continue
                body = request.body
                rid = 'cmpl-' + uuid.uuid4().hex
                created = int(time.time())
                ids, decoded, final_reason = [], '', None
                started = time.perf_counter_ns()
                def emit(text, finish=None, usage=None):
                    event = {'id': rid, 'object': 'text_completion', 'created': created,
                             'model': model_name,
                             'choices': [] if usage is not None else [
                                 {'index': 0, 'text': text, 'logprobs': None,
                                  'finish_reason': finish}]}
                    if usage is not None:
                        event['usage'] = usage
                    if not request.disconnected.is_set():
                        request.events.put(event)
                try:
                    prompt = body['prompt']
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
                    cap = body.get('max_tokens', args.max_new_tokens)
                    if not prompt_ids or len(prompt_ids) + cap > args.max_context:
                        raise ValueError('prompt plus generation exceeds context')
                    params = SamplingParams(temperature=0, max_tokens=cap,
                        ignore_eos=body.get('ignore_eos', False), seq_seed=body.get('seed', 0) or 0)
                    llm.add_request('http', rid, params, np.asarray(prompt_ids, dtype=np.int32))
                    for _ in range(args.max_context + cap + 64):
                        if not llm.has_unfinished_requests():
                            break
                        for ptr in llm.step():
                            ptr = int(ptr)
                            token = int(engine.RequestOutput.get_token_id(ptr))
                            if token >= 0:
                                ids.append(token)
                                text = tokenizer.decode(ids, skip_special_tokens=True)
                                # Delay incomplete UTF-8 until the tokenizer can emit a stable prefix.
                                if not text.endswith('\ufffd'):
                                    if not text.startswith(decoded):
                                        raise RuntimeError('tokenizer changed already emitted text')
                                    if len(text) > len(decoded):
                                        emit(text[len(decoded):])
                                    decoded = text
                            if engine.RequestOutput.get_finished(ptr):
                                final_reason = engine.RequestOutput.get_finish_reason(ptr)
                    else:
                        raise RuntimeError('native scheduler iteration bound exceeded')
                    if worker._glm_worker_runtime.backend._owner is not None:
                        raise RuntimeError('terminal request retained model cache owner')
                    if final_reason is None:
                        raise RuntimeError('native request missing terminal output')
                    remaining = tokenizer.decode(ids, skip_special_tokens=True)
                    if remaining != decoded:
                        if not remaining.startswith(decoded):
                            raise RuntimeError('unstable final decoded prefix')
                        emit(remaining[len(decoded):])
                    emit('', final_reason)
                    if body.get('stream_options', {}).get('include_usage'):
                        emit('', usage={'prompt_tokens': len(prompt_ids),
                             'completion_tokens': len(ids), 'total_tokens': len(prompt_ids) + len(ids)})
                    log.write(json.dumps({'request_id': rid, 'input_tokens': len(prompt_ids),
                        'generated_token_ids': ids, 'finish_reason': final_reason,
                        'elapsed_ns': time.perf_counter_ns()-started,
                        'driver_owner_released': True, 'disconnected': request.disconnected.is_set()}) + '\n')
                    log.flush()
                except Exception as error:
                    request.events.put({'error': {'message': str(error), 'type': type(error).__name__}})
                    # An error after engine mutation is fatal: do not serve later requests on unknown state.
                    if llm.has_unfinished_requests() or worker._glm_worker_runtime.backend._owner is not None:
                        raise
                finally:
                    request.events.put(None)
                completed += 1
                if args.serve_max_requests and completed >= args.serve_max_requests:
                    (out / 'result.json').write_text(json.dumps({
                        'status': 'executed', 'scope': 'bounded native HTTP integration',
                        'completed_requests': completed, 'driver_owner_released':
                            worker._glm_worker_runtime.backend._owner is None}) + '\n')
                    break
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        http_thread.join(timeout=5)
