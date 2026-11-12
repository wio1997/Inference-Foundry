# SPDX-License-Identifier: Apache-2.0
"""OpenAI completion SSE transport for the native DeepInfer engine.

HTTP threads only validate and queue requests. The rank-zero calling thread owns
all engine/model operations. This candidate admits bounded concurrent requests
and routes real native scheduler outputs by request identity; not yet validated.
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


def serve_sdk_batch_engine(llm, worker, args, out):
    import numpy as np
    from transformers import AutoTokenizer
    from deepinfer._C import engine
    from deepinfer.sampling_params import SamplingParams
    from deepinfer.model_executor.models.glm5_next.scheduler_eos import configure_scheduler_eos

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation = json.loads((Path(args.model) / 'generation_config.json').read_text())
    configure_scheduler_eos(llm.scheduler, generation,
        tokenizer_eos_id=tokenizer.eos_token_id,
        vocab_size=worker._glm_worker_runtime.vocab_size)
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
                                'max_active_requests': args.max_sequences,
                                'prefix_cache': dict(worker._glm_worker_runtime.adapter.controller.prefix_stats),
                                'mtp': dict(getattr(worker._glm_worker_runtime.adapter.controller, 'mtp_stats', {})),
                                'kda_decode_capture': (worker._glm_worker_runtime.kda_capture.snapshot()
                                    if getattr(worker._glm_worker_runtime, 'kda_capture', None) else None)})
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
                    all(type(i) is int and 0 <= i < worker._glm_worker_runtime.vocab_size
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
        'max_active_requests': args.max_sequences, 'http_pending_limit': 64,
        'continuous_batching': True, 'world_size': args.world_size}) + '\n')
    active = {}
    completed = 0
    completed_ids = set()
    runtime = worker._glm_worker_runtime

    def emit(state, text, finish=None, usage=None):
        event = {'id': state['rid'], 'object': 'text_completion',
            'created': state['created'], 'model': model_name,
            'choices': [] if usage is not None else [
                {'index': 0, 'text': text, 'logprobs': None, 'finish_reason': finish}]}
        if usage is not None:
            event['usage'] = usage
        if not state['request'].disconnected.is_set():
            state['request'].events.put(event)

    def admit(request):
        if request.disconnected.is_set():
            return
        body = request.body
        prompt = body['prompt']
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
        cap = body.get('max_tokens', args.max_new_tokens)
        if not prompt_ids or len(prompt_ids) + cap > args.max_context:
            raise ValueError('prompt plus generation exceeds context')
        rid = 'cmpl-' + uuid.uuid4().hex
        params = SamplingParams(temperature=0, max_tokens=cap,
            ignore_eos=body.get('ignore_eos', False), seq_seed=body.get('seed', 0) or 0)
        seq = llm.add_request('http', rid, params, np.asarray(prompt_ids, dtype=np.int32))
        if seq is None:
            raise RuntimeError('native scheduler rejected admitted HTTP request')
        active[rid] = {'rid': rid, 'request': request, 'created': int(time.time()),
            'ids': [], 'decoded': '', 'input_count': len(prompt_ids),
            'started': time.perf_counter_ns(), 'steps': 0}

    try:
        with (out / 'service-requests.jsonl').open('a') as log:
            while not stopped.is_set():
                # Only the rank-zero engine thread admits/steps requests.
                # HTTP threads own transport and never call model methods.
                while (len(active) < args.max_sequences
                        and llm.get_num_unfinished_requests() < args.max_sequences
                        and (not args.serve_max_requests
                             or completed + len(active) < args.serve_max_requests)):
                    try:
                        request = waiting.get(block=not active and not llm.has_unfinished_requests())
                    except queue.Empty:
                        break
                    try:
                        admit(request)
                    except Exception as error:
                        request.events.put({'error': {'message': str(error), 'type': type(error).__name__}})
                        request.events.put(None)
                        if runtime.adapter.poisoned:
                            raise
                    if not active:
                        continue
                if not active:
                    if llm.has_unfinished_requests():
                        # Native terminal queues are reaped on the next
                        # scheduler pass, after the HTTP terminal event.
                        for pointer in llm.step():
                            pointer = int(pointer)
                            rid = engine.RequestOutput.get_request_id(pointer)
                            if (rid not in completed_ids
                                    or int(engine.RequestOutput.get_token_id(pointer)) >= 0):
                                raise RuntimeError('unexpected output while reaping closed HTTP requests')
                        continue
                    if args.serve_max_requests and completed >= args.serve_max_requests:
                        break
                    continue
                for state in active.values():
                    state['steps'] += 1
                    if state['steps'] > args.max_context + args.max_new_tokens + 128:
                        raise RuntimeError('native HTTP request exceeded bounded scheduling steps')
                for pointer in llm.step():
                    pointer = int(pointer)
                    rid = engine.RequestOutput.get_request_id(pointer)
                    token = int(engine.RequestOutput.get_token_id(pointer))
                    if rid not in active:
                        if rid in completed_ids and token < 0:
                            continue
                        raise RuntimeError('output does not belong to an active HTTP request')
                    state = active[rid]
                    if token >= 0:
                        state['ids'].append(token)
                        text = tokenizer.decode(state['ids'], skip_special_tokens=True)
                        if not text.endswith('\ufffd'):
                            if not text.startswith(state['decoded']):
                                raise RuntimeError('tokenizer changed an emitted prefix')
                            if len(text) > len(state['decoded']):
                                emit(state, text[len(state['decoded']):])
                            state['decoded'] = text
                    if not engine.RequestOutput.get_finished(pointer):
                        continue
                    if rid in runtime.adapter.controller.requests:
                        raise RuntimeError('terminal request retained SDK allocator ownership')
                    final = tokenizer.decode(state['ids'], skip_special_tokens=True)
                    if not final.startswith(state['decoded']):
                        raise RuntimeError('unstable final prefix')
                    if final != state['decoded']:
                        emit(state, final[len(state['decoded']):])
                    reason = engine.RequestOutput.get_finish_reason(pointer)
                    emit(state, '', reason)
                    if state['request'].body.get('stream_options', {}).get('include_usage'):
                        emit(state, '', usage={'prompt_tokens': state['input_count'],
                            'completion_tokens': len(state['ids']),
                            'total_tokens': state['input_count'] + len(state['ids'])})
                    log.write(json.dumps({'request_id': rid, 'input_tokens': state['input_count'],
                        'generated_token_ids': state['ids'], 'finish_reason': reason,
                        'elapsed_ns': time.perf_counter_ns()-state['started'],
                        'driver_owner_released': True,
                        'disconnected': state['request'].disconnected.is_set()})+'\n')
                    log.flush()
                    state['request'].events.put(None)
                    del active[rid]
                    completed_ids.add(rid)
                    completed += 1
            (out / 'result.json').write_text(json.dumps({'status': 'executed',
                'scope': 'native SDK batch HTTP;quality/performance separately measured',
                'completed_requests': completed,
                'max_actual_batch': max(runtime.adapter.batch_sizes, default=0),
                'batch_sizes': runtime.adapter.batch_sizes,
                'driver_owner_released': not runtime.adapter.controller.requests,
                'prefix_cache': dict(runtime.adapter.controller.prefix_stats),
                'mtp': dict(getattr(runtime.adapter.controller, 'mtp_stats', {})),
                'mtp_output': dict(getattr(runtime, 'mtp_output_stats', {})),
                'async_execution': dict(getattr(runtime.adapter, 'async_stats', {})),
                'async_schedule': dict(getattr(runtime.adapter.controller, 'async_stats', {}))})+'\n')
    except Exception as error:
        for state in active.values():
            state['request'].events.put({'error': {'message': str(error), 'type': type(error).__name__}})
            state['request'].events.put(None)
        raise
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        http_thread.join(timeout=5)
