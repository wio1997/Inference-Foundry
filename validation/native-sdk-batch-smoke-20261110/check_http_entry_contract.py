"""Exercise actual HTTP handler code without importing or simulating a model.

Only transport validation and SSE framing run; no socket, GPU, Torch, model
arithmetic, inference quality or performance is tested here.
"""
import ast
import dataclasses
import io
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler


source = Path(__file__).resolve().parents[2] / 'DeepInfer/deepinfer/model_executor/models/glm5_next/full_model/sdk_http.py'
tree = ast.parse(source.read_text())
pending = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Pending')
serve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'serve_sdk_batch_engine')
handler = next(n for n in serve.body if isinstance(n, ast.ClassDef) and n.name == 'Handler')


class TransportQueue:
    def put_nowait(self, request):
        request.events.put({'choices': [{'text': 'transport-only'}]})
        request.events.put(None)


scope = dict(json=json, queue=queue, threading=threading,
    dataclass=dataclasses.dataclass, field=dataclasses.field,
    BaseHTTPRequestHandler=BaseHTTPRequestHandler,
    worker=SimpleNamespace(_glm_worker_runtime=SimpleNamespace(vocab_size=154880)),
    model_name='glm53flash', waiting=TransportQueue(),
    args=SimpleNamespace(max_new_tokens=256, max_sequences=8))
exec(compile(ast.fix_missing_locations(ast.Module(body=[pending, handler], type_ignores=[])), str(source), 'exec'), scope)


def call(body):
    raw = json.dumps(body).encode()
    request = object.__new__(scope['Handler'])
    request.path = '/v1/completions'
    request.headers = {'Content-Length': str(len(raw))}
    request.rfile, request.wfile = io.BytesIO(raw), io.BytesIO()
    codes = []
    request.send_response = lambda code: codes.append(code)
    request.send_header = lambda *args: None
    request.end_headers = lambda: None
    request.do_POST()
    return codes, request.wfile.getvalue()


base = {'model': 'glm53flash', 'stream': True, 'max_tokens': 128, 'temperature': 0}
for prompt in ([1, 2, 3], 'plain text'):
    codes, output = call(dict(base, prompt=prompt))
    assert codes == [200] and b'transport-only' in output and output.endswith(b'data: [DONE]\n\n')
for extra in ({'prompt': [-1]}, {'prompt': [154880]}, {'prompt': [1], 'max_tokens': 0},
              {'prompt': [1], 'model': 'wrong-model'}):
    codes, output = call(base | extra)
    assert codes == [400] and 'error' in json.loads(output)
for path in source.parent.glob('sdk_*.py'):
    assert '_glm_worker_runtime.backend' not in path.read_text(), path
print(json.dumps({'passed': True, 'valid_requests': 2, 'invalid_requests': 4,
    'scope': 'actual HTTP input validation and SSE framing only;no model execution'}))
