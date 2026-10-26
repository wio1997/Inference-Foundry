"""Independent in-memory transport checks; no HTTP server or NPU."""
import ast
import asyncio
import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
class Timeout:
    def __init__(self, **kwargs): pass
sys.modules['aiohttp'] = types.SimpleNamespace(ClientTimeout=Timeout)
spec = importlib.util.spec_from_file_location('review_client', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tree = ast.parse(Path('scripts/bench.py').read_text())
node = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'body' for t in n.targets))
reference = eval(compile(ast.fix_missing_locations(ast.Expression(node)), 'body', 'eval'), {'prompt': '测试', 'a': types.SimpleNamespace(max_tokens=1024)})
assert reference == m.body('测试', 1024)
good = [
    {'id': 'x', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]},
    {'id': 'x', 'choices': [{'index': 0, 'delta': {'reasoning': '测试'}, 'finish_reason': None}]},
    {'id': 'x', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]},
    {'id': 'x', 'choices': [], 'usage': {'completion_tokens': 1024, 'prompt_tokens': 32768, 'total_tokens': 33792}},
]
def wire(events):
    return b''.join(b'data: ' + json.dumps(e, ensure_ascii=False).encode() + b'\n\n' for e in events) + b'data: [DONE]\n\n'
cases = {}
def add(name, events, expected=1, count=1): cases[name] = ([wire(events)], expected, count)
add('valid', good, 0)
w = wire(good)
cases['fragmented_valid'] = ([w[:3], w[3:17], w[17:61], w[61:]], 0, 1)
add('missing_usage', good[:-1])
add('missing_finish', good[:2] + good[3:])
add('usage_not_final', good + [good[1]])
add('duplicate_usage', good + [good[-1]])
add('duplicate_request_id', good, 1, 2)
for name, change in (
    ('numeric_id', lambda e: e[1].update(id=42)),
    ('wrong_choice', lambda e: e[1]['choices'][0].update(index=1)),
    ('error_payload', lambda e: e[1].update(error={'message': 'bad'})),
    ('bad_usage_type', lambda e: e[-1].update(usage=[1])),
    ('bool_count', lambda e: e[-1]['usage'].update(completion_tokens=True)),
    ('wrong_count', lambda e: e[-1]['usage'].update(completion_tokens=1023)),
    ('error_finish', lambda e: e[2]['choices'][0].update(finish_reason='error')),
):
    changed = copy.deepcopy(good); change(changed); add(name, changed)
cases['trailing_bytes'] = ([w+b'x'], 1, 1)
cases['missing_done'] = ([w[:-14]], 1, 1)
cases['malformed_json'] = ([b'data: {bad}\n\ndata: [DONE]\n\n'], 1, 1)
class Response:
    status=200
    headers={}
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    @property
    def content(self): return self
    async def iter_any(self):
        for fragment in fragments: yield fragment
class Session:
    def __init__(self, **kwargs): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    def post(self, *args, **kwargs): return Response()
m.aiohttp.ClientSession = Session
results={}
for name, (fragments, expected, count) in cases.items():
    with tempfile.TemporaryDirectory() as td:
        t=Path(td); (t/'dataset').write_text('{"question":"test"}\n'*count)
        args=types.SimpleNamespace(dataset=str(t/'dataset'), offset=0, limit=count, out_dir=t/'out', concurrency=12, max_tokens=1024, url='fake://no-network', phase='measured')
        rc=0; exc=None
        try:
            with contextlib.redirect_stdout(io.StringIO()): asyncio.run(m.acquire(args))
        except SystemExit as e: rc=e.code
        except Exception as e: rc='uncaught'; exc=repr(e)
        summary=(t/'out/summary.json').exists()
        record_path=t/'out/request_000.json'
        record=json.loads(record_path.read_text()) if record_path.exists() else {}
        payloads=[base64.b64decode(x['payload_b64']) for x in record.get('events', [])]
        conservation=True
        if name in ('valid','fragmented_valid'):
            conservation = payloads == [json.dumps(e,ensure_ascii=False).encode() for e in good]+[b'[DONE]']
        elif record.get('row',{}).get('error'):
            saved=record.get('wire_fragments_b64_on_error', [])
            conservation=b''.join(base64.b64decode(x) for x in saved)==b''.join(fragments)
        results[name]={'expected_exit':expected,'actual_exit':rc,'summary_preserved':summary,'payload_conservation':conservation,'uncaught':exc,'error':record.get('row',{}).get('error')}
proof={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'CPU fake transport, no HTTP/server/NPU','body_ast_equal':True,'cases':results,'pass':all(x['actual_exit']==x['expected_exit'] and x['summary_preserved'] and x['payload_conservation'] for x in results.values())}
dest.write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
