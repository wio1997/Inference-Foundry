"""Framework-neutral timed scopes, with a version-pinned source adapter.

Only selected Python functions are enclosed in a context manager in memory.
No argument/result tensors are inspected, no device synchronize is inserted.
"""
import ast
import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time

ROOT = os.environ.get('FRAMEWORK_TIMING_ROOT', '')
_TLS = threading.local()
_FD = None
_FD_PID = None
_SEQ = 0
_ADAPTER = {}


def _emit(rows):
    global _FD, _FD_PID
    pid = os.getpid()
    if _FD_PID != pid:
        if _FD is not None:
            os.close(_FD)
        _FD = os.open(ROOT + '/spans-' + str(pid) + '.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        _FD_PID = pid
    payload = ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows).encode()
    while payload:
        n = os.write(_FD, payload)
        payload = payload[n:]


class Scope:
    def __init__(self, name, source, line, coarse):
        self.name, self.source, self.line, self.coarse = name, source, line, coarse
        self.on = False
        self.suppressed = False

    def __enter__(self):
        global _SEQ
        if getattr(_TLS, 'suppressed_depth', 0):
            _TLS.suppressed_depth += 1
            self.suppressed = True
            return self
        stack = getattr(_TLS, 'stack', None)
        if not stack:
            try:
                with open(ROOT + '/control.json') as f:
                    control = json.load(f)
            except (OSError, ValueError):
                return self
            _TLS.mode = control.get('mode', 'off')
            _TLS.label = control.get('label', '')
            _TLS.rows = []
            _TLS.stack = []
            stack = _TLS.stack
        if _TLS.mode == 'off':
            _TLS.suppressed_depth = 1
            self.suppressed = True
            return self
        if _TLS.mode != 'detailed' and not self.coarse:
            return self
        self.on = True
        _SEQ += 1
        self.id = _SEQ
        self.parent = stack[-1].id if stack else None
        self.root_id = stack[0].id if stack else self.id
        self.recorder = None
        # torch is already loaded by the framework; never import it in startup.
        if _TLS.mode == 'detailed' and 'torch' in sys.modules:
            self.recorder = sys.modules['torch'].autograd.profiler.record_function('FT::' + self.name)
            self.recorder.__enter__()
        self.mono = time.perf_counter_ns()
        self.wall = time.time_ns()
        self.cpu = time.thread_time_ns()
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.suppressed:
            _TLS.suppressed_depth -= 1
            return False
        if not self.on:
            return False
        cpu_end = time.thread_time_ns()
        end = time.perf_counter_ns()
        stack = _TLS.stack
        assert stack.pop() is self
        _TLS.rows.append({'schema': 1, 'name': self.name, 'source': self.source, 'line': self.line,
                          'pid': os.getpid(), 'tid': threading.get_native_id(), 'id': self.id,
                          'parent': self.parent, 'root': self.root_id, 'label': _TLS.label,
                          'mode': _TLS.mode, 'mono_ns': self.mono, 'wall_ns': self.wall,
                          'duration_ns': end - self.mono, 'thread_cpu_ns': cpu_end - self.cpu,
                          'exception': exc_type.__name__ if exc_type else None})
        if self.recorder is not None:
            self.recorder.__exit__(exc_type, exc_value, tb)
        if not stack:
            _emit(_TLS.rows)
            _TLS.rows = []
        return False


def _span(name, source, line, coarse):
    return Scope(name, source, line, coarse)


class Transformer(ast.NodeTransformer):
    def __init__(self, module, source, spec):
        self.module, self.source, self.spec = module, source, spec
        self.path = []
        self.changed = []

    def _timed_statement(self, node):
        self.generic_visit(node)
        if not self.spec.get('timed_calls') or not self.path:
            return node
        hits = [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)
                and ast.unparse(n.func) in self.spec['timed_calls']]
        if not hits:
            return node
        name = self.module + ':' + '.'.join(self.path) + '/' + hits[0] + '@' + str(node.lineno)
        call = ast.Call(func=ast.Name(id='_ft_span', ctx=ast.Load()),
                        args=[ast.Constant(name), ast.Constant(self.source), ast.Constant(node.lineno), ast.Constant(True)], keywords=[])
        wrapped = ast.With(items=[ast.withitem(context_expr=call)], body=[node])
        return ast.copy_location(wrapped, node)

    visit_Assign = _timed_statement
    visit_Expr = _timed_statement

    def visit_ClassDef(self, node):
        self.path.append(node.name)
        self.generic_visit(node)
        self.path.pop()
        return node

    def visit_FunctionDef(self, node):
        qual = '.'.join(self.path + [node.name])
        self.path.append(node.name)
        self.generic_visit(node)
        self.path.pop()
        selected = qual in self.spec.get('functions', [])
        selected |= bool(self.spec.get('metadata_builders') and any('MetadataBuilder' in p for p in self.path)
                         and node.name != '__init__')
        if not selected:
            return node
        coarse = qual in self.spec.get('coarse', [])
        name = self.module + ':' + qual
        call = ast.Call(func=ast.Name(id='_ft_span', ctx=ast.Load()),
                        args=[ast.Constant(name), ast.Constant(self.source), ast.Constant(node.lineno), ast.Constant(coarse)], keywords=[])
        # Preserve docstring and signature, decorators, return/exception behavior.
        offset = int(bool(node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                          and isinstance(node.body[0].value.value, str)))
        body = node.body[offset:]
        wrapped = ast.With(items=[ast.withitem(context_expr=call)], body=body)
        ast.copy_location(wrapped, body[0])
        node.body = node.body[:offset] + [wrapped]
        self.changed.append({'name': name, 'line': node.lineno, 'end_line': node.end_lineno, 'coarse': coarse})
        return node


class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path)
        spec = _ADAPTER[fullname]
        digest = hashlib.sha256(source).hexdigest()
        if digest != spec['sha256']:
            raise ImportError('Timing adapter source mismatch: ' + fullname)
        tree = ast.parse(source, filename=self.path)
        transform = Transformer(fullname, self.path, spec)
        tree = transform.visit(tree)
        stmt = ast.ImportFrom(module='ft_hook', names=[ast.alias(name='_span', asname='_ft_span')], level=0)
        pos = 0
        while pos < len(tree.body):
            node = tree.body[pos]
            if ((isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
                    or (isinstance(node, ast.ImportFrom) and node.module == '__future__')):
                pos += 1
            else:
                break
        tree.body.insert(pos, stmt)
        ast.fix_missing_locations(tree)
        os.makedirs(ROOT, exist_ok=True)
        _emit([{'schema': 1, 'kind': 'adapter', 'pid': os.getpid(), 'module': fullname,
                'path': self.path, 'sha256': digest, 'functions': transform.changed}])
        return compile(tree, self.path, 'exec', dont_inherit=True)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _ADAPTER:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError('No source loader for timing adapter: ' + fullname)
        spec.loader = Loader(fullname, spec.origin)
        return spec


def install():
    global _ADAPTER
    with open(os.environ['FRAMEWORK_TIMING_ADAPTER']) as f:
        _ADAPTER = json.load(f)['modules']
    sys.meta_path.insert(0, Finder())
