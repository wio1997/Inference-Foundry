#!/usr/bin/env python3
"""Exact Ascend sender -> common transfer -> engine wrapper; SDK is a fixture.
Real stdlib threads/Futures; integer addresses never refer to real memory.
"""
from __future__ import annotations
import ast
import concurrent.futures
import copy
import hashlib
import json
import pathlib
import threading
from types import SimpleNamespace as NS

R = pathlib.Path(__file__).resolve().parents[1]
O = R / 'runs/L027-R006-ascend-drain'
O.mkdir(parents=True, exist_ok=False)
protocol = json.loads((R / 'specs/L027-ascend-drain.json').read_text())
for name, h in protocol['source_hashes'].items():
    assert hashlib.sha256((R / 'sources/installed' / name).read_bytes()).hexdigest() == h
g = {'concurrent': concurrent, 'logger': NS(debug=lambda *a, **kw: None),
     'group_concurrent_contiguous': lambda source, dest: ([[0]], [[0]])}
bodies, cases = [], []
checks = 0
def check(value, label):
    global checks
    checks += 1
    assert value, label
def bind(name, owner, method, target):
    p = R / 'sources/installed' / name
    text = p.read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    bodies.append({'path': name, 'owner': owner, 'name': method, 'line': node.lineno,
                   'end_line': node.end_lineno, 'body_sha256': hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()})
    m = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), copy.deepcopy(node)], type_ignores=[])
    ast.fix_missing_locations(m)
    exec(compile(m, str(p), 'exec'), g)
    setattr(target, method, g[method])

class Sender: pass
class Engine: pass
bind('python/sglang/srt/disaggregation/ascend/conn.py', 'AscendKVManager', 'send_kvcache', Sender)
bind('python/sglang/srt/disaggregation/mooncake/conn.py', 'MooncakeKVManager', '_transfer_data', Sender)
bind('python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py', 'MooncakeTransferEngine', 'batch_transfer_sync', Engine)

class Executor(concurrent.futures.ThreadPoolExecutor):
    def __init__(self): super().__init__(max_workers=2); self.owned_futures = []
    def submit(self, *a, **kw):
        f = super().submit(*a, **kw)
        self.owned_futures.append(f)
        return f

for enabled in [False, True]:
    for mode in ['STATUS', 'SDK_EXCEPTION', 'COMBINED_SUCCESS']:
        second_started, allow_second = threading.Event(), threading.Event()
        sdk_calls = []
        class SDK:
            def batch_transfer_sync_write(self, session, source, dest, lengths):
                sdk_calls.append((list(source), list(dest), list(lengths)))
                if len(source) == 2: return 0
                if source == [200]:
                    second_started.set()
                    if not allow_second.wait(3.0): raise AssertionError('owned SDK fixture gate timed out')
                    return 0
                if not second_started.wait(3.0): raise AssertionError('second source job never started')
                if mode == 'SDK_EXCEPTION': raise RuntimeError('injected SDK transfer error')
                return -7
        engine = Engine()
        engine.engine = SDK()
        sender = Sender()
        sender.engine = engine
        sender.pp_size = 1
        sender.is_hybrid_mla_backend = False
        sender.is_mla_backend = True
        sender.enable_custom_mem_pool = mode != 'COMBINED_SUCCESS'
        sender.enable_deferred_decode_kv_release = enabled
        sender.kv_args = NS(kv_data_ptrs=[100, 200], kv_item_lens=[8, 8])
        sender._validate_envelope_kv_layout = lambda *a: None
        sender._await_transfer_futures = lambda *a: (_ for _ in ()).throw(AssertionError('unexpected generic drain helper'))
        executor = Executor()
        try:
            ret = sender.send_kvcache('fixture-session', [0], [1000, 2000], [0], executor)
            if mode == 'COMBINED_SUCCESS':
                check(ret == 0 and not executor.owned_futures, 'noncustom path batches layers without futures')
                check(sdk_calls == [([100, 200], [1000, 2000], [8, 8])], 'combined address geometry')
                running = False
            else:
                check(ret == (-1 if mode == 'SDK_EXCEPTION' else -7), 'engine wrapper translates exception/status')
                check(len(executor.owned_futures) == 2, 'actual source submitted both layers')
                check(second_started.is_set() and not allow_second.is_set(), 'second SDK boundary still blocked at return')
                running = executor.owned_futures[1].running()
                check(running, 'Ascend override returned with second Future running')
                check(sorted(sdk_calls) == [([100], [1000], [8]), ([200], [2000], [8])], 'per-layer address geometry')
            cases.append({'deferred_flag': enabled, 'mode': mode, 'custom_pool': sender.enable_custom_mem_pool,
                          'returned': ret, 'running_future_at_return': running,
                          'generic_drain_helper_called': False, 'SDK_boundary_is_fixture': True})
        finally:
            allow_second.set()
            executor.shutdown(wait=True)
            check(all(f.done() for f in executor.owned_futures), 'all owned CPU jobs drained during test cleanup')

result = {'status': 'PASS', 'semantic_checks': checks, 'case_count': len(cases), 'cases': cases,
          'source_commit': protocol['source_commit'], 'source_hashes': protocol['source_hashes'],
          'complete_source_bodies': bodies, 'real_stdlib_executor_and_futures': True,
          'SDK_and_layout_validation_boundaries_are_fixtures': True,
          'native_executed': False, 'physical_transfer_executed': False,
          'candidate_accepted': False, 'accepted_baseline': None, 'performance': 'NOT_MEASURED',
          'limitations': ['No physical DMA or memory, actual custom-pool selection/flags, DCP/PP/hybrid-path execution.',
                          'The target source path structurally returns before a per-layer Future completes; active runtime UAF is not proven.',
                          'Fixture addresses and grouping represent a minimal one-entry transfer only.']}
(O / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['status', 'semantic_checks', 'case_count', 'native_executed']}))
