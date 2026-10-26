#!/usr/bin/env python3
"""Complete canonical source method with real stdlib Future/thread semantics.
No actual transfer, NPU, graph, tensor, physical KV allocation or timing result.
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
O = R / 'runs/L027-R005-future-drain'
O.mkdir(parents=True, exist_ok=False)
protocol = json.loads((R / 'specs/L027-future-drain.json').read_text())
p = R / protocol['source_path']
assert hashlib.sha256(p.read_bytes()).hexdigest() == protocol['source_sha256']
text = p.read_text()
cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'MooncakeKVManager')
node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_await_transfer_futures')
body_sha = hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
m = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
ast.fix_missing_locations(m)
g = {'concurrent': concurrent}
exec(compile(m, str(p), 'exec'), g)
await_futures = g['_await_transfer_futures']
cases, checks = [], 0
def check(condition, label):
    global checks
    checks += 1
    assert condition, label
def finished(value=0, exception=None):
    f = concurrent.futures.Future()
    if exception is None: f.set_result(value)
    else: f.set_exception(exception)
    return f

pending = concurrent.futures.Future()
pending.set_running_or_notify_cancel()
try:
    ret = await_futures(NS(enable_deferred_decode_kv_release=False), [finished(5), pending])
    check(ret == 5 and pending.running(), 'legacy nonzero status returns while other future still running')
    cases.append({'name': 'STATUS_DEFERRED_OFF', 'returned': ret, 'running_future_at_return': True})
finally:
    if not pending.done(): pending.set_result(0)

cancel_attempted, done = threading.Event(), threading.Event()
class TrackedFuture(concurrent.futures.Future):
    def cancel(self):
        result = super().cancel()
        cancel_attempted.set()
        return result
pending = TrackedFuture()
pending.set_running_or_notify_cancel()
observed = {}
def worker():
    try: observed['returned'] = await_futures(NS(enable_deferred_decode_kv_release=True), [finished(5), pending])
    except BaseException as e: observed['exception'] = type(e).__name__
    finally: done.set()
t = threading.Thread(target=worker, daemon=True)
t.start()
try:
    check(cancel_attempted.wait(3.0), 'source reached cancel of running future')
    check(not done.is_set() and pending.running(), 'deferred nonzero status remains blocked before running future completes')
    pending.set_result(0)
    check(done.wait(3.0), 'deferred nonzero status completes after explicit future completion')
    check(observed == {'returned': 5}, 'deferred drain preserves first nonzero status')
    cases.append({'name': 'STATUS_DEFERRED_ON', 'source_wait_observed_after_failed_cancel': True,
                  'returned_only_after_running_future_completed': True, 'returned': observed['returned']})
finally:
    if not pending.done(): pending.set_result(0)
    t.join(3.0)
    check(not t.is_alive(), 'owned CPU test thread joined')

for enabled in [False, True]:
    pending = concurrent.futures.Future()
    pending.set_running_or_notify_cancel()
    try:
        try:
            await_futures(NS(enable_deferred_decode_kv_release=enabled),
                          [finished(exception=RuntimeError('explicit transfer boundary failure')), pending])
        except RuntimeError as exc:
            check(str(exc) == 'explicit transfer boundary failure', 'original future exception escapes')
        else: raise AssertionError('expected exception')
        check(pending.running(), 'exception escaped while other future remains running')
        cases.append({'name': 'EXCEPTION_DEFERRED_ON' if enabled else 'EXCEPTION_DEFERRED_OFF',
                      'exception': 'RuntimeError', 'running_future_at_exception': True,
                      'counterexample_to_unconditional_drain_claim': bool(enabled)})
    finally:
        if not pending.done(): pending.set_result(0)

cancelled = concurrent.futures.Future()
cancelled.cancel()
check(cancelled.set_running_or_notify_cancel() is False, 'cancel notification delivered to stdlib waiters')
ret = await_futures(NS(enable_deferred_decode_kv_release=True), [cancelled, finished(0)])
check(ret == 0, 'cancelled future ignored')
cases.append({'name': 'CANCELLED', 'returned': ret})
ret = await_futures(NS(enable_deferred_decode_kv_release=True), [finished(5), finished(7), finished(0)])
check(ret in [5, 7], 'first completed nonzero status remains valid despite iteration order')
cases.append({'name': 'MULTIPLE_STATUS_ERRORS', 'returned': ret, 'first_error_order_not_assumed': True})

result = {'status': 'PASS', 'semantic_checks': checks, 'case_count': len(cases), 'cases': cases,
          'source_commit': protocol['source_commit'], 'source_path': protocol['source_path'],
          'source_sha256': protocol['source_sha256'], 'complete_body_sha256': body_sha,
          'source_line': node.lineno, 'source_end_line': node.end_lineno,
          'real_stdlib_future_and_thread_executed': True, 'physical_transfer_executed': False,
          'native_executed': False, 'candidate_accepted': False, 'accepted_baseline': None,
          'performance': 'NOT_MEASURED',
          'limitations': ['RuntimeError is an explicit transfer-function boundary failure, not observed transport failure.',
                          'No proof of active device write or use-after-free; parent worker/SDK failure and cleanup paths require audit.',
                          'Event waits are test watchdogs, not performance measurements.']}
(O / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['status', 'semantic_checks', 'case_count', 'native_executed']}))
