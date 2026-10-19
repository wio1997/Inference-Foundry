#!/usr/bin/env python3
"""Exact cached source control flow; explicit tensor/network/free boundaries.
No Torch/NPU import, transport, inference, physical KV allocation or timing.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import itertools
import json
import pathlib
from types import SimpleNamespace as NS

R = pathlib.Path(__file__).resolve().parents[1]
O = R / 'runs/L027-R003-release-policy'
O.mkdir(parents=True, exist_ok=False)
protocol = json.loads((R / 'specs/L027-release-policy.json').read_text())
bodies, cases, trace = [], [], []
checks = 0
for name, spec in protocol['canonical_sources'].items():
    data = (R / 'sources/installed' / name).read_bytes()
    original = R / spec['prior_source_receipt']
    assert hashlib.sha256(original.read_bytes()).hexdigest() == spec['prior_source_receipt_sha256']
    prior = json.loads(original.read_text())
    assert prior['source_commit'] == protocol['source_commit']
    item = prior['files'][name]
    assert item['matches_git_head'] and item['content'].encode() == data
    assert hashlib.sha256(data).hexdigest() == item['sha256'] == spec['sha256']

def check(value, label):
    global checks
    checks += 1
    assert value, label

g = {'time': NS(monotonic=lambda: now), 'logger': NS(
    warning=lambda msg: trace.append('warning:timeout'),
    exception=lambda msg: trace.append('exception:release'))}

def bind(path, owner, names, target):
    p = R / 'sources/installed/python/sglang/srt/disaggregation' / path
    text = p.read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == owner)
    for name in names:
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        source = ast.get_source_segment(text, node)
        bodies.append({'path': str(p.relative_to(R)), 'owner': owner, 'name': name,
                       'line': node.lineno, 'end_line': node.end_lineno,
                       'body_sha256': hashlib.sha256(source.encode()).hexdigest(), 'complete_body': True})
        node = copy.deepcopy(node)
        node.decorator_list = []
        m = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        ast.fix_missing_locations(m)
        exec(compile(m, str(p), 'exec'), g)
        setattr(target, name, g[name])

class Manager:
    def __init__(self):
        self._deferred_abort_ack_tracker = {}
        self._staging_outstanding = {}
        self._deferred_ack_targets = {}
    def _send_abort_ack(self, ip, port, room): trace.append(('ack', room))

bind('common/conn.py', 'CommonKVManager', [
    'register_deferred_abort_room', 'note_abort_ack', 'is_abort_release_safe',
    'clear_deferred_abort_state', '_maybe_ack_drained_abort', 'register_deferred_ack_target'], Manager)

manager = Manager()
manager.note_abort_ack(11, 1)
check(not manager._deferred_abort_ack_tracker, 'ignore unarmed ack')
manager.register_deferred_abort_room(11)
manager.note_abort_ack(11, 1)
manager.note_abort_ack(11, 1)
check(not manager.is_abort_release_safe(11, 2), 'duplicate ack deduplicated')
manager.note_abort_ack(11, 2)
check(manager.is_abort_release_safe(11, 2), 'two unique acks meet captured cardinality')
manager.register_deferred_abort_room(11)
check(not manager.is_abort_release_safe(11, 2), 'rearmed room clears stale set')
manager.clear_deferred_abort_state(11)
check(11 not in manager._deferred_abort_ack_tracker, 'clear tracker')
cases.append({'family': 'ACK_TRACKER', 'deduplicates': True, 'rearm_clears': True,
              'rank_authentication_and_room_generation_not_tested': True})

for outstanding in [0, 1, 2]:
    trace.clear()
    manager = Manager()
    manager._staging_outstanding[11] = outstanding
    manager.register_deferred_ack_target(11, 'fixture-ip', 1)
    manager._maybe_ack_drained_abort(11)
    check(trace == ([('ack', 11)] if outstanding == 0 else []), 'drain ack withheld while outstanding')
    manager._staging_outstanding[11] = 0
    manager._maybe_ack_drained_abort(11)
    manager._maybe_ack_drained_abort(11)
    check(trace == [('ack', 11)], 'ack target popped exactly once')
    cases.append({'family': 'ACK_DRAIN', 'initial_outstanding': outstanding, 'ack_count': len(trace)})

class TensorFixture:
    def __init__(self, values): self.values = [int(v) for v in values]
    def numel(self): return len(self.values)
    def tolist(self): return list(self.values)

def reduce_min(tensor, **kwargs):
    trace.append('all_reduce_MIN_fixture')
    tensor.values = [min(a, b) for a, b in zip(tensor.values, remote_ready)]

g['torch'] = NS(tensor=lambda values, dtype: TensorFixture(values), int32='int32-fixture',
               distributed=NS(get_world_size=lambda group: world_size,
                              all_reduce=reduce_min, ReduceOp=NS(MIN='MIN-fixture')))
g['release_kv_cache'] = lambda req, tree, is_insert: trace.append(('free-device', req.bootstrap_room))
g['discard_kv_cache_backup'] = lambda req, tree, kind: trace.append(('free-host-backup', req.bootstrap_room))

class Queue:
    def __init__(self, host, staging=False):
        self.enable_host_receive = host
        self.enable_staging = staging
        self.tree_cache = object()
        self.gloo_group = object()
        self.deferred_kv_release_timeout = 30.0
        self._deferred_releases = []
        self.metadata_buffers = NS(bootstrap_room={})
        self.req_to_metadata_buffer_idx_allocator = NS(free=lambda idx: trace.append(('free-metadata', idx)))
        self.staging_handler = NS(is_staging_room=lambda room: True,
                                  unregister_decode_req=lambda room: trace.append(('unregister', room)))

bind('decode.py', 'DecodeTransferQueue', ['_release_request', '_defer_release', '_do_release',
                                        'has_pending_deferred_releases', 'resolve_deferred_releases'], Queue)

def request(room, ack_count, host_staged):
    mgr = Manager()
    mgr.register_deferred_abort_room(room)
    for rank in range(ack_count): mgr.note_abort_ack(room, rank)
    receiver = NS(kv_mgr=mgr, bootstrap_infos=['sender0', 'sender1'],
                  clear=lambda: trace.append(('receiver-clear', room)))
    return NS(req=NS(bootstrap_room=room), kv_receiver=receiver,
              metadata_buffer_index=room, host_staged=host_staged)

world_size, remote_ready = 1, []
for host_staged, ack_count, moment in itertools.product([False, True], [0, 1, 2], [29.9, 30.0, 31.0]):
    trace.clear()
    now = 0.0
    q = Queue(host=host_staged, staging=True)
    req = request(11, ack_count, host_staged)
    q.metadata_buffers.bootstrap_room[11] = 11
    q._defer_release(req)
    receiver = req.kv_receiver
    receiver.bootstrap_infos.clear()
    check(q._deferred_releases[0][3] == 2, 'required ack count snapshots before receiver mutation')
    now = moment
    q.resolve_deferred_releases()
    should_release = ack_count == 2 or (not host_staged and moment >= 30.0)
    check((req.kv_receiver is None) == should_release, 'release policy by host/drain/deadline')
    check(q.has_pending_deferred_releases() == (not should_release), 'pending ownership ledger')
    check(('warning:timeout' in trace) == (should_release and ack_count < 2), 'warn for undrained device release')
    check(q.metadata_buffers.bootstrap_room[11] == (0 if should_release else 11), 'metadata ownership')
    expected = ['warning:timeout'] if should_release and ack_count < 2 else []
    if should_release:
        expected += [('unregister', 11), ('free-host-backup' if host_staged else 'free-device', 11),
                     ('free-metadata', 11), ('receiver-clear', 11)]
        check(11 not in receiver.kv_mgr._deferred_abort_ack_tracker, 'drain accounting cleared at release')
    check(trace == expected, 'actual do_release call ordering with explicit physical boundaries')
    cases.append({'family': 'DEFER_RESOLVE', 'host_staged': host_staged, 'acks': ack_count,
                  'captured_required_acks': 2, 'now': moment, 'deadline': 30.0,
                  'released': should_release, 'release_without_all_acks': should_release and ack_count < 2,
                  'trace': list(trace)})

world_size = 2
for remote in [0, 1]:
    trace.clear()
    remote_ready = [remote]
    now = 0.0
    q = Queue(host=True)
    req = request(12, 2, True)
    q.metadata_buffers.bootstrap_room[12] = 12
    q._defer_release(req)
    now = 100.0
    q.resolve_deferred_releases()
    check((req.kv_receiver is None) == bool(remote), 'host MIN readiness requires remote rank')
    check('all_reduce_MIN_fixture' in trace, 'host readiness reduction reached')
    cases.append({'family': 'HOST_MIN_REDUCTION', 'remote_ready_fixture': remote,
                  'local_ready': True, 'released': bool(remote), 'collective_is_fixture': True})

world_size, remote_ready = 1, []
trace.clear()
now = 0.0
q = Queue(host=False)
first, second = request(21, 2, False), request(22, 2, False)
for req in [first, second]:
    q.metadata_buffers.bootstrap_room[req.metadata_buffer_index] = req.req.bootstrap_room
    q._defer_release(req)
original = q._do_release
def fail_first(req, idx):
    check(not q._deferred_releases, 'survivors committed before release callbacks')
    if idx == 21: raise RuntimeError('explicit injected physical release boundary failure')
    return original(req, idx)
q._do_release = fail_first
q.resolve_deferred_releases()
check(first.kv_receiver is not None and second.kv_receiver is None, 'one release failure does not block second')
check(not q._deferred_releases, 'failed release is not automatically retried')
check('exception:release' in trace, 'failed release logged')
cases.append({'family': 'RELEASE_EXCEPTION', 'failed_entry_still_has_receiver': True,
              'failed_entry_in_retry_ledger': False, 'second_released': True,
              'physical_cleanup_recovery_not_proven': True})

result = {'status': 'PASS', 'semantic_checks': checks, 'case_count': len(cases),
          'cases': cases, 'complete_source_bodies': bodies,
          'canonical_sources': protocol['canonical_sources'], 'source_commit': protocol['source_commit'],
          'native_executed': False, 'whole_modules_imported': False, 'torch_imported': False,
          'candidate_accepted': False, 'accepted_baseline': None, 'performance': 'NOT_MEASURED',
          'limitations': ['Stream/transport/free and collective boundaries are explicit fixtures.',
                          'No physical KV allocation, remote writer, SDMA, NPU inference or runtime flag proof.',
                          'Device deadline release is source policy, not evidence that an active use-after-free occurs.',
                          'Room generation/authentication and full sender/receiver inheritance not executed.',
                          'Release exception demonstrates ledger policy, not complete cleanup recovery.']}
(O / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['status', 'semantic_checks', 'case_count', 'native_executed']}))
