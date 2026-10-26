#!/usr/bin/env python3
"""CPU fake-Event checks for bounded prior-generation query bookkeeping."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

root = Path('/data/wio/Inference_Foundry')
spec = importlib.util.spec_from_file_location(
    'run659_fake', root / 'scripts/test_ready_edge_observer_run659.py')
fake = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fake
spec.loader.exec_module(fake)
for row in fake.result['typed_rows']:
    if row['cycle'] == 64 and row['role'] == 'target_forward.target_input_ids':
        row['python_version'] -= 1
path = root / 'runtime/ready_query_observer_run661.py'
spec = importlib.util.spec_from_file_location('runtime.ready_query_observer', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Q = module.ReadyQueryRecorder

def from_complete_fake():
    q = Q((63, 64, 65), current_stream=fake.current)
    q.rows = fake.r.rows
    q.streams = fake.r.streams
    q.stream_ids = fake.r.stream_ids
    q.typed_rows = [dict(row) for row in fake.result['typed_rows']]
    q.waits = fake.r.waits
    return q

q = from_complete_fake()
for cycle in (64, 65):
    for point in Q.POINTS:
        q.poll_prior(cycle, point)
packet = q.export_after_existing_drain(fake.identity)
assert len(packet['query_rows']) == 12
assert all(row['completed'] for row in packet['query_rows'])
negative = 0
for action in (
    lambda: q.poll_prior(64, 'cycle_begin'),
    lambda: from_complete_fake().export_after_existing_drain(fake.identity),
):
    try:
        action()
    except RuntimeError:
        negative += 1
    else:
        raise AssertionError('invalid query packet accepted')
q.query_rows[0]['completed'] = True
q.query_rows[2]['completed'] = False
try:
    q.export_after_existing_drain(fake.identity)
except RuntimeError:
    negative += 1
else:
    raise AssertionError('completion regression accepted')
q2 = from_complete_fake()
for cycle in (64, 65):
    for point in Q.POINTS:
        q2.poll_prior(cycle, point)
next(row for row in q2.typed_rows if row['cycle'] == 64 and
     row['role'] == 'target_forward.target_input_ids')['data_ptr'] += 1
try:
    q2.export_after_existing_drain(fake.identity)
except RuntimeError:
    negative += 1
else:
    raise AssertionError('changed target input address accepted')

class NoVersionTensor(fake.Tensor):
    @property
    def _version(self):
        raise RuntimeError('Inference tensors do not track version counter')
    @_version.setter
    def _version(self, value):
        pass

fresh = Q((63, 64, 65), current_stream=fake.current)
fresh.mark(63, 'target_before')
fresh.observe(63, 'inference_tensor', NoVersionTensor(900, (96,)),
              after_label='target_before')
assert fresh.typed_rows[0]['tensor_version'] is None
print({'status': 'pass', 'query_rows': 12, 'negative_cases': negative})
