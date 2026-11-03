#!/usr/bin/env python3
"""Offline rank-local duration comparison; no IDs, clocks, or workloads joined across devices."""
import collections
import decimal
import hashlib
import json
from pathlib import Path
D = decimal.Decimal
P = Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl')
expected = '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(P.read_bytes()).hexdigest() == expected
out = collections.defaultdict(list)
for line in P.open():
    row = json.loads(line)
    if row.get('kind') != 'rank':
        continue
    data = row['data']
    ep = data['ep']
    if ep not in range(8):
        continue
    for layer in data['layers']:
        digest = layer['scheduler_digest']
        if layer['layer'] != 23 or digest[:8] not in ('7ddf7ae9', 'db852bff'):
            continue
        matching = [t for group in layer['operator_groups']
                    if group['operator_name'] == 'reduce_scatterAicpuKernel'
                    for t in group['tasks']]
        assert len(matching) == 1, (ep, digest[:8], len(matching))
        task = matching[0]
        assert task['csv']['Name'] == 'reduce_scatterAicpuKernel'
        assert task['csv']['Accelerator Core'] == 'AI_CPU'
        assert task['csv']['Input Shapes'] == 'N/A'
        assert task['csv']['Input Data Types'] == 'N/A'
        assert task['csv']['Stream ID'] == ('90' if ep == 0 else '92')
        assert str(task['device_track_tid']) == task['csv']['Stream ID']
        assert task['csv_line'] is not None and task['flow_id'] is not None
        duration = D(task['trace_duration_us'])
        csv = D(task['csv']['Duration(us)'].strip())
        assert duration > 0 and abs(duration - csv) <= D('0.01')
        out[digest[:8]].append((ep, duration))
assert set(out) == {'7ddf7ae9', 'db852bff'}
result = {}
for digest, items in sorted(out.items()):
    assert len(items) == 8 and {ep for ep, _ in items} == set(range(8))
    a = [t / 1000 for ep, t in items if ep < 4]
    b = [t / 1000 for ep, t in items if ep >= 4]
    result[digest] = {
        'EP0_3_duration_ms_min_max': [str(min(a)), str(max(a))],
        'EP4_7_duration_ms_min_max': [str(min(b)), str(max(b))],
        'min_long_over_max_short': str(min(b) / max(a)),
        'per_ep_duration_ms': {str(ep): str(d / 1000) for ep, d in sorted(items)},
    }
print(json.dumps({'scope': 'one captured L23 host scope AICPU task per EP and digest',
                  'source_sha256': expected, 'groups': result,
                  'limitations': ['duration only, no cross-rank timestamp subtraction',
                                  'same name/scope/digest does not prove same collective or workload',
                                  'AICPU record duration is not physical network duration',
                                  'no causality or service baseline inference']}, indent=2))
