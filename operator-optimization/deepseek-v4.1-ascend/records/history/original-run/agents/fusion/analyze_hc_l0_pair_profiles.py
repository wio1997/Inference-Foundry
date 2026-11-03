"""Descriptive comparison of already selected layer-0 HcPre pairs; CPU only."""
import hashlib
import json
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
groups = defaultdict(list)
for line in source.read_text().splitlines():
    data = json.loads(line)
    if data['kind'] != 'rank':
        continue
    for scope in data['data']['layers']:
        if scope['layer'] != 0:
            continue
        candidates = [g for g in scope['operator_groups'] if g['operator_name'] == 'HcPre']
        assert len(candidates) == 1 and len(candidates[0]['tasks']) == 2
        group = candidates[0]
        tasks = sorted(group['tasks'], key=lambda t: Decimal(t['device_start_us']))
        assert len({(t['device_track_pid'], t['device_track_tid']) for t in tasks}) == 1
        assert len({t['flow_id'] for t in tasks}) == 2
        for task in tasks:
            csv = task['csv']
            assert Decimal(task['aic_mac_cycle_proxy']) == Decimal(csv['aic_total_cycles']) * Decimal(csv['aic_mac_ratio'])
            assert Decimal(task['aiv_vec_cycle_proxy']) == Decimal(csv['aiv_total_cycles']) * Decimal(csv['aiv_vec_ratio'])
        rows = int(group['physical_shape_dtype_format']['Input Shapes'].strip('"').split(',')[0])
        def ratio(key):
            a, b = (Decimal(t[key]) for t in tasks)
            assert a > 0 and b > 0
            return a / b
        groups[rows].append((ratio('trace_duration_us'), ratio('aic_mac_cycle_proxy'),
                             ratio('aiv_vec_cycle_proxy')))
assert {k:len(v) for k,v in groups.items()} == {595:8,643:8,1023:16}
for rows, pairs in sorted(groups.items()):
    print(f'M={rows}, n={len(pairs)}')
    for pos, name in enumerate(('duration', 'aic_cycle_times_mac_ratio', 'aiv_cycle_times_vec_ratio')):
        values = [x[pos] for x in pairs]
        print(f'  first/second {name}: min={min(values):.6f}, median={statistics.median(values):.6f}, max={max(values):.6f}')
print('Scope-matched descriptive proxies, not utilization, equal-work proof or causal evidence.')
