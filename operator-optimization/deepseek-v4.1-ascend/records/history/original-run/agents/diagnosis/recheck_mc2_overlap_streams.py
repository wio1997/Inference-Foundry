#!/usr/bin/env python3
"""Recheck selected MC2 intervals, stream intersections, and aggregate ratios offline."""
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from statistics import median
import json

root = Path(__file__).resolve().parents[2]
source = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
aggregate = root / 'agents/diagnosis/16rank-selected-device-intervals.json'
assert sha256(source.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert sha256(aggregate.read_bytes()).hexdigest() == '525a2ab842ddaf89809777d0e34d5126dc11d14c86671b269b93733a077b3559'
summary = json.loads(aggregate.read_text())
ratios = [Decimal(r['mc2_other_overlap_us']) / Decimal(r['mc2_union_us']) for r in summary['results']]
groups = same_stream = other_stream = 0
for line in source.open():
    record = json.loads(line)
    if record.get('kind') != 'rank':
        continue
    by_digest = defaultdict(list)
    for layer in record['data']['layers']:
        for group in layer['operator_groups']:
            for task in group['tasks']:
                start = Decimal(task['device_start_us'])
                end = start + Decimal(task['trace_duration_us'])
                assert end > start
                by_digest[layer['scheduler_digest']].append((group['operator_name'], task['csv']['Stream ID'], start, end))
    for rows in by_digest.values():
        mc2 = [r for r in rows if r[0] == 'DispatchFFNCombine']
        others = [r for r in rows if r[0] != 'DispatchFFNCombine']
        assert len(mc2) == 40 and len({r[1] for r in mc2}) == 1
        groups += 1
        for a in mc2:
            for b in others:
                if a[2] < b[3] and b[2] < a[3]:
                    if a[1] == b[1]:
                        same_stream += 1
                    else:
                        other_stream += 1
assert groups == len(ratios) == 32 and same_stream == 0
print(json.dumps({'groups': groups, 'same_stream_interval_pairs': same_stream,
                  'other_stream_interval_pairs': other_stream,
                  'overlap_fraction_min_median_max': [str(min(ratios)), str(median(ratios)), str(max(ratios))],
                  'limits': 'Selected task intervals only; interval pairs can overlap each other. No causal or critical-path conclusion.'}, indent=2))
