#!/usr/bin/env python3
"""Recheck EP8 L14/L39 saved task-level CSV and PMU summary; no NPU access."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parents[2]
paths = [root / 'diagnostic-20261029/mc2-extra-pipe-fields-16-tasks.json', root / 'agents/diagnosis/existing-mc2-pc-capture-audit.json']
expected = ['57d21f3ad1789ec38fdf0499ea65b6f5ca793e676d68193d3dbdc320b96f755c', '8363012053e686d5e06303194d039b3394f5f038796311b25ba77a03a323dfb6']
items = []
for path, digest in zip(paths, expected):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, path
    items.append(json.loads(data))
summary, pmu = items
out = {}
for layer in (14, 39):
    rows = [x for x in summary['rows'] if x['scope']['ep'] == 8 and x['scope']['layer'] == layer]
    targets = [x for x in pmu['ep8_targets'] if x['target']['layer'] == layer]
    assert len(rows) == len(targets) == 1
    row, target = rows[0]['row'], targets[0]
    assert int(row['Task ID']) == target['target']['task']
    assert int(row['Stream ID']) == target['target']['stream']
    assert int(row['Model ID']) == target['target']['model']
    assert int(Decimal(row['Start Time(us)'].strip()) * 1000) == target['target']['start_ns']
    assert target['pmu_row_count'] == 28 and len(target['pmu_distinct_timestamp_ns']) == 1
    assert target['end_ns'] - target['pmu_distinct_timestamp_ns'][0] == 20
    f = target['fields']
    assert int(row['aic_total_cycles']) == f['aic_total_cycles']
    out[layer] = {'duration_us': float(row['Duration(us)']), 'aic_total_cycles': f['aic_total_cycles'], 'aic_mac_ratio': f['aic_mac_ratio'], 'aic_mac_cycle_proxy': f['aic_total_cycles'] * f['aic_mac_ratio'], 'aiv_mte2_ratio': f['aiv_mte2_ratio']}
    print('layer', layer, out[layer])
print('L14/L39', {k: out[14][k] / out[39][k] for k in out[14]})
