#!/usr/bin/env python3
"""Compare exported counter products within stored, matched boundary pairs."""
import hashlib
import json
import statistics
from pathlib import Path

p = Path('agents/diagnosis/engram-pairwise-existing.json')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '0487234fa2637a5c1c6add39bb54960b865f11b25644dedc9c3792fb15be9efc'
rows = json.loads(p.read_text())['rows']
assert len(rows) == 48 and len({(r['rank'], r['run']) for r in rows}) == 48
for run in (1, 2, 3):
    group = [r for r in rows if r['run'] == run]
    assert len(group) == 16
    result = {}
    for ratio in ('aic_mac_ratio', 'aic_mte2_ratio'):
        products = []
        for r in group:
            a = int(r['aic_total_cycles']['after0']) * float(r[ratio]['after0'])
            b = int(r['aic_total_cycles']['after13']) * float(r[ratio]['after13'])
            assert a > 0 and b > 0
            products.append(a / b)
        result[ratio] = {'min': min(products), 'median': statistics.median(products), 'max': max(products)}
    for field in ('aic_total_cycles', 'aic_mte2_ratio'):
        values = [float(r[field]['after0']) / float(r[field]['after13']) for r in group]
        result[field + '_field_ratio'] = {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
    print(json.dumps({'run': run, 'pairs': len(group), **result}, ensure_ascii=False))
