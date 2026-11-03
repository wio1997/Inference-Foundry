#!/usr/bin/env python3
"""Conditional rounding sensitivity of the previously saved layer-0 MC2 rows."""
import hashlib
import json
from pathlib import Path
from statistics import median

source = Path(__file__).with_name('layer0-engine-work-proxies.json')
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '9738b889bdf3f8d418f5bb1e9c2bd83c4c07f7c32e36e62c9c1c08482258f860'
rows = json.loads(raw)['per_rank']
for run in (1, 2, 3):
    groups = {dp: [r for r in rows if r['run'] == run and r['dp'] == dp]
              for dp in ('dp0', 'dp1')}
    assert all(len(group) == 8 and len({r['rank'] for r in group}) == 8
               for group in groups.values())
    print(f'run={run}')
    for key in ('mac_ratio', 'mte2_ratio'):
        bounds = {}
        for dp, group in groups.items():
            bounds[dp] = [median(r['aic_cycles'] * max(0, min(1, r[key] + d))
                                 for r in group) for d in (-0.0005, 0.0005)]
        low = bounds['dp1'][0] / bounds['dp0'][1]
        high = bounds['dp1'][1] / bounds['dp0'][0]
        print(f'  {key}: [{low:.4f}, {high:.4f}]')
