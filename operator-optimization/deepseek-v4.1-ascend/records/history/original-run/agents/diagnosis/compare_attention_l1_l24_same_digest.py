#!/usr/bin/env python3
"""Read-only comparison of selected same-digest SparseFlashMla records."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[2]
source = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
records, digest_values = {}, set()
with source.open() as lines:
    for line in lines:
        obj = json.loads(line)
        if obj.get('kind') != 'rank':
            continue
        for s in obj['data']['layers']:
            if not s['scheduler_digest'].startswith('db852bff') or s['ep'] not in range(7) or s['layer'] not in (1, 24):
                continue
            digest_values.add(s['scheduler_digest'])
            key = s['ep'], s['layer']
            assert key not in records
            tasks = [x for group in s['operator_groups'] if group['operator_name'] == 'SparseFlashMla' for x in group['tasks']]
            assert len(tasks) == 1
            records[key] = tasks[0]
assert len(digest_values) == 1 and len(records) == 14
ratios, vectors = [], {1: [], 24: []}
for ep in range(7):
    a, b = records[ep, 1], records[ep, 24]
    ca, cb = a['csv'], b['csv']
    sa, sb = [c['Input Shapes'].strip('"').split(';') for c in (ca, cb)]
    assert sa[0] == sb[0] == '1023,64,512'
    assert sa[1] == sb[1] == '17315,128,1,512'
    assert sa[2] == '' and sb[2] == '17315,128,1,512' and sb[4] == '1023,1,512'
    assert ca['Output Shapes'] == cb['Output Shapes'] == '"1023,64,512;"'
    dtype_a, dtype_b = ca['Input Data Types'].split(';'), cb['Input Data Types'].split(';')
    assert dtype_a[:3] == ['DT_BF16', 'DT_BF16', 'DT_UNDEFINED']
    assert dtype_b[:3] == ['DT_BF16', 'DT_BF16', 'DT_BF16'] and dtype_b[4] == 'INT32'
    assert ca['Output Data Types'] == cb['Output Data Types']
    d1, d24 = Decimal(a['trace_duration_us']), Decimal(b['trace_duration_us'])
    assert d1 > 0 and d24 > 0
    ratios.append(d24/d1)
    for layer, csv in ((1, ca), (24, cb)):
        vectors[layer].append(Decimal(csv['aiv_vec_ratio']))
print('duration ratio L24/L1 min/median/max', min(ratios), statistics.median(ratios), max(ratios))
for layer, vals in vectors.items():
    print('layer', layer, 'AIV vector displayed ratio min/max', min(vals), max(vals))
