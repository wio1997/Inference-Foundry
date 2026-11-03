#!/usr/bin/env python3
"""Compare selected same-scope HcPost and rotary task-pair duration ratios."""
from collections import Counter, defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from statistics import median
import json
import sys

root = Path(__file__).resolve().parents[2]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
)
assert sha256(path.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
digests = ('db852bff', '1b826883')
rows = defaultdict(list)
excluded = Counter()
seen = set()

def shape(group):
    return group['physical_shape_dtype_format']['Input Shapes'].strip('"')

def pair_ratio(group):
    assert len(group['tasks']) == 2
    first, second = sorted(group['tasks'], key=lambda task: Decimal(task['device_start_us']))
    identity = lambda t: (t['device_track_pid'], t['device_track_tid'], t['csv']['Stream ID'])
    assert identity(first) == identity(second)
    for field in ('Name', 'Type', 'Input Shapes', 'Output Shapes', 'Input Data Types',
                  'Output Data Types', 'Input Formats', 'Output Formats', 'Accelerator Core',
                  'Block Num', 'Mix Block Num'):
        assert first['csv'][field] == second['csv'][field]
    d1, d2 = Decimal(first['trace_duration_us']), Decimal(second['trace_duration_us'])
    assert min(d1, d2) > 0
    assert Decimal(first['device_start_us']) + d1 <= Decimal(second['device_start_us'])
    return d2 / d1

with path.open() as source:
    for line in source:
        record = json.loads(line)
        if record.get('kind') != 'rank':
            continue
        for scope in record['data']['layers']:
            digest = scope['scheduler_digest'][:8]
            if digest not in digests:
                continue
            ep, layer = scope['ep'], scope['layer']
            key = (digest, ep, layer)
            assert key not in seen
            seen.add(key)
            gs = scope['operator_groups']
            hcs = [g for g in gs if g['operator_name'] == 'HcPost'
                   and shape(g).startswith('1,1023,5120;') and len(g['tasks']) == 2]
            assert len(hcs) == 1
            rg = [g for g in gs if g['operator_name'] == 'InplacePartialRotaryMul'
                  and shape(g) == '1023,1,64,512;1023,1,1,64;1023,1,1,64'
                  and len(g['tasks']) == 2]
            if not rg:
                short = [g for g in gs if g['operator_name'] == 'InplacePartialRotaryMul'
                         and shape(g) == '1019,1,64,512;1019,1,1,64;1019,1,1,64'
                         and len(g['tasks']) == 2]
                assert len(short) == 1 and ep in (7, 15)
                excluded[(digest, ep)] += 1
                continue
            assert len(rg) == 1 and ep not in (7, 15)
            for task in rg[0]['tasks']:
                c = task['csv']
                assert (c['Type'], c['Input Data Types'], c['Output Data Types'],
                        c['Input Formats'], c['Output Formats'], c['Accelerator Core'],
                        c['Block Num'], c['Mix Block Num']) == (
                    'InplacePartialRotaryMul', 'DT_BF16;FLOAT;FLOAT', 'DT_BF16',
                    'ND;ND;ND', 'ND', 'AI_VECTOR_CORE', '47', '0')
            rows[digest].append((pair_ratio(rg[0]), pair_ratio(hcs[0])))

assert len(seen) == 640
assert excluded == {('db852bff', 7): 40, ('1b826883', 15): 40}
assert all(len(rows[d]) == 280 for d in digests)
result = {}
for digest, pairs in rows.items():
    result[digest] = {
        'scopes': len(pairs),
        'rope_second_over_first_median': str(median(a for a, _ in pairs)),
        'hc_second_over_first_median': str(median(b for _, b in pairs)),
        'rope_ratio_greater_than_hc_ratio': sum(a > b for a, b in pairs),
        'hc_under_0_9_and_rope_at_least_0_95': sum(
            b < Decimal('0.9') and a >= Decimal('0.95') for a, b in pairs
        ),
    }
print(json.dumps({
    'source_sha256': '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4',
    'results': result,
    'excluded': {'db852bff_ep7_1019_rows': 40, '1b826883_ep15_1019_rows': 40},
    'limits': 'Same-scope exported trace durations; not tensor identity, equal work, or causality.',
}, indent=2))
