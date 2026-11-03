#!/usr/bin/env python3
"""Recompute anonymous CSV summaries and exact-signature duration histograms."""
import json
import math
from pathlib import Path
from statistics import median

root = Path(__file__).parent
data = json.loads((root / 'official-operators.json').read_text())
rows = data['samples']
assert len(rows) == 32 and data['counts']['failures'] == []
assert {(r['sample_id'], r['window_ordinal']) for r in rows} == {(r,w) for r in range(16) for w in (1,2)}
assert all(not r['invalid_or_nonpositive_rows'] for r in rows)
for window in (1,2):
    cohort = [r for r in rows if r['window_ordinal'] == window]
    assert all(r['kernel_observed_span_ns'] > 0 for r in cohort)
    print(f'Window {window}: span median {median(r["kernel_observed_span_ns"] for r in cohort)/1e6:.3f} ms')
    for kind in ('DispatchFFNCombine','QuantBatchMatmulV3','MatMulV2','HcPre',
                 'VllmAscendQuantLightningIndexerV2','SparseFlashMla'):
        assert all(0 <= r['types'][kind]['union_ns'] <= r['types'][kind]['sum_ns'] for r in cohort)
        t = median(r['types'][kind]['union_ns'] for r in cohort)/1e6
        share = median(r['types'][kind]['union_ns']/r['kernel_observed_span_ns']*100 for r in cohort)
        print(f'  {kind}: union median {t:.3f} ms; within-rank share median {share:.2f}%')
    print('  explicit communication union median ms:',
          median(r['communication_label_union_ns'] for r in cohort)/1e6)
    assert all(r['compute_comm_intersection_ns'] == 0 for r in cohort)

def stats(group):
    pairs = sorted((int(k),v) for k,v in group['histogram_ns'].items())
    assert pairs and all(k > 0 and isinstance(v,int) and v > 0 for k,v in pairs)
    n = sum(v for _,v in pairs)
    assert n == group['count'] == sum(group['rank_counts'].values())
    def at(i):
        count = 0
        for value, freq in pairs:
            count += freq
            if i < count:
                return value
        raise AssertionError('rank outside histogram')
    return n, pairs[0][0]/1000, (at((n-1)//2)+at(n//2))/2000, at(math.ceil(.95*n)-1)/1000, pairs[-1][0]/1000

shapes = json.loads((root / 'official-operator-shapes.json').read_text())
for group in shapes['groups']:
    n, lo, mid, p95, hi = stats(group)
    print(f'W{group["window"]} {group["type"]} {group["input_shapes"]}: '
          f'n={n}; us min/median/p95/max {lo:.3f}/{mid:.3f}/{p95:.3f}/{hi:.3f}')
print('Arithmetic checks only. CSV unions/attribution, clock mapping and hardware ceilings are not independently proved.')
