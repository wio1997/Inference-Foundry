#!/usr/bin/env python3
"""Sensitivity of previously extracted pooled MAC proxy to rank weighting.

No profiler, network, device, or original-source changes. Rankings describe the
existing selected windows only; equal-rank weighting does not correct selection.
"""
import hashlib
import json
from collections import defaultdict
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 40
root = Path(__file__).resolve().parents[2]
source = root / 'agents/diagnosis/draft-target-operator-decomposition.json'
expected = '7384031bcfa37219d347f535ae5cf8cf7a099d4f89565867af7288de9d795197'
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected
value = json.loads(raw)
groups = defaultdict(lambda: defaultdict(lambda: {'num': Decimal(0), 'den': Decimal(0), 'windows': 0, 'positive': 0, 'skipped': 0}))
identities = set()
for window in value['windows']:
    kind, rank = window['kind'], window['rank']
    ident = (kind, rank, window['start'], window['end'])
    assert ident not in identities and window['start'] < window['end']
    identities.add(ident)
    item = groups[kind][rank]
    item['num'] += Decimal(window['num'])
    item['den'] += Decimal(window['den'])
    item['windows'] += 1
    item['positive'] += window['positive']
    item['skipped'] += window['skipped']
assert set(groups) == {'draft', 'target'}
assert set(groups['draft']) == set(groups['target']) and len(groups['draft']) == 15
assert sum(x['windows'] for x in groups['draft'].values()) == 162
assert sum(x['windows'] for x in groups['target'].values()) == 45
assert all(x['den'] > 0 for g in groups.values() for x in g.values())
results = {}
for kind, ranks in groups.items():
    num = sum(x['num'] for x in ranks.values())
    den = sum(x['den'] for x in ranks.values())
    loo = {r: (num - x['num']) / (den - x['den']) for r, x in ranks.items()}
    results[kind] = {
        'windows': sum(x['windows'] for x in ranks.values()),
        'positive_rows': sum(x['positive'] for x in ranks.values()),
        'skipped_rows': sum(x['skipped'] for x in ranks.values()),
        'pooled_ratio': str(num / den),
        'equal_rank_mean_of_ratios': str(sum(x['num'] / x['den'] for x in ranks.values()) / len(ranks)),
        'max_single_rank_share_of_pooled_denominator': str(max(x['den'] / den for x in ranks.values())),
        'leave_one_rank_out_min': str(min(loo.values())),
        'leave_one_rank_out_max': str(max(loo.values())),
        'per_rank': {r: {'windows': x['windows'], 'ratio': str(x['num'] / x['den']), 'leave_out_ratio': str(loo[r])} for r, x in sorted(ranks.items())},
    }
assert Decimal(results['draft']['pooled_ratio']) < Decimal(results['target']['pooled_ratio'])
paired = {r: Decimal(results['target']['per_rank'][r]['leave_out_ratio']) - Decimal(results['draft']['per_rank'][r]['leave_out_ratio']) for r in groups['draft']}
assert all(v > 0 for v in paired.values())
output = {'input': str(source.relative_to(root)), 'input_sha256': expected, 'results': results,
          'paired_leave_one_rank_out_target_minus_draft_min': str(min(paired.values())),
          'limits': ['Selected historical windows, not full workload or whole-device utilization.',
                     'Ranks and work are not paired; equal-rank means do not correct missing windows or work imbalance.',
                     'Only retained positive AIC-cycle rows on main stream; excluded rows may contain work.',
                     'Values reflect exported ratio precision and do not establish physical independent cycles.']}
print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
