#!/usr/bin/env python3
"""Offline conditional rounding sensitivity of archived Engram pairs."""
from collections import defaultdict
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
import json

root = Path(__file__).resolve().parents[2]
relative = 'agents/diagnosis/engram-pairwise-existing.json'
expected = '0487234fa2637a5c1c6add39bb54960b865f11b25644dedc9c3792fb15be9efc'
data = (root / relative).read_bytes()
if sha256(data).hexdigest() != expected:
    raise SystemExit('STOP: input hash mismatch')
j = json.loads(data)
rows = []
seen = set()
for record in j['rows']:
    key = (record['rank'], record['run'])
    if key in seen:
        raise SystemExit('STOP: duplicate rank/run')
    seen.add(key)
    c0 = D(record['aic_total_cycles']['after0'])
    c1 = D(record['aic_total_cycles']['after13'])
    r0 = D(record['aic_mac_ratio']['after0'])
    r1 = D(record['aic_mac_ratio']['after13'])
    if c0 <= 0 or c1 <= 0:
        raise SystemExit('STOP: nonpositive cycle denominator')
    # Candidate only: displayed ratios rounded to nearest 0.001, exact cycle
    # denominator, and common formula r = MAC cycles / total cycles.
    midpoint = c0 * r0 - c1 * r1
    halfwidth = D('0.0005') * (c0 + c1)
    low, high = midpoint - halfwidth, midpoint + halfwidth
    rows.append({'rank': key[0], 'run': key[1],
                 'midpoint_diff_cycles': str(midpoint),
                 'halfwidth_cycles': str(halfwidth),
                 'interval_lower_cycles': str(low),
                 'interval_upper_cycles': str(high),
                 'zero_in_interval': low <= 0 <= high,
                 'relative_abs_midpoint': str(abs(midpoint)/halfwidth)})
if len(rows) != 48 or len(seen) != 48:
    raise SystemExit('STOP: expected exactly 48 pairs')
summary = {}
for run in (1, 2, 3):
    rs = [r for r in rows if r['run'] == run]
    if len(rs) != 16:
        raise SystemExit('STOP: expected 16 pairs per run')
    summary[str(run)] = {'pairs': len(rs),
        'positive_midpoints': sum(D(r['midpoint_diff_cycles']) > 0 for r in rs),
        'negative_midpoints': sum(D(r['midpoint_diff_cycles']) < 0 for r in rs),
        'zero_in_interval': sum(r['zero_in_interval'] for r in rs)}
output = {'scope': '48 selected same-rank/run pairs; conditional interval sensitivity',
          'input_sha256': expected,
          'hypotheses_not_validated': ['ratio is MAC cycles divided by total cycles',
             'ratio displayed after nearest rounding to 0.001',
             'displayed total cycle denominators exact and not truncated',
             'the reported fields correspond to the same actual replay'],
          'formula': 'difference = c0*r0 - c1*r1; halfwidth = 0.0005*(c0+c1)',
          'conclusion_limit': 'Intervals containing zero do not prove the actual values are equal.',
          'summary': summary,
          'max_abs_midpoint_cycles': str(max(abs(D(r['midpoint_diff_cycles'])) for r in rows)),
          'max_abs_midpoint_over_halfwidth': str(max(D(r['relative_abs_midpoint']) for r in rows)),
          'rows': rows}
print(json.dumps(output, indent=2, ensure_ascii=False))
