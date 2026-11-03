#!/usr/bin/env python3
"""Describe within-round association using frozen anonymous client timing only."""
from hashlib import sha256
from pathlib import Path
from statistics import mean
import json

root = Path(__file__).resolve().parents[2]
raw = (root / 'agents/diagnosis/baseline-latency-anonymized.json').read_bytes()
assert sha256(raw).hexdigest() == '0819ff3140d285cac881f2e48141548bd08b734e516a1e91687e2108a6dce4e9'
data = json.loads(raw)
out = {}
for phase in ('measure1', 'measure2'):
    rows = sorted(data['phases'][phase]['records'], key=lambda r: r['start_offset_ms'])
    assert len(rows) == 8 and len({r['start_offset_ms'] for r in rows}) == 8 and len({r['ttft_ms'] for r in rows}) == 8
    ttft_order = sorted(range(8), key=lambda i: rows[i]['ttft_ms'])
    rank = {idx: k + 1 for k, idx in enumerate(ttft_order)}
    ranks = [rank[i] for i in range(8)]
    rho = 1 - 6 * sum((i + 1 - ranks[i]) ** 2 for i in range(8)) / (8 * (8 * 8 - 1))
    out[phase] = {'start_span_ms': rows[-1]['start_offset_ms'] - rows[0]['start_offset_ms'],
                  'first_response_delay_ranks_in_start_order': ranks,
                  'spearman_start_vs_first_response_delay': rho,
                  'first_four_mean_ms': mean(r['ttft_ms'] for r in rows[:4]),
                  'last_four_mean_ms': mean(r['ttft_ms'] for r in rows[4:])}
print(json.dumps({'rounds': out, 'limits': 'Post hoc paired client timings, not service admission/order, device activity, matched work, causal effect or independent repetitions.'}, indent=2))
