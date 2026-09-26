#!/usr/bin/env python3
"""Paired all-rank A96/B16/A96 event-span analysis; no E2E extrapolation."""
import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--timing-dir', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
rank_rows = []
for rank in range(8):
    row = json.loads((a.timing_dir / f'rank{rank}.json').read_text())
    if not row['pass'] or row['stage'] != 'complete':
        raise SystemExit(f'rank{rank} gate failed')
    pairs = []
    for triplet in range(2, 7):
        items = {v['arm']: v for v in row['samples'] if v['triplet'] == triplet}
        if set(items) != {'A', 'B', 'A2'}:
            raise SystemExit(f'rank{rank} triplet{triplet} missing arm')
        item = {'triplet': triplet}
        for metric in ('update_ms', 'qli_tail_ms', 'through_qli_ms'):
            a0, b, a2 = (items[name][metric] for name in ('A', 'B', 'A2'))
            ctrl = (a0 + a2) / 2
            item[metric] = {'A': a0, 'B': b, 'A2': a2,
                            'full_control_mean': ctrl,
                            'B_minus_full_control': b - ctrl,
                            'A_A2_abs_drift': abs(a0 - a2),
                            'B_faster_than_both': b < min(a0, a2)}
        pairs.append(item)
    summary = {}
    for metric in ('update_ms', 'qli_tail_ms', 'through_qli_ms'):
        m = [x[metric] for x in pairs]
        summary[metric] = {
            'median_full_control_ms': statistics.median(x['full_control_mean'] for x in m),
            'median_B_ms': statistics.median(x['B'] for x in m),
            'median_paired_B_minus_full_ms': statistics.median(x['B_minus_full_control'] for x in m),
            'median_A_A2_abs_drift_ms': statistics.median(x['A_A2_abs_drift'] for x in m),
            'B_faster_than_both_count': sum(x['B_faster_than_both'] for x in m),
            'pairs': len(m),
        }
    rank_rows.append({'rank': rank, 'summary': summary, 'pairs': pairs})
aggregate = {}
for metric in ('update_ms', 'qli_tail_ms', 'through_qli_ms'):
    aggregate[metric] = {
        'all_rank_paired_B_minus_full_median_ms': statistics.median(
            r['summary'][metric]['median_paired_B_minus_full_ms'] for r in rank_rows),
        'rank_medians_B_faster_count': sum(
            r['summary'][metric]['median_paired_B_minus_full_ms'] < 0 for r in rank_rows),
        'all_rank_both_control_pair_wins': sum(
            r['summary'][metric]['B_faster_than_both_count'] for r in rank_rows),
        'total_pairs': 40,
    }
out = {'status': 'valid_private_eager_event_span', 'scope': 'same-prestate one-layer private chain, not Graph/Product',
       'ranks': rank_rows, 'aggregate': aggregate}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(aggregate, indent=2))
