#!/usr/bin/env python3
"""Pair private full-producer Graph durations without extrapolating to Product."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    roots = a.run_dir
    gate = json.loads((roots / 'graph_check.json').read_text())
    if gate['status'] != 'private_full_producer_graph_pass':
        raise RuntimeError('Graph gate did not pass')
    rank_rows = []
    by_triplet = {}
    all_pair_diffs = []
    all_control_drift = []
    strict_wins = 0
    for rank in range(8):
        row = json.loads((roots / 'graph' / f'rank{rank}.json').read_text())
        groups = {}
        for sample in row['samples']:
            if sample['warmup']:
                continue
            groups.setdefault(sample['triplet'], {})[sample['arm']] = sample['replay_ms']
            by_triplet.setdefault(sample['triplet'], {}).setdefault(sample['arm'], []).append(sample['replay_ms'])
        if set(groups) != set(range(3, 13)):
            raise RuntimeError(f'rank{rank} measured triplets incomplete')
        pair_diffs = []
        drift = []
        for triplet, arms in sorted(groups.items()):
            if set(arms) != {'A', 'B', 'A2'}:
                raise RuntimeError(f'rank{rank} triplet{triplet} incomplete')
            pair_diffs.append(arms['B'] - (arms['A'] + arms['A2']) / 2)
            drift.append(abs(arms['A2'] - arms['A']))
            strict_wins += arms['B'] < min(arms['A'], arms['A2'])
        all_pair_diffs.extend(pair_diffs)
        all_control_drift.extend(drift)
        rank_rows.append({'rank': rank, 'median_B_minus_A_mid_us': 1000 * statistics.median(pair_diffs),
                          'median_abs_A_A2_drift_us': 1000 * statistics.median(drift),
                          'strict_B_faster_than_both_A': row['strict_B_faster_than_both_A_pairs']})
    max_rank_windows = []
    for triplet, arms in sorted(by_triplet.items()):
        max_a = max(arms['A'])
        max_b = max(arms['B'])
        max_a2 = max(arms['A2'])
        max_rank_windows.append({'triplet': triplet,
                                 'max_rank_B_minus_control_mid_us': 1000 * (max_b - (max_a + max_a2) / 2),
                                 'max_rank_A_A2_drift_us': 1000 * abs(max_a2 - max_a)})
    result = {
        'status': 'private_graph_pair_analysis_pass',
        'rank_rows': rank_rows,
        'strict_pair_wins': strict_wins, 'strict_pair_total': 80,
        'median_all_rank_pair_B_minus_control_mid_us': 1000 * statistics.median(all_pair_diffs),
        'median_all_rank_abs_control_drift_us': 1000 * statistics.median(all_control_drift),
        'median_max_rank_B_minus_control_mid_us': statistics.median(
            row['max_rank_B_minus_control_mid_us'] for row in max_rank_windows),
        'max_rank_windows': max_rank_windows,
        'source': 'Run307 8-rank private same-prestate A/B/A Graph replay, full producer→Sparse; restoration/sync outside events',
        'limits': [
            'Not a live FULL Target cycle or rank-rendezvous/communication measurement.',
            'Graph replay parity checked Sparse/QLI; post-Graph persistent cache bytes were not compared.',
            'No linear extrapolation across all c4 layers or Product TPS is justified.'
        ]}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'strict_pair_wins',
                  'median_all_rank_pair_B_minus_control_mid_us',
                  'median_max_rank_B_minus_control_mid_us')}))


if __name__ == '__main__':
    main()
