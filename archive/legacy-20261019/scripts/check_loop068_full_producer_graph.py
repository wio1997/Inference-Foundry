#!/usr/bin/env python3
"""Fail-closed all-rank private full-producer Graph replay validator."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args()
    rows = []
    for rank in range(8):
        row = json.loads((a.run_dir / 'graph' / f'rank{rank}.json').read_text())
        if row['rank'] != rank or row['stage'] != 'complete' or not row['pass']:
            raise RuntimeError(f'rank{rank} Graph failed: {row.get("error")}')
        if len(row['samples']) != 39 or any(not x['sparse_exact_to_eager_A'] or
                                              not x['topk_exact_to_eager_A'] or
                                              not all(x['persistent_owner_exact_to_eager_A'].values())
                                              for x in row['samples']):
            raise RuntimeError(f'rank{rank} Graph sample parity/inventory failed')
        rows.append(row)
    print(json.dumps({'status': 'private_full_producer_graph_pass',
                      'ranks': 8, 'paired_triplets_per_rank': 10,
                      'median_replay_ms_by_rank': {str(x['rank']): x['median_replay_ms_by_arm'] for x in rows},
                      'scope': 'private isolated Graph, not live full cycle or Product E2E'}, indent=2))


if __name__ == '__main__':
    main()
