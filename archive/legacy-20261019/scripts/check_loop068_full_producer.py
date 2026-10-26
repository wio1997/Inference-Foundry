#!/usr/bin/env python3
"""Fail-closed eight-rank validator for the private full-producer value gate."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args()
    root = a.run_dir
    bench = json.loads((root / 'bench12.json').read_text())
    if bench['summary']['success'] != 12 or any(x['output_tokens'] != 1024 for x in bench['requests']):
        raise RuntimeError('12x1024 client correctness gate failed')
    runtime_files = sorted((root / 'runtime').glob('rank*_cohort*.json'))
    if len(runtime_files) != 8:
        raise RuntimeError('expected eight Runtime reports')
    runtime = [json.loads(path.read_text()) for path in runtime_files]
    if {x['rank'] for x in runtime} != set(range(8)) or not all(x['pass'] for x in runtime):
        raise RuntimeError('eight-rank Runtime gate failed')
    fixtures = []
    for rank in range(8):
        row = json.loads((root / 'fixture' / f'rank{rank}.json').read_text())
        if row['rank'] != rank or row['stage'] != 'complete' or not row['pass']:
            raise RuntimeError(f'rank{rank} private fixture failed: {row.get("error")}')
        if [arm['arm'] for arm in row['arms']] != ['A', 'A2', 'B', 'A3']:
            raise RuntimeError(f'rank{rank} fixture arm order changed')
        if any(not all(arm['exact_to_first_A'].values()) for arm in row['arms']):
            raise RuntimeError(f'rank{rank} typed/Sparse parity failed')
        fixtures.append(row)
    result = {'status': 'private_full_producer_value_gate_pass',
              'ranks': 8, 'client_success': 12, 'tokens_each': 1024,
              'scope': 'one eager real layer2 prestate, private bank restored each arm; not persistent Product correctness or performance',
              'private_store_bytes_per_rank': fixtures[0]['private_store_bytes'],
              'owner_sparse_shape': fixtures[0]['arms'][2]['sparse_shape'],
              'owner_topk_shape': fixtures[0]['arms'][2]['topk_shape']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
