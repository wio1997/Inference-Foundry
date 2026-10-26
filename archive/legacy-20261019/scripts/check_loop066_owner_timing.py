#!/usr/bin/env python3
"""Validate all-rank private timing artifact and repeated topk parity."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--timing-dir', type=Path, required=True)
a = p.parse_args()
rows = []
for rank in range(8):
    path = a.timing_dir / f'rank{rank}.json'
    if not path.exists():
        raise SystemExit(f'missing timing rank{rank}')
    row = json.loads(path.read_text())
    if row.get('rank') != rank or row.get('stage') != 'complete' or row.get('pass') is not True:
        raise SystemExit(f'timing gate failed rank{rank}: stage={row.get("stage")} error={row.get("error")}')
    samples = row['samples']
    if len(samples) != 21 or sum(not x['warmup'] for x in samples) != 15:
        raise SystemExit(f'timing samples malformed rank{rank}')
    if any(not x['topk_exact_to_first_A'] for x in samples):
        raise SystemExit(f'timing topk mismatch rank{rank}')
    rows.append(row)
print(json.dumps({'ranks': len(rows), 'pass': True,
                  'median_ms_by_rank': {str(r['rank']): r['median_ms_by_arm'] for r in rows}},
                 indent=2))
