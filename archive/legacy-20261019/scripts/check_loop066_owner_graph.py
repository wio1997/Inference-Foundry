#!/usr/bin/env python3
"""Require all-rank private Graph capture/replay semantic and sample gate."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--graph-dir', type=Path, required=True)
a = p.parse_args()
rows = []
for rank in range(8):
    path = a.graph_dir / f'rank{rank}.json'
    if not path.exists():
        raise SystemExit(f'missing graph rank{rank}')
    row = json.loads(path.read_text())
    if row.get('rank') != rank or row.get('stage') != 'complete' or row.get('pass') is not True:
        raise SystemExit(f'graph gate failed rank{rank}: stage={row.get("stage")} error={row.get("error")}')
    samples = row['samples']
    if len(samples) != 39 or sum(not x['warmup'] for x in samples) != 30:
        raise SystemExit(f'graph sample count malformed rank{rank}')
    rows.append(row)
print(json.dumps({'ranks': len(rows), 'pass': True,
                  'captured_kv_shapes': {str(r['rank']): r['captured_kv_shapes'] for r in rows},
                  'median_replay_ms_by_rank': {str(r['rank']): r['median_replay_ms_by_arm'] for r in rows}},
                 indent=2))
