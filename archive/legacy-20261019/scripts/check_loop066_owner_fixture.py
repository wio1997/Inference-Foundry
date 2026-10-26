#!/usr/bin/env python3
"""Require all eight real-entry owner fixture gates before a diagnostic passes."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixture-dir',required=True);a=ap.parse_args()
    root=Path(a.fixture_dir)
    files=sorted(root.glob('rank*.json'))
    expected=[root/f'rank{rank}.json' for rank in range(8)]
    if sorted(files)!=sorted(expected):
        raise SystemExit(f'expected eight exact fixture files, got {[p.name for p in files]}')
    rows=[json.loads(path.read_text()) for path in expected]
    for rank,row in enumerate(rows):
        if row['rank']!=rank or row['layer']!='model.layers.2.self_attn.attn':
            raise SystemExit(f'wrong rank/layer in rank{rank}')
        if row['x_shape']!=[96,4096] or row['owner_x_shape']!=[16,4096]:
            raise SystemExit(f'wrong shape in rank{rank}')
        if not row['gate_pass'] or not row['private_alias_preserved']:
            raise SystemExit(f'fixture gate failed rank{rank}')
        if any(not view['same_storage_as_state'] for view in row['indexer_typed_abi']):
            raise SystemExit(f'original typed alias changed rank{rank}')
    print(json.dumps({'rank_count':8,'all_pass':True,
                      'owner_pages_by_rank':[row['owner_state_pages'] for row in rows]}))

if __name__=='__main__':main()
