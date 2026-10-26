#!/usr/bin/env python3
"""All-rank one-call state/scatter/QLI gate for Run296."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--fixture-dir',type=Path,required=True)
a=p.parse_args()
rows=[]
for rank in range(8):
    path=a.fixture_dir/f'rank{rank}.json'
    if not path.exists():
        raise SystemExit(f'missing fixture rank{rank}')
    row=json.loads(path.read_text())
    rows.append(row)
    if row.get('rank')!=rank or row.get('stage')!='complete' or row.get('gate_pass') is not True:
        raise SystemExit(f'consumer fixture gate failed rank{rank}: stage={row.get("stage")} error={row.get("error")}')
    if row['output_A_B']['other_valid_slots']!=4 or row['owner_typed_slot']['valid_owner_slots']!=4:
        raise SystemExit(f'consumer owner slot count changed rank{rank}')
print(json.dumps({'ranks':len(rows),'gate_pass':True,
                  'qli_shapes':[r['qli_output_shape'] for r in rows]},indent=2))
