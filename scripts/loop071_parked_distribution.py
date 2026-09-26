"""Summarize existing original-Graph active masks without inferring saved FLOPs."""

import argparse
import collections
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    by_rank = {}
    for rank in range(8):
        files = sorted(args.source.glob(f'rank{rank}_cohort*.jsonl'))
        if len(files) != 5:
            raise SystemExit(f'rank{rank}: expected five cohorts, got {len(files)}')
        rows = {}
        for file in files:
            cohort = int(file.stem.split('cohort')[-1])
            for line in file.read_text().splitlines():
                item = json.loads(line)
                key = (cohort, int(item['cycle']))
                active = tuple(int(v) for v in item['active_mask'])
                if len(active) != 12 or key in rows:
                    raise SystemExit(f'bad active mask or duplicate {key}')
                rows[key] = active
        by_rank[rank] = rows
    keys = sorted(by_rank[0])
    for rank in range(1, 8):
        if set(by_rank[rank]) != set(keys):
            raise SystemExit(f'rank{rank} cycle set differs')
        for key in keys:
            if by_rank[rank][key] != by_rank[0][key]:
                raise SystemExit(f'rank{rank} mask differs at {key}')
    hist = collections.Counter(sum(by_rank[0][key]) for key in keys)
    cohorts = {}
    for cohort in sorted({key[0] for key in keys}):
        sub = [key for key in keys if key[0] == cohort]
        active = [sum(by_rank[0][key]) for key in sub]
        cohorts[str(cohort)] = {'cycles': len(sub), 'first_parking_cycle': next(
            (key[1] for key, value in zip(sub, active) if value < 12), None),
            'parked_slot_cycles': sum(12 - value for value in active),
            'active_histogram': {str(k): v for k, v in sorted(collections.Counter(active).items())}}
    parked = sum((12 - active) * count for active, count in hist.items())
    doc = {'scope': 'Run287 original-path 8-rank x 5-cohort active mask census, one copy per cycle after exact cross-rank gate; no MoE arithmetic or time saving claim',
           'cycles': len(keys), 'total_slot_cycles': 12 * len(keys),
           'parked_slot_cycles': parked, 'parked_slot_fraction': parked / (12 * len(keys)),
           'active_count_histogram': {str(k): v for k, v in sorted(hist.items())},
           'cohorts': cohorts,
           'limitations': ['Instrumented Run287 trajectory, not formal Run99',
                           'Parked rows may still be required by some layers/communication layouts',
                           'Compaction and restore overheads unknown',
                           'No useful-token or HBM proportionality implied']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'cycles': doc['cycles'], 'parked_slot_cycles': parked,
                      'parked_slot_fraction': doc['parked_slot_fraction'],
                      'active_count_histogram': doc['active_count_histogram']}))


if __name__ == '__main__':
    main()
