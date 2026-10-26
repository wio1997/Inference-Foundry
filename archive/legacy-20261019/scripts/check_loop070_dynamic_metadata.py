#!/usr/bin/env python3
"""Fail-closed gate for private dynamic metadata Graph comparison."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    root = p.parse_args().run_dir
    bench = json.loads((root / 'bench12.json').read_text())
    if bench['summary']['success'] != 12 or any(
            req['output_tokens'] != 1024 for req in bench['requests']):
        raise RuntimeError('12x1024 client carrier failed')
    paths = list((root / 'runtime').glob('rank*_cohort*.json'))
    if len(paths) != 8:
        raise RuntimeError(f'eight Runtime reports required, found {len(paths)}')
    reports = [json.loads(x.read_text()) for x in paths]
    if {x['rank'] for x in reports} != set(range(8)) or not all(
            x['pass'] for x in reports):
        raise RuntimeError('all-eight Runtime carrier failed')
    probes = [json.loads((root / 'fixture' / f'rank{rank}.json').read_text())
              for rank in range(8)]
    for rank, probe in enumerate(probes):
        if probe['rank'] != rank or not probe['pass'] or probe['stage'] != 'complete':
            raise RuntimeError(f'rank{rank} dynamic metadata probe failed')
        reference_changed = probe['reference_metadata_changed_by_delta']['-4']
        captured_changed = probe['captured_metadata_changed_by_delta']['-4']
        if (not any(reference_changed.values())
                or captured_changed != reference_changed
                or probe.get('capture_still_active')):
            raise RuntimeError(f'rank{rank} Graph metadata did not respond to start shift')
        rows = probe['samples']
        if [(x['delta'], x['arm']) for x in rows] != [
                (delta, arm) for delta in (0, -1, -4) for arm in ('A', 'B', 'A2')]:
            raise RuntimeError(f'rank{rank} missing scenario/arm')
        if any(not x['sparse_finite'] or not x['sparse_finite_eager_A'] or
               not all(x['value_exact_to_graph_A'].values()) or
               not all(x['owner_metadata_exact'].values()) or
               any(not value for key, value in x['value_exact_to_eager_A'].items()
                   if key != 'sparse') or
               (x['delta'] == 0 and not x['value_exact_to_eager_A']['sparse'])
               for x in rows):
            raise RuntimeError(f'rank{rank} value/metadata parity failed')
    print(json.dumps({
        'status': 'dynamic_metadata_private_graph_pass', 'ranks': 8,
        'deltas': [0, -1, -4], 'arms': ['A', 'B', 'A2'],
        'carrier': '12x1024 + eight Runtime reports',
        'scope': 'single real prestate with synthetic start shifts; Graph A/B parity required. Synthetic eager-vs-Graph Sparse may differ and is reported, not Product correctness',
    }, indent=2))


if __name__ == '__main__':
    main()
