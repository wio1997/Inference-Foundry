#!/usr/bin/env python3
"""Fail-closed frozen c12 FULL Graph carrier gate for live owner diagnostic."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--owner', action='store_true')
    a = p.parse_args()
    root = a.run_dir
    bench = json.loads((root / 'bench12.json').read_text())
    if bench['summary']['success'] != 12 or any(x['output_tokens'] != 1024 for x in bench['requests']):
        raise RuntimeError('12x1024 client correctness gate failed')
    paths = sorted((root / 'runtime').glob('rank*_cohort*.json'))
    if len(paths) != 8:
        raise RuntimeError(f'eight Runtime reports required, found {len(paths)}')
    rows = [json.loads(path.read_text()) for path in paths]
    if {x['rank'] for x in rows} != set(range(8)) or not all(
            x['pass'] and x['target_graph_mode'] == 'FULL' for x in rows):
        raise RuntimeError('all-eight FULL Graph Runtime gate failed')
    trace = {}
    if a.owner:
        for rank in range(8):
            path = root / 'owner_trace' / f'rank{rank}.jsonl'
            events = [json.loads(x) for x in path.read_text().splitlines()]
            if not any(x['capturing'] and x['full_shape'] == [96, 4096] for x in events):
                raise RuntimeError(f'rank{rank} owner path not captured in FULL Graph')
            trace[rank] = events
        compare = json.loads((root / 'content_compare.json').read_text())
        if compare['status'] != 'same_prompt_content_exact':
            raise RuntimeError('candidate output content differs from original')
    print(json.dumps({'status': 'live_owner_carrier_pass' if a.owner else 'original_carrier_pass',
                      'ranks': 8, 'target_graph_mode': 'FULL',
                      'client_success': 12, 'tokens_each': 1024,
                      'owner_capture_ranks': len(trace),
                      'scope': 'single diagnostic cohort, not formal E2E performance'}, indent=2))


if __name__ == '__main__':
    main()
