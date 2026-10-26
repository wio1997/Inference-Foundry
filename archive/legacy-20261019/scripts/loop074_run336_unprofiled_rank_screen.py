"""Retrospective rank duration skew on Run188 original no-profiler prefill calls."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir
    tags = {}
    for tag in ('A', 'A2'):
        ranks = []
        for rank in range(8):
            path = root / 'forward' / f'rank{rank}_{tag}.jsonl'
            ranks.append([json.loads(x) for x in path.read_text().splitlines() if x])
        if len({len(x) for x in ranks}) != 1:
            raise RuntimeError(f'{tag} rank count mismatch')
        calls = []
        for index in range(len(ranks[0])):
            rows = [rank[index] for rank in ranks]
            if len({(x['mode'], x['num_actual_tokens'], x['num_tokens_padded'],
                     x['num_reqs']) for x in rows}) != 1:
                raise RuntimeError(f'{tag} call{index} rank shape mismatch')
            if rows[0]['mode'] != 'NONE':
                continue
            wall = [x['forward_wall_ms'] for x in rows]
            calls.append({'call_index': index, 'tokens': rows[0]['num_actual_tokens'],
                          'requests': rows[0]['num_reqs'], 'wall_ms_by_rank': wall,
                          'longest_rank': max(range(8), key=lambda rank: wall[rank]),
                          'duration_spread_ms': max(wall) - min(wall)})
        tags[tag] = {'calls': calls,
                     'longest_rank_counts': [sum(x['longest_rank'] == rank for x in calls)
                                             for rank in range(8)],
                     'duration_spread_ms_median': statistics.median(
                         x['duration_spread_ms'] for x in calls)}
    print(json.dumps({'status': 'retrospective_unprofiled_host_forward_duration',
                      'source': 'Run188 A/A2 original no-profiler warmed legal cohorts',
                      'tags': tags,
                      'limits': [
                          'Per-call wall duration does not record absolute start or collective enqueue; a rank can arrive last despite a shorter duration.',
                          'Different Run188 trajectory from Run333; duration ranking cannot prove the profiler created or removed a critical path.',
                          'No device/link service or Product bound follows.',
                      ]}, indent=2))


if __name__ == '__main__':
    main()
