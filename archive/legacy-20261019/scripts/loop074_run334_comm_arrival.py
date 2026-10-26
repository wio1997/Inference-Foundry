"""Align same ordered CANN HCCL tasks across all8 profiled prefill ranks."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir / 'profiler'
    ranks = []
    for rank in range(8):
        dirs = list((root / f'rank{rank}').glob('*/ASCEND_PROFILER_OUTPUT'))
        if len(dirs) != 1:
            raise RuntimeError(f'rank{rank} output count {len(dirs)}')
        rows = list(csv.DictReader((dirs[0] / 'kernel_details.csv').open()))
        comm = [row for row in rows if row['Accelerator Core'] == 'COMMUNICATION']
        ranks.append(comm)
    counts = [len(x) for x in ranks]
    if counts != [264] * 8:
        raise RuntimeError(f'communication task count mismatch {counts}')
    task = []
    for index in range(264):
        rows = [rank[index] for rank in ranks]
        names = [x['Name'] for x in rows]
        if len(set(names)) != 1:
            raise RuntimeError(f'collective name/order mismatch {index}: {names}')
        starts = [float(x['Start Time(us)'].strip()) for x in rows]
        durations = [float(x['Duration(us)']) for x in rows]
        ends = [a + b for a, b in zip(starts, durations)]
        latest_start = max(range(8), key=lambda rank: starts[rank])
        earliest_start = min(range(8), key=lambda rank: starts[rank])
        task.append({'index': index, 'name': names[0],
                     'start_skew_us': max(starts) - min(starts),
                     'end_skew_us': max(ends) - min(ends),
                     'latest_start_to_latest_end_us': max(ends) - max(starts),
                     'latest_start_rank': latest_start,
                     'earliest_start_rank': earliest_start,
                     'rank_durations_us': durations,
                     'rank_start_offset_us': [x - min(starts) for x in starts],
                     'rank_end_offset_us': [x - min(ends) for x in ends]})
    def quantiles(key):
        vals = [row[key] for row in task]
        return {'min': min(vals), 'median': statistics.median(vals),
                'p90': sorted(vals)[int(0.9 * (len(vals)-1))], 'max': max(vals)}
    late_counts = [sum(x['latest_start_rank'] == rank for x in task) for rank in range(8)]
    long_tasks = [x for x in task if max(x['rank_durations_us']) > 5000]
    print(json.dumps({
        'status': 'profiled_collective_arrival_alignment',
        'scope': 'Run333 one profiler-perturbed residual-prefill forward; CANN COMMUNICATION task duration includes rank wait and is not link-only latency',
        'all8_order_and_name_parity': True, 'ordered_collectives': 264,
        'start_skew_us': quantiles('start_skew_us'),
        'end_skew_us': quantiles('end_skew_us'),
        'latest_start_to_latest_end_us': quantiles('latest_start_to_latest_end_us'),
        'sum_latest_start_to_latest_end_us': sum(x['latest_start_to_latest_end_us'] for x in task),
        'latest_start_rank_counts': late_counts,
        'tasks_with_any_rank_duration_gt_5ms': len(long_tasks),
        'first_task': task[0], 'last_task': task[-1],
        'largest_start_skew_tasks': sorted(task, key=lambda x: -x['start_skew_us'])[:10],
        'limits': [
            'CANN timestamps are compared within the same host profiler capture; clock alignment must be treated as conditional.',
            'Duration differences and common completion patterns are evidence of arrival/wait contamination. Latest task start is not proven data-ready; the post-latest-start window and its sum are not a calibrated HCCL lower bound or removable time.',
            'Profiler plus explicit NPU synchronization perturbs Host and collective scheduling; no Current or Product bound follows.',
        ],
    }, indent=2))


if __name__ == '__main__':
    main()
