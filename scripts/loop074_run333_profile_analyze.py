"""CANN step/device coverage across eight simultaneous residual-prefill calls."""

import argparse
import collections
import csv
import json
from pathlib import Path


def rank_profile(root, rank):
    capture = json.loads((root / 'profiler' / f'rank{rank}_capture.json').read_text())
    dirs = list((root / 'profiler' / f'rank{rank}').glob('*/ASCEND_PROFILER_OUTPUT'))
    if len(dirs) != 1:
        raise RuntimeError(f'rank{rank} profiler output dirs {len(dirs)}')
    path = dirs[0]
    kernels = list(csv.DictReader((path / 'kernel_details.csv').open()))
    steps = list(csv.DictReader((path / 'step_trace_time.csv').open()))
    if len(steps) != 1 or len(kernels) < 1000:
        raise RuntimeError(f'rank{rank} profiler data missing')
    step = steps[0]
    intervals = []
    for row in kernels:
        start = float(row['Start Time(us)'].strip())
        end = start + float(row['Duration(us)'])
        intervals.append((start, end, row['Accelerator Core'] == 'COMMUNICATION'))
    comm_pairs = {(start, end) for start, end, comm in intervals if comm}
    duplicate = sum(1 for start, end, comm in intervals if not comm and
                    (start, end) in comm_pairs)
    intervals = [(start, end, comm) for start, end, comm in intervals
                 if comm or (start, end) not in comm_pairs]
    events = []
    for start, end, comm in intervals:
        events.append((start, 1, int(comm)))
        events.append((end, -1, -int(comm)))
    events.sort()
    active = communication = 0
    previous = events[0][0]
    covered = collections.Counter()
    for instant, delta_active, delta_comm in events:
        if instant > previous:
            category = ('free' if active == 0 else
                        'overlap' if 0 < communication < active else
                        'communication' if communication else 'compute')
            covered[category] += instant - previous
        active += delta_active
        communication += delta_comm
        previous = instant
    stage_us = float(step['Stage'])
    if abs(sum(covered.values()) - stage_us) > 200:
        raise RuntimeError(f'rank{rank} CANN stage/interval discrepancy')
    return {'rank': rank, 'capture': capture,
            'kernel_count': len(kernels), 'hccl_aiv_duplicate_intervals_removed': duplicate,
            'step_trace_us': {key: float(step[key]) for key in
                              ('Computing', 'Communication(Not Overlapped)',
                               'Overlapped', 'Communication', 'Free', 'Stage', 'Preparing')},
            'device_union_us': dict(covered),
            'device_free_fraction': covered['free'] / stage_us,
            'trace_start_us': events[0][0], 'trace_end_us': events[-1][0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir
    clients = {key: json.loads((root / f'{key}.json').read_text())
               for key in ('warmup48', 'measured12')}
    for key, n in (('warmup48', 48), ('measured12', 12)):
        summary = clients[key]['summary']
        if (summary['n'], summary['success'], summary['fail'],
            summary['concurrency'], summary['max_tokens']) != (n, n, 0, 12, 1024):
            raise RuntimeError(f'{key} client gate failed')
        if any(x['output_tokens'] != 1024 for x in clients[key]['requests']):
            raise RuntimeError(f'{key} output gate failed')
    reports = [json.loads((root / 'runtime' / f'rank{rank}_cohort5.json').read_text())
               for rank in range(8)]
    if any(not x['pass'] or x['target_graph_mode'] != 'FULL' or
           x['generated_output_counts'] != [1024] * 12 for x in reports):
        raise RuntimeError('measured cohort Runtime gate failed')
    ranks = [rank_profile(root, rank) for rank in range(8)]
    if any(x['capture']['served_cohorts'] != 4 or x['capture']['rank'] != x['rank']
           for x in ranks):
        raise RuntimeError('capture rank/cohort mismatch')
    pads = [x['capture']['num_tokens_padded'] for x in ranks]
    if len(set(pads)) != 1:
        raise RuntimeError(f'first prefill shape mismatch {pads}')
    print(json.dumps({
        'status': 'profiled_all8_first_warmed_prefill_forward',
        'scope': 'first measured prefill _model_forward after 48 warmup; explicit per-rank NPU synchronize and profiler analysis perturb call; not formal E2E',
        'shape_num_tokens_padded': pads[0],
        'all8_measured_runtime_full_graph': True,
        'max_rank_step_stage_ms': max(x['step_trace_us']['Stage'] for x in ranks) / 1000,
        'rank_compute_occupancy_ms_range': [min(x['step_trace_us']['Computing'] for x in ranks) / 1000,
                                            max(x['step_trace_us']['Computing'] for x in ranks) / 1000],
        'rank_device_free_fraction_range': [min(x['device_free_fraction'] for x in ranks),
                                            max(x['device_free_fraction'] for x in ranks)],
        'ranks': ranks,
        'limits': [
            'CANN Free is no recorded device kernel and may reflect host enqueue, dependency waits or profiling effects.',
            'This one forward excludes scheduler arrival, other prefill calls, DSpark seed and Runtime build; no complete refill cost or Product bound follows.',
            'Per-rank explicit synchronize and on-trace analysis alter subsequent service trajectory; measured12 client TPS is diagnostic only.',
            'Do not sum rank stage times or treat one-card bandwidth as eight-card attainable capacity.',
        ],
    }, indent=2))


if __name__ == '__main__':
    main()
