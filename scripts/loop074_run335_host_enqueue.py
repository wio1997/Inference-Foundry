"""Match Host HCCL enqueue/dequeue to 264 all-rank CANN collective tasks."""

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path


def one_rank(root, rank):
    dirs = list((root / 'profiler' / f'rank{rank}').glob('*/ASCEND_PROFILER_OUTPUT'))
    if len(dirs) != 1:
        raise RuntimeError(f'rank{rank} profiler output count')
    trace = json.loads((dirs[0] / 'trace_view.json').read_text())
    enqueue = [x for x in trace if x.get('cat') == 'enqueue' and
               x.get('name', '').startswith('Enqueue@Hccl')]
    dequeue_by_id = {x.get('args', {}).get('correlation_id'): x for x in trace
                     if x.get('cat') == 'dequeue' and
                     x.get('name', '').startswith('Dequeue@Hccl')}
    kernels = [x for x in csv.DictReader((dirs[0] / 'kernel_details.csv').open())
               if x['Accelerator Core'] == 'COMMUNICATION']
    if (len(enqueue), len(dequeue_by_id), len(kernels)) != (264, 264, 264):
        raise RuntimeError(f'rank{rank} count mismatch')
    rows = []
    for index, (host, kernel) in enumerate(zip(enqueue, kernels)):
        cid = host['args']['correlation_id']
        deq = dequeue_by_id[cid]
        kind = host['name'].removeprefix('Enqueue@Hccl').lower()
        kernel_kind = kernel['Name'].split('__')[0].removeprefix('hcom_').lower()
        if kind != kernel_kind or deq['name'] != host['name'].replace('Enqueue@', 'Dequeue@'):
            raise RuntimeError(f'rank{rank} task{index} type mismatch {kind}/{kernel_kind}')
        rows.append({'kind': kind, 'enqueue_us': float(host['ts']),
                     'dequeue_us': float(deq['ts']),
                     'device_start_us': float(kernel['Start Time(us)'].strip()),
                     'device_end_us': float(kernel['Start Time(us)'].strip()) +
                                      float(kernel['Duration(us)'])})
    return rows


def stats(values):
    ordered = sorted(values)
    return {'min': ordered[0], 'median': statistics.median(ordered),
            'p90': ordered[int(0.9 * (len(ordered)-1))], 'max': ordered[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    ranks = [one_rank(args.run_dir, rank) for rank in range(8)]
    if any([x['kind'] for x in ranks[rank]] != [x['kind'] for x in ranks[0]]
           for rank in range(1, 8)):
        raise RuntimeError('all-rank HCCL type sequence mismatch')
    rows = []
    for index in range(264):
        group = [rank[index] for rank in ranks]
        enq = [x['enqueue_us'] for x in group]
        dev = [x['device_start_us'] for x in group]
        end = [x['device_end_us'] for x in group]
        rows.append({'index': index, 'kind': group[0]['kind'],
                     'latest_enqueue_rank': max(range(8), key=lambda rank: enq[rank]),
                     'latest_device_start_rank': max(range(8), key=lambda rank: dev[rank]),
                     'enqueue_spread_us': max(enq) - min(enq),
                     'device_start_spread_us': max(dev) - min(dev),
                     'device_end_spread_us': max(end) - min(end),
                     'latest_enqueue_to_latest_device_start_us': max(dev) - max(enq)})
    print(json.dumps({
        'status': 'profiled_host_collective_enqueue_alignment',
        'scope': 'Run333 profiler-perturbed one warmed residual-prefill forward; all8 Host Enqueue/Dequeue and CANN task clocks within same capture',
        'all8_264_group_kind_correlation_parity': True,
        'latest_enqueue_rank_counts': [sum(x['latest_enqueue_rank'] == rank for x in rows)
                                       for rank in range(8)],
        'latest_device_start_rank_counts': [sum(x['latest_device_start_rank'] == rank for x in rows)
                                            for rank in range(8)],
        'latest_enqueue_and_device_start_same_rank_count': sum(
            x['latest_enqueue_rank'] == x['latest_device_start_rank'] for x in rows),
        'enqueue_spread_us': stats([x['enqueue_spread_us'] for x in rows]),
        'latest_enqueue_to_latest_device_start_us': stats([
            x['latest_enqueue_to_latest_device_start_us'] for x in rows]),
        'rank4_enqueue_to_dequeue_us': stats([
            x['dequeue_us'] - x['enqueue_us'] for x in ranks[4]]),
        'kind_counts': dict(collections.Counter(x['kind'] for x in rows)),
        'limits': [
            'Enqueue/dequeue/device timestamps are aligned only within the same host trace; this is a profiled execution, not unprofiled Current.',
            'Enqueue is submission progress, not proof of input data-ready or an intrinsic HCCL start.',
            'No profiler-free rank arrival or complete prefill/seed DAG is established; no bound or Product saving follows.',
        ],
    }, indent=2))


if __name__ == '__main__':
    main()
