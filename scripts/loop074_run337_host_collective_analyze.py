"""Validate unprofiled all8 Host collective submission sequence and skew."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def describe(values):
    ordered = sorted(values)
    return {'min': ordered[0], 'median': statistics.median(ordered),
            'p90': ordered[int(.9 * (len(ordered)-1))], 'max': ordered[-1]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir
    for label, count in (('warmup48', 48), ('measured12', 12)):
        data = json.loads((root / f'{label}.json').read_text())
        summary = data['summary']
        if (summary['n'], summary['success'], summary['fail'],
            summary['concurrency'], summary['max_tokens']) != (count, count, 0, 12, 1024):
            raise RuntimeError(f'{label} client gate invalid')
        if any(x['output_tokens'] != 1024 for x in data['requests']):
            raise RuntimeError(f'{label} output gate invalid')
    reports = [json.loads((root / 'runtime' / f'rank{rank}_cohort5.json').read_text())
               for rank in range(8)]
    if any(not x['pass'] or x['target_graph_mode'] != 'FULL' or
           x['generated_output_counts'] != [1024] * 12 for x in reports):
        raise RuntimeError('all8 Runtime gate invalid')
    ranks = [json.loads((root / 'marks' / f'rank{rank}.json').read_text())
             for rank in range(8)]
    if any(x['rank'] != rank or x['served_cohorts'] != 4 for rank, x in enumerate(ranks)):
        raise RuntimeError('rank/cohort capture mismatch')
    pads = [x['num_tokens_padded'] for x in ranks]
    if len(set(pads)) != 1:
        raise RuntimeError(f'first forward shape mismatch {pads}')
    counts = [len(x['collectives']) for x in ranks]
    kind_seqs = [[event[0] for event in x['collectives']] for x in ranks]
    seqs = [[(event[0], event[5], event[6]) for event in x['collectives']]
            for x in ranks]
    kind_sequence_parity = all(x == kind_seqs[0] for x in kind_seqs[1:])
    sequence_parity = all(x == seqs[0] for x in seqs[1:])
    expected = collections.Counter({'reduce_scatter': 87, 'all_gather': 134,
                                    'all_to_all': 43})
    observed = collections.Counter(x[0] for x in ranks[0]['collectives'])
    reference_complete = counts == [264] * 8 and sequence_parity and observed == expected
    captured_scope_complete = len(set(counts)) == 1 and sequence_parity
    per_rank = []
    for rank, x in enumerate(ranks):
        wall = x['forward_end_ns'] - x['forward_start_ns']
        per_rank.append({'rank': rank, 'forward_wall_ms': wall / 1e6,
                         'forward_thread_cpu_ms': x['forward_thread_cpu_ns'] / 1e6,
                         'thread_cpu_over_wall': x['forward_thread_cpu_ns'] / wall,
                         'collective_count': counts[rank]})
    report = {
        'status': 'complete_reference_collective_capture' if reference_complete else
                  'complete_wrapped_scope_partial_reference_coverage' if captured_scope_complete else
                  'partial_unprofiled_host_collective_capture',
        'scope': 'Run337 one first warmed padded88 prefill forward per rank; no profiler and no new synchronize; Host timestamps only',
        'all8_client_and_runtime_pass': True,
        'shape_num_tokens_padded': pads[0], 'counts_by_rank': counts,
        'all8_kind_sequence_parity': kind_sequence_parity,
        'all8_kind_bytes_dtype_sequence_parity': sequence_parity,
        'profiled_reference_collective_count': 264,
        'wrapped_scope_fraction_of_profiled_reference': counts[0] / 264,
        'kind_counts_rank0': dict(observed),
        'forward_start_spread_ms': (max(x['forward_start_ns'] for x in ranks) -
                                    min(x['forward_start_ns'] for x in ranks)) / 1e6,
        'forward_end_spread_ms': (max(x['forward_end_ns'] for x in ranks) -
                                  min(x['forward_end_ns'] for x in ranks)) / 1e6,
        'ranks': per_rank,
        'limits': [
            'Host collective wrapper entry is not proven input data-ready or device task start; its wall can include CPU enqueue/wait.',
            'No device/HCCL service or complete prefill/seed path measured; one diagnostic cohort, not formal E2E.',
            'The wrappers add Python timestamp/list-append overhead. Compare only with the perturbation caveat.',
            'The wrapper omits 43 AllGather calls relative to Run333 CANN HCCL sequence; captured Host marks are a subset, not a complete HCCL arrival trace.',
        ],
    }
    if captured_scope_complete:
        groups = [[rank['collectives'][i] for rank in ranks] for i in range(counts[0])]
        starts = [[x[1] for x in group] for group in groups]
        ends = [[x[2] for x in group] for group in groups]
        report.update({
            'collective_entry_spread_us': describe([(max(x)-min(x))/1000 for x in starts]),
            'collective_return_spread_us': describe([(max(x)-min(x))/1000 for x in ends]),
            'latest_entry_rank_counts': [sum(max(range(8),key=lambda rank: s[rank]) == rank
                                             for s in starts) for rank in range(8)],
            'first_last_entry_spread_us': [(max(s)-min(s))/1000 for s in (starts[0], starts[-1])],
            'per_kind_input_bytes': {kind: sorted(set(x[5] for x in ranks[0]['collectives']
                                                       if x[0] == kind))
                                     for kind in expected},
            'first_collective_entry_offset_from_forward_start_ms': [
                (x['collectives'][0][1] - x['forward_start_ns']) / 1e6 for x in ranks],
        })
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
