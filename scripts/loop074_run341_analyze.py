#!/usr/bin/env python3
"""All-rank natural warm prefill call timing; never promotes it to E2E cost."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    root = args.run_dir
    marks = [json.loads(line) for line in (root / 'marks' / 'rank0.jsonl').read_text().splitlines() if line]
    mark_groups = []
    group = []
    prompt_lengths = {}
    for mark in marks:
        if mark['kind'] == 'runtime_built':
            mark_groups.append(group)
            group = []
        elif mark['kind'] == 'execute_entry':
            group.append(mark)
            for req in mark['new']:
                prompt_lengths[req['req_id']] = req['prompt_len']
    assert len(mark_groups) == 4
    cohorts = []
    for cohort in range(5, 9):
        ranks = [json.loads((root / 'marks' / f'rank{rank}_cohort{cohort}_calls.json').read_text())
                 for rank in range(8)]
        assert all(x['rank'] == rank and x['cohort'] == cohort for rank, x in enumerate(ranks))
        count = len(ranks[0]['calls'])
        assert count > 0 and all(len(x['calls']) == count for x in ranks)
        calls = []
        for i in range(count):
            group = [x['calls'][i] for x in ranks]
            shapes = {x['num_tokens_padded'] for x in group}
            assert len(shapes) == 1, (cohort, i, shapes)
            walls = [(x['host_end_ns'] - x['host_start_ns']) / 1e6 for x in group]
            cpus = [x['thread_cpu_ns'] / 1e6 for x in group]
            devices = [x['current_stream_elapsed_ms'] for x in group]
            mark_count = group[0]['execute_mark_count']
            assert all(x['execute_mark_count'] == mark_count for x in group)
            mark = mark_groups[cohort - 5][mark_count - 1]
            assert mark['kind'] == 'execute_entry'
            assert mark['t_ns'] <= min(x['host_start_ns'] for x in group)
            current = mark['new'] + mark['cached']
            prompt_scheduled = sum(min(x['scheduled_tokens'], max(
                0, prompt_lengths[x['req_id']] - x['num_computed_tokens']))
                for x in current)
            calls.append({
                'call_index': i, 'num_tokens_padded': shapes.pop(),
                'new_request_count': len(mark['new']),
                'cached_request_count': len(mark['cached']),
                'prompt_residual_scheduled_tokens': prompt_scheduled,
                'host_entry_spread_ms': (max(x['host_start_ns'] for x in group)
                                         - min(x['host_start_ns'] for x in group)) / 1e6,
                'host_return_spread_ms': (max(x['host_end_ns'] for x in group)
                                          - min(x['host_end_ns'] for x in group)) / 1e6,
                'max_rank_host_wall_ms': max(walls),
                'min_rank_host_wall_ms': min(walls),
                'max_rank_current_stream_event_ms': max(devices),
                'median_rank_current_stream_event_ms': statistics.median(devices),
                'median_rank_thread_cpu_over_host_wall': statistics.median(
                    cpu / wall for cpu, wall in zip(cpus, walls)),
            })
        cohorts.append({
            'cohort': cohort, 'all_rank_call_count': count,
            'calls': calls,
            'new_request_group_sizes': [x['new_request_count'] for x in calls
                                        if x['new_request_count']],
            'sum_call_max_rank_host_wall_ms': sum(x['max_rank_host_wall_ms'] for x in calls),
            'sum_call_max_rank_current_stream_event_ms': sum(
                x['max_rank_current_stream_event_ms'] for x in calls),
        })
    out = {
        'status': 'valid_original_path_host_and_current_stream_call_marks',
        'cohorts': cohorts,
        'limits': [
            'The NPU event records only the current stream around each _model_forward; unjoined HCCL/side-stream work can extend beyond it.',
            'Individual maximum-rank durations occur on possibly different ranks and their sum is not a critical path.',
            'The same cohort preparation includes multiple prompt, spec and decode calls; each shape is not automatically prompt residual work.',
            'This measures original cohort preparation, not the incremental cost or contention of an early b2-b4 refill.',
            'Instrumented 48+48 diagnostic carrier is not repeated formal E2E.',
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
