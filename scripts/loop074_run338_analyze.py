#!/usr/bin/env python3
"""Validate and summarize all-rank private warm-prefill full-MoE execution gate."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir
    for label, n in (('warmup48', 48), ('measured12', 12)):
        data = json.loads((root / f'{label}.json').read_text())
        s = data['summary']
        if (s['n'], s['success'], s['fail'], s['concurrency'], s['max_tokens']) != (n, n, 0, 12, 1024):
            raise RuntimeError(f'{label} client contract failed')
        if any(x['output_tokens'] != 1024 for x in data['requests']):
            raise RuntimeError(f'{label} completion length failed')
    runtime = [json.loads((root / 'runtime' / f'rank{i}_cohort5.json').read_text())
               for i in range(8)]
    if any(not x['pass'] or x['target_graph_mode'] != 'FULL' or
           x['generated_output_counts'] != [1024] * 12 for x in runtime):
        raise RuntimeError('eight-rank Runtime FULL Graph gate failed')
    reports = [json.loads((root / 'fixture' / f'rank{i}.json').read_text()) for i in range(8)]
    header = [{'rank': x['rank'], 'stage': x['stage'], 'pass': x['pass'],
               'input_shape': x.get('input_shape'),
               'context_num_tokens': x.get('context_num_tokens'),
               'context_padded_num_tokens': x.get('context_padded_num_tokens'),
               'context_pad_size': x.get('context_pad_size'),
               'context_mode': x.get('context_mode'),
               'capture_wall_ms': x.get('capture_wall_ms'),
               'error': x.get('error'), 'traceback': x.get('traceback')}
              for x in reports]
    out = {'scope': 'private original complete MoE eager A/Graph B/eager A2, first measured warm padded88 prefill layer4',
           'all8_client_runtime_gate': True, 'ranks': header,
           'formal_e2e_result': False, 'numeric_product_bound': False}
    complete = all(x['rank'] == i and x['stage'] == 'complete' and x['pass']
                   and len(x['samples']) == 39 for i, x in enumerate(reports))
    if not complete:
        out['status'] = 'incomplete_fixture'
        print(json.dumps(out, indent=2))
        return
    output_diff = []
    pairs = []
    for t in range(3, 13):
        arms = {}
        for arm_idx, arm in enumerate(('A', 'B', 'A2')):
            rows = [x['samples'][t*3+arm_idx] for x in reports]
            if any(x['triplet'] != t or x['arm'] != arm or x['warmup'] for x in rows):
                raise RuntimeError('all-rank sample order mismatch')
            arms[arm] = {
                'max_rank_stream_ms': max(x['stream_endpoint_ms'] for x in rows),
                'max_rank_host_submit_ms': max(x['host_submit_ms'] for x in rows),
                'rendezvous_host_complete_ms': (max(x['host_completed_ns'] for x in rows) -
                                                min(x['host_start_ns'] for x in rows))/1e6,
            }
        row = {'triplet': t, 'arms': arms}
        for metric in ('max_rank_stream_ms', 'max_rank_host_submit_ms', 'rendezvous_host_complete_ms'):
            row[f'B_minus_control_mean_{metric}'] = (arms['B'][metric] -
                                                      (arms['A'][metric]+arms['A2'][metric])/2)
            row[f'B_faster_both_{metric}'] = (arms['B'][metric] < arms['A'][metric] and
                                              arms['B'][metric] < arms['A2'][metric])
        pairs.append(row)
    for rank, x in enumerate(reports):
        values = [s['comparisons'] for s in x['samples'] if 'comparisons' in s and not s['warmup']]
        output_diff.append({'rank': rank,
                            'A_A2_max_abs_max': max(c['A_A2']['max_abs'] for c in values),
                            'A_B_max_abs_max': max(c['A_B']['max_abs'] for c in values),
                            'A_A2_rms_max': max(c['A_A2']['rms'] for c in values),
                            'A_B_rms_max': max(c['A_B']['rms'] for c in values),
                            'A_B_all_finite': all(c['A_B']['finite'] for c in values)})
    out.update({'status': 'complete_private_fixture', 'pairs': pairs,
                'numeric_screen': output_diff,
                'paired_summary': {
                    metric: {'median_B_minus_control_mean_ms': statistics.median(
                                x[f'B_minus_control_mean_{metric}'] for x in pairs),
                             'strict_B_faster_both_count': sum(
                                x[f'B_faster_both_{metric}'] for x in pairs)}
                    for metric in ('max_rank_stream_ms', 'max_rank_host_submit_ms',
                                   'rendezvous_host_complete_ms')},
                'limits': ['Stream event endpoint can include Host enqueue gaps; it is not isolated device compute.',
                           'Barrier, synchronize and private capture perturb service; this is not Product E2E.',
                           'Original warmed prefill and Graph capture shape/state churn must be assessed before live integration.']})
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
