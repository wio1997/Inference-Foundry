"""Read-only Run287 refill granularity and prior prefill evidence audit.

All scheduling numbers hold the original per-request decode durations fixed.
The prefill evidence is descriptive and is never injected as a per-slot delay.
"""

import argparse
import heapq
import json
from pathlib import Path


def microbatch_makespan(durations, batch):
    heap = [(durations[slot], slot) for slot in range(12)]
    heapq.heapify(heap)
    next_request = 12
    free = []
    batches = []
    while next_request < len(durations):
        finish, slot = heapq.heappop(heap)
        free.append((finish, slot))
        count = min(batch, len(durations) - next_request)
        if len(free) < count:
            continue
        start = max(x[0] for x in free)
        chosen = free[:count]
        free = free[count:]
        for _, slot in chosen:
            heapq.heappush(heap, (start + durations[next_request], slot))
            next_request += 1
        batches.append({'start_cycle': start, 'size': count})
    return {'cycles': max([x[0] for x in heap] + [x[0] for x in free]),
            'refill_batches': len(batches), 'batch_sizes': [x['size'] for x in batches],
            'batch_starts': [x['start_cycle'] for x in batches]}


def forward_rows(root, tag):
    ranks = []
    for rank in range(8):
        path = root / 'forward' / f'rank{rank}_{tag}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        ranks.append(rows)
    if len({len(rows) for rows in ranks}) != 1:
        raise RuntimeError(f'{tag} rank call count mismatch')
    prefill = []
    for index in range(len(ranks[0])):
        calls = [rows[index] for rows in ranks]
        signatures = {(x['num_tokens_padded'], x['num_actual_tokens'],
                       x['num_reqs'], x['mode']) for x in calls}
        if len(signatures) != 1:
            raise RuntimeError(f'{tag} call{index} rank shape mismatch')
        call = calls[0]
        if call['mode'] != 'NONE':
            continue
        prefill.append({'tokens': call['num_actual_tokens'],
                        'requests': call['num_reqs'],
                        'max_rank_forward_wall_ms': max(x['forward_wall_ms'] for x in calls),
                        'min_rank_forward_wall_ms': min(x['forward_wall_ms'] for x in calls)})
    return {'prefill_calls': prefill,
            'sum_max_rank_forward_wall_ms': sum(x['max_rank_forward_wall_ms'] for x in prefill),
            'sum_tokens': sum(x['tokens'] for x in prefill),
            'all_rank_call_shape_parity': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--foundry', type=Path, required=True)
    args = p.parse_args()
    base = args.foundry / 'evidence'
    prior = json.loads((base / '20260926_loop074_refill/run330/counterfactual.json').read_text())
    durations = [x for cohort in prior['cohorts']
                 for x in cohort['host_mirror_release_cycles']]
    assert len(durations) == 48 and prior['observed_cohort_wave_cycles'] == 1203
    granularity = {str(batch): microbatch_makespan(durations, batch)
                   for batch in (1, 2, 3, 4, 6, 12)}
    assert granularity['1']['cycles'] == 1122
    assert granularity['12']['cycles'] == 1203
    run188 = base / '20260925_loop048_prefill/run188'
    forward = {tag: forward_rows(run188, tag) for tag in ('A', 'B', 'A2')}
    analysis188 = json.loads((run188 / 'analysis.json').read_text())
    for tag, measured in forward.items():
        old = analysis188['phase_summary'][tag]
        assert old['prefill_calls'] == len(measured['prefill_calls'])
        assert old['prefill_tokens'] == [x['tokens'] for x in measured['prefill_calls']]
        assert abs(old['max_rank_prefill_forward_wall_sum_s'] * 1000 -
                   measured['sum_max_rank_forward_wall_ms']) < 0.001
    run239 = json.loads((base / '20260926_loop059_boundary/run239/phase_analysis.json').read_text())
    waves = [{'wave': x['wave'], 'prefill_to_handoff_s':
              x['phases']['first_execute_to_handoff_s'], 'cycles': x['cycles']}
             for x in run239['waves']]
    boundary_root = base / '20260926_loop059_boundary/run239/boundary'
    boundary = []
    for cohort in (5, 6, 7, 8):
        ranks = [json.loads((boundary_root / f'rank{rank}_cohort{cohort}.json').read_text())
                 for rank in range(8)]
        sequences = [[x['scheduled_tokens'] for x in r['calls']] for r in ranks]
        if any(x != sequences[0] for x in sequences[1:]):
            raise RuntimeError(f'Run239 cohort{cohort} scheduled-token rank mismatch')
        if sequences[0][:2] != [96, 0]:
            raise RuntimeError(f'Run239 cohort{cohort} boundary prefix changed')
        if len({r['handoff_ns'] > r['calls'][-1]['t_ns'] for r in ranks}) != 1:
            raise RuntimeError(f'Run239 cohort{cohort} handoff order mismatch')
        boundary.append({'cohort': cohort, 'all_rank_scheduled_token_parity': True,
                         'scheduled_tokens_by_execute_call': sequences[0],
                         'current_wave_calls_after_prior_96_and_zero': sequences[0][2:],
                         'max_rank_current_wave_first_call_to_handoff_s': max(
                             (r['handoff_ns'] - r['calls'][2]['t_ns']) / 1e9
                             for r in ranks),
                         'max_rank_last_call_to_handoff_s': max(
                             (r['handoff_ns'] - r['calls'][-1]['t_ns']) / 1e9
                             for r in ranks)})
    print(json.dumps({
        'status': 'retrospective_conditional_screen',
        'source_run330': 'Run287 original all8 ownership; Run330 validated shape and client',
        'granularity': granularity,
        'run188_all_rank_prefill_forward': forward,
        'run239_original_wave_prepare': waves,
        'run239_all_rank_execute_boundaries': boundary,
        'known': [
            'warmed original prefill forward processed 88-264 tokens/call in Run188 A, not 32K/call',
            'Run188 B consolidated 12 requests into one 1000-token forward',
            'Run239 first-execute-to-handoff includes work outside prefill forward',
        ],
        'unknown': [
            'per-request prefix-hit and exact residual scheduled tokens at refill arrival',
            'all8 device critical path from request arrival through prefill and DSpark seed readiness',
            'incremental cost versus existing four cohort preparations, including decode contention',
            'request-to-runtime-slot mapping and change in acceptance/route after refill',
        ],
        'interpretation': 'batch sizes are fixed-duration zero-incremental-cost scheduling screens, not achievable latency or TPS bounds; Run188/239 costs are different trajectories and cannot be subtracted directly',
    }, indent=2))


if __name__ == '__main__':
    main()
