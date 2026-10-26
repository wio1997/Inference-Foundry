"""Read-only fixed-cohort versus FIFO-slot refill scheduling sensitivity.

Input is original Run287 FULL Graph ownership JSONL, not a replay. Reported
counterfactuals hold each request's observed decode duration fixed and omit
prefill/seed/Host/state costs except the explicit per-refill delay parameter.
They are conditional scheduling screens, never Hardware or Product bounds.
"""

import argparse
import heapq
import json
import math
import random
from pathlib import Path


def load_cohort(root, cohort):
    by_rank = []
    for rank in range(8):
        path = root / 'ownership' / f'rank{rank}_cohort{cohort}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows or [x['cycle'] for x in rows] != list(range(len(rows))):
            raise RuntimeError(f'invalid cycle sequence {path}')
        by_rank.append(rows)
    count = len(by_rank[0])
    for rank in range(1, 8):
        if len(by_rank[rank]) != count:
            raise RuntimeError(f'cycle count mismatch cohort{cohort} rank{rank}')
        for cycle in range(count):
            for field in ('active_mask', 'emitted_token_count'):
                if by_rank[rank][cycle][field] != by_rank[0][cycle][field]:
                    raise RuntimeError(f'{field} mismatch cohort{cohort} cycle{cycle} rank{rank}')
    masks = [x['active_mask'] for x in by_rank[0]]
    if any(len(x) != 12 for x in masks) or masks[0] != [1] * 12:
        raise RuntimeError(f'invalid fixed c12 entry cohort{cohort}')
    immediate = []
    mirror = []
    for slot in range(12):
        first_parked = next((cycle for cycle, mask in enumerate(masks)
                             if mask[slot] == 0), count)
        first_complete = next((row['cycle'] for row in by_rank[0]
                               if row['emitted_token_count'][slot] >= 1024), None)
        if first_complete is None or min(first_complete + 1, count) != first_parked:
            raise RuntimeError(f'Host-mirror release mismatch cohort{cohort} slot{slot}')
        if first_parked < count and any(mask[slot] for mask in masks[first_parked:]):
            raise RuntimeError(f'slot reactivated cohort{cohort} slot{slot}')
        immediate.append(first_complete)
        mirror.append(first_complete + 1)
    return {'cohort': cohort, 'cycles': count,
            'immediate_completion_cycles': immediate,
            'host_mirror_release_cycles': mirror,
            'free_slot_cycles': sum(count - min(x, count) for x in mirror),
            'first_release_cycle': min(mirror),
            'last_release_cycle': max(mirror),
            'rank_mask_and_emitted_parity': True}


def fifo_makespan(durations, delay):
    queue = list(durations)
    if len(queue) < 12:
        raise RuntimeError('need at least 12 requests')
    heap = [(queue[slot], slot) for slot in range(12)]
    heapq.heapify(heap)
    starts = [0] * 12
    for duration in queue[12:]:
        finished, slot = heapq.heappop(heap)
        start = finished + delay
        starts.append(start)
        heapq.heappush(heap, (start + duration, slot))
    return {'cycles': max(end for end, _ in heap),
            'refill_start_cycles': starts[12:],
            'last_refill_start_cycle': max(starts[12:])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cohorts', type=int, default=4)
    args = parser.parse_args()
    if args.cohorts != 4:
        raise RuntimeError('freeze this screen to four Run287 warmup48 cohorts')
    cohorts = [load_cohort(args.run_dir, i) for i in range(args.cohorts)]
    client = json.loads((args.run_dir / 'warmup48.json').read_text())
    if (client['summary']['n'] != 48 or client['summary']['success'] != 48 or
            client['summary']['concurrency'] != 12 or
            sorted(x['i'] for x in client['requests']) != list(range(48)) or
            any(x['output_tokens'] != 1024 for x in client['requests'])):
        raise RuntimeError('Run287 warmup48 client contract mismatch')
    wave_edges = []
    for cohort in cohorts:
        runtime = [json.loads((args.run_dir / 'runtime' /
                              f'rank{rank}_cohort{cohort["cohort"] + 1}.json').read_text())
                   for rank in range(8)]
        if any(not x['pass'] or x['cycles'] != cohort['cycles'] or
               x['target_graph_mode'] != 'FULL' or
               x['generated_output_counts'] != [1024] * 12 or
               len(x['req_ids']) != 12 for x in runtime):
            raise RuntimeError(f'Run287 runtime contract failed cohort{cohort["cohort"]}')
        wave = [x for x in client['requests']
                if x['i'] // 12 == cohort['cohort']]
        wave_edges.append({'cohort': cohort['cohort'],
                           'client_first_start': min(x['start'] for x in wave),
                           'client_last_start': max(x['start'] for x in wave),
                           'client_first_end': min(x['end'] for x in wave),
                           'client_last_end': max(x['end'] for x in wave)})
    durations = [duration for cohort in cohorts
                 for duration in cohort['host_mirror_release_cycles']]
    instant_durations = [duration for cohort in cohorts
                         for duration in cohort['immediate_completion_cycles']]
    observed_cycles = sum(x['cycles'] for x in cohorts)
    occupied = sum(durations)
    if occupied + sum(x['free_slot_cycles'] for x in cohorts) != 12 * observed_cycles:
        raise RuntimeError('slot-cycle conservation failed')
    sensitivity = {str(delay): fifo_makespan(durations, delay)
                   for delay in (0, 1, 2, 4, 8, 16, 24, 32, 48, 64)}
    instant = fifo_makespan(instant_durations, 0)
    rng = random.Random(741)
    sampled_order = []
    for _ in range(512):
        reordered = []
        for cohort in cohorts:
            part = cohort['host_mirror_release_cycles'].copy()
            rng.shuffle(part)
            reordered.extend(part)
        sampled_order.append(fifo_makespan(reordered, 0)['cycles'])
    print(json.dumps({
        'status': 'conditional_offline_scheduling_screen',
        'source': str(args.run_dir / 'ownership' / 'rank*_cohort0..3.jsonl'),
        'scope': 'Run287 original FULL Graph warmup48 4x12 requests; not formal Run99',
        'assumption': 'same per-request decode-cycle durations after FIFO refill, no route/acceptance/contention change; incremental refill delay occupies only the newly free slot, with no cross-rank global prefill pause or state/Host/Graph overhead beyond delay',
        'cohorts': cohorts,
        'client_wave_edges': wave_edges,
        'client_to_runtime_slot_mapping': 'unknown; client JSON lacks server req_id',
        'observed_cohort_wave_cycles': observed_cycles,
        'observed_occupied_slot_cycles': occupied,
        'observed_free_slot_cycles': 12 * observed_cycles - occupied,
        'observed_free_slot_fraction': (12 * observed_cycles - occupied) /
                                       (12 * observed_cycles),
        'instant_completion_fixed_duration_floor_cycles': max(
            max(instant_durations), math.ceil(sum(instant_durations) / 12)),
        'instant_completion_zero_incremental_cost_fifo_cycles': instant['cycles'],
        'host_mirror_fixed_duration_floor_cycles': max(
            max(durations), math.ceil(occupied / 12)),
        'fifo_per_slot_delay_cycles': sensitivity,
        'within_wave_client_slot_order_sensitivity_512_shuffles': {
            'seed': 741, 'min_cycles': min(sampled_order),
            'median_cycles': sorted(sampled_order)[len(sampled_order) // 2],
            'max_cycles': max(sampled_order),
            'sampling_scope': 'permute durations within each 12-request cohort, preserve cohort order; sample range is not a rigorous bound'},
        'not_a_bound_reasons': [
            'new 32K prefill, DSpark seed, KV/state reset and graph-key setup omitted or modeled only as per-slot delay',
            'refill can change all-rank resource contention, route/acceptance and each request duration',
            'Run287 diagnostic warmup trajectory differs from formal Run99 samples',
            'client publication/admission and 48-request output drain are not replayed',
        ],
    }, indent=2))


if __name__ == '__main__':
    main()
