"""Account for device replay lanes outside the nonzero target scheduler ranges."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path


FILES = {
    'graph': ('logical_device_map_G867/D0_first_actual_connection_map.json',
              'a284de49baabef729d0caaa60ecafed93bfb66913613bc5ec1b61de0ba488dd5'),
    'shape': ('logical_device_map_G867/D0_query_shape_map.json',
              '131ac32099d8c918c408e4845ff8ad1d07e3886abeb7492e7356a2c35b04afdb'),
    'target': ('profile_capture_G804/derived/D0_target_position_ledger_G1447.json',
               '41422552a67de6aee0a58616ec3828111f13c8ec59ffa6638a1c98fce87f707e'),
}
DRAFT_WIDTH = 5


def load(root):
    evidence = {}
    for key, (name, digest) in FILES.items():
        raw = (root / name).read_bytes()
        assert sha256(raw).hexdigest() == digest, name
        evidence[key] = json.loads(raw)
    return evidence


def replay_capacities(shape, operators, outside):
    capacities = []
    for model, repeats in sorted(outside.items()):
        rows = [row for row in shape['actual_SparseFlashMla_shapes'] if row['model_id'] == model]
        width, = {int(row['input_shapes'].strip('"').split(',')[0]) for row in rows}
        count = sum(row['occurrences'] for row in rows)
        assert count == 3 * repeats
        capacity, remainder = divmod(width, DRAFT_WIDTH)
        assert not remainder and capacity > 0
        sparse, = [row for row in operators[model] if row['op_type'] == 'SparseFlashMla']
        assert sparse['count'] == count
        capacities.append({
            'model_id': model, 'replays_with_direct_device_correlation': repeats,
            'query_lanes_per_replay_in_SparseFlashMla': width, 'request_capacity_at_draft_width': capacity,
            'SparseFlashMla_task_count': count,
            'SparseFlashMla_task_duration_sum_ns': sparse['sum_duration_ns_not_walltime'],
            'all_named_compute_task_duration_sum_ns': sum(
                row['sum_duration_ns_not_walltime'] for row in operators[model]),
        })
    return sorted(capacities, key=lambda row: row['request_capacity_at_draft_width'])


def uniquely_saturated_buckets(logical_requests, physical):
    """Check uniqueness of aggregate counts under one non-underfilled replay per target step."""
    pending = dict(logical_requests)
    matched = []
    for shape in physical:
        capacity = shape['request_capacity_at_draft_width']
        assign = {requests: count for requests, count in pending.items() if requests <= capacity}
        if sum(assign.values()) != shape['replays_with_direct_device_correlation']:
            return None
        for size in assign:
            del pending[size]
        padding = sum((capacity - size) * DRAFT_WIDTH * count for size, count in assign.items())
        matched.append({'physical_request_capacity': capacity, 'target_request_count_to_steps': assign,
                        'conditional_extra_shape_lanes': padding})
    return matched if not pending else None


def summarize(data):
    graph = data['graph']['graph_map']['result.json']
    shape = data['shape']
    target = data['target']
    assert graph['RI_execute_calls'] == graph['RI_calls_directly_correlated_with_device_model'] == 1812
    assert graph['nonzero_logical_steps'] == graph['RI_calls_enclosed_by_same_thread_scheduler_range'] == 906
    assert graph['calls_per_nonzero_iteration'] == {'1': 906}
    assert shape['all1812_RI_calls_direct_device_correlated'] and shape['inside_scheduler_ranges_906_each_once']
    assert target['capture'] == 'G804' and target['position_and_work_counts']['steps'] == 906
    inside = {row['device_model_id'] for row in shape['inside_replay_query_shape_map']}
    outside = {int(key): value for key, value in shape['other906_RI_model_distribution'].items()}
    assert inside.isdisjoint(outside) and sum(outside.values()) == 906
    by_model = defaultdict(list)
    for row in graph['compute_ops']:
        by_model[row['model_id']].append(row)
    physical = replay_capacities(shape, by_model, outside)
    physical_lanes = sum(row['query_lanes_per_replay_in_SparseFlashMla'] *
                         row['replays_with_direct_device_correlation'] for row in physical)

    logical_requests = {int(size): row['steps'] for size, row in target['by_active_request_count'].items()}
    assert sum(logical_requests.values()) == sum(outside.values())
    logical_slots = DRAFT_WIDTH * sum(size * count for size, count in logical_requests.items())
    slots = target['proposal_slots_and_response_counters']
    assert logical_slots == slots['on_first_request_rows'] + slots['on_cached_request_rows']
    pairing = uniquely_saturated_buckets(logical_requests, physical)
    assert pairing is not None
    extra = sum(row['conditional_extra_shape_lanes'] for row in pairing)
    assert physical_lanes == logical_slots + extra
    return {
        'epoch': 727, 'capture': 'G804', 'physical_rank': 0,
        'outside_nonzero_target_scheduler_range_replays': sum(outside.values()), 'physical_models': physical,
        'physical_query_lanes_summed_once_per_outside_replay': physical_lanes,
        'scheduler_proposal_slots_in_corresponding_cohort': logical_slots,
        'assumptions_for_logical_capacity_comparison': [
            'One outside-range draft replay covers the requests of each nonzero target step',
            'The draft graph request capacity cannot be less than the number of active requests it serves',
            'Each logical draft request uses five query lanes',
        ],
        'conditional_unique_aggregate_bucket_assignment': pairing,
        'conditional_extra_lanes': extra,
        'conditional_extra_lanes_divided_by_scheduler_proposal_slots': extra / logical_slots,
        'outside_model_named_compute_task_duration_sum_ns': sum(
            row['all_named_compute_task_duration_sum_ns'] for row in physical),
        'inside_model_named_compute_task_duration_sum_ns': sum(
            row['sum_duration_ns_not_walltime'] for model in inside for row in by_model[model]),
        'per_iteration_draft_to_target_join_verified': False,
        'scope': 'Use graph models, recorded SparseFlashMla input shapes and one-rank task duration sums. '
                 'Aggregate capacity accounting under the listed assumptions does not prove per-step matches, '
                 'executed masked-lane work or elapsed time savings.',
        'input_evidence_sha256': {path: digest for path, digest in FILES.values()},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(load(args.evidence_root)), indent=2, sort_keys=True))
