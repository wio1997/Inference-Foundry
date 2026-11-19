"""Reproduce the ordered D0 target/draft pairing in the actual G804 capture."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


REPLAY_SOURCE_SHA = 'f123e4466bf2740b1ac4778790b2fad4b16e8bdbfafd2b3c0520c22525df10f4'
ANNOTATION_FILE = 'profile_capture_G804/logical/decode/canonical_annotations.json'
ANNOTATION_SHA = '7b7b9b2ee04a00ce589cdbd62c412bb4fff6976c4076977c87074bfdb1a455e0'
SHAPE_FILE = 'logical_device_map_G867/D0_query_shape_map.json'
SHAPE_SHA = '131ac32099d8c918c408e4845ff8ad1d07e3886abeb7492e7356a2c35b04afdb'
CAPACITY_FILE = 'profile_capture_G804/derived/D0_outside_target_graph_lanes_G1454.json'
CAPACITY_SHA = 'f5e45588f023a4fbce5af06e6d60474ab8e12c6f5cb96fa1bb931b0fe2805979'
DRAFT_WIDTH = 5
TARGET_WIDTH = 6


def read_pinned_json(path, digest):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise ValueError(f'Input hash mismatch: {path.name}')
    return json.loads(raw)


def device_model_id(row):
    (model, count), = row['direct_device_model_counts'].items()
    assert count > 0
    return int(model)


def summarize(calls, scheduler, shape, draft):
    assert len(calls) == 1812 and len(scheduler) == 913
    assert len({row['connectionId'] for row in calls}) == len(calls)
    assert all(row['start_ns'] < row['end_ns'] for row in calls)
    assert all(previous['end_ns'] <= following['start_ns']
               for previous, following in zip(calls, calls[1:]))

    logical_steps = [row['logical'] for row in scheduler if row['logical']['requests']]
    assert len(logical_steps) == 906
    assert [step['iteration'] for step in logical_steps] == [
        call['annotation_iteration'] for call in calls[::2]]
    assert all(call['annotation_iteration'] is None and call['logical_requests'] is None
               and call['logical_query_tokens'] is None for call in calls[1::2])

    target_shape = {row['logical_requests']: row for row in shape['inside_replay_query_shape_map']}
    draft_shape = {row['model_id']: row for row in draft['physical_models']}
    assert {row['device_model_id'] for row in target_shape.values()}.isdisjoint(draft_shape)
    draft_capacities = sorted(row['request_capacity_at_draft_width'] for row in draft_shape.values())
    buckets, cap_shortfalls, gaps_ns = Counter(), Counter(), []
    next_underfill, next_nonminimal = 0, 0
    logical_draft, physical_draft = 0, 0

    for index, (step, target_call, draft_call) in enumerate(
            zip(logical_steps, calls[::2], calls[1::2], strict=True)):
        requests = len(step['requests'])
        assert all(row['draft_proposals'] == DRAFT_WIDTH for row in step['requests'])
        assert target_call['logical_requests'] == requests
        assert step['scheduled_query_tokens'] == target_call['logical_query_tokens'] == TARGET_WIDTH * requests
        target_model, draft_model = device_model_id(target_call), device_model_id(draft_call)
        assert target_model == target_shape[requests]['device_model_id']
        capacity = draft_shape[draft_model]['request_capacity_at_draft_width']
        lanes = draft_shape[draft_model]['query_lanes_per_replay_in_SparseFlashMla']
        assert lanes == DRAFT_WIDTH * capacity
        assert capacity == next(size for size in draft_capacities if size >= requests)
        buckets[(requests, target_model, draft_model, capacity)] += 1
        logical_draft += requests * DRAFT_WIDTH
        physical_draft += lanes
        gaps_ns.append(draft_call['start_ns'] - target_call['end_ns'])

        if index + 1 < len(logical_steps):
            next_requests = len(logical_steps[index + 1]['requests'])
            next_underfill += capacity < next_requests
            minimal = next(size for size in draft_capacities if size >= next_requests)
            next_nonminimal += capacity != minimal
            if capacity < next_requests:
                cap_shortfalls[(capacity, next_requests)] += 1

    assert logical_draft == draft['scheduler_proposal_slots_in_corresponding_cohort']
    assert physical_draft == draft['physical_query_lanes_summed_once_per_outside_replay']
    assert physical_draft - logical_draft == draft['conditional_extra_lanes']
    assert next_nonminimal >= next_underfill > 0
    return {
        'capture': 'G804',
        'epoch': 727,
        'decode_physical_rank': 0,
        'CANN_replay_count': len(calls),
        'CANN_calls_in_strict_timestamp_order_with_nonoverlapping_API_spans': True,
        'target_CANN_replays_directly_correlated_and_enclosed_by_nonzero_scheduler_span': len(logical_steps),
        'following_draft_CANN_replays_directly_correlated_with_device_graph': len(logical_steps),
        'all_following_draft_graphs_fit_smallest_observed_capacity_for_target_batch': True,
        'by_paired_batches': [
            {'logical_target_requests': req, 'target_model_id': target, 'draft_model_id': draft_id,
             'draft_graph_request_capacity': capacity, 'steps': count,
             'extra_draft_shape_lanes': (capacity - req) * DRAFT_WIDTH * count}
            for (req, target, draft_id, capacity), count in sorted(buckets.items())],
        'next_target_pairing_negative_control': {
            'checked_draft_replays_with_following_target': len(logical_steps) - 1,
            'would_underfill': next_underfill,
            'would_differ_from_smallest_fitting_capacity': next_nonminimal,
            'underfilled_buckets': [
                {'draft_graph_request_capacity': cap, 'next_step_logical_requests': req, 'steps': count}
                for (cap, req), count in sorted(cap_shortfalls.items())]},
        'logical_scheduled_draft_slots': logical_draft,
        'physical_draft_graph_shape_lanes': physical_draft,
        'extra_draft_graph_shape_lanes': physical_draft - logical_draft,
        'extra_as_fraction_of_logical_draft_slots': (physical_draft - logical_draft) / logical_draft,
        'target_CANN_API_end_to_draft_CANN_API_start_gap_ns': {
            'min': min(gaps_ns), 'median': median(gaps_ns), 'max': max(gaps_ns)},
        'pairing_scope': ('Same D0 CANN API clock and ordered graph calls, scheduler enclosure for target, '
                          'direct device connection for both graphs, and independent draft capacity check. '
                          'The pinned worker sequence passes the target input batch to the following proposer.'),
        'source_scope': ('Read the private original G860 graph_calls output verified from the G804 device DB. '
                         'The draft call has no independent scheduler iteration annotation. The CANN API gap is a '
                         'host enqueue interval. Graph capacity does not measure masked execution or E2E savings.'),
        'source_SHA256': {'G860_graph_calls.json': REPLAY_SOURCE_SHA, ANNOTATION_FILE: ANNOTATION_SHA,
                          SHAPE_FILE: SHAPE_SHA, CAPACITY_FILE: CAPACITY_SHA},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph_calls', type=Path, help='private original G860 graph_calls output')
    parser.add_argument('evidence', type=Path, help='archived evidence root')
    args = parser.parse_args()
    calls = read_pinned_json(args.graph_calls, REPLAY_SOURCE_SHA)
    scheduler = read_pinned_json(args.evidence / ANNOTATION_FILE, ANNOTATION_SHA)
    shape = read_pinned_json(args.evidence / SHAPE_FILE, SHAPE_SHA)
    capacity = read_pinned_json(args.evidence / CAPACITY_FILE, CAPACITY_SHA)
    print(json.dumps(summarize(calls, scheduler, shape, capacity), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
