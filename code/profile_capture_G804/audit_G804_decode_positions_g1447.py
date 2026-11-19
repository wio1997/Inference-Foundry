"""Audit actual G804 vLLM Decode scheduler positions and target graph input lanes."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path


FILES = {
    'scheduler': ('profile_capture_G804/logical/decode/canonical_annotations.json',
                  '7b7b9b2ee04a00ce589cdbd62c412bb4fff6976c4076977c87074bfdb1a455e0'),
    'ranks': ('profile_capture_G804/logical/decode/all8_summary.json',
              '2252b5323daf74a0e0252d615e77de21c4de76b661868fde2497a2cadbd5e271'),
    'shape': ('logical_device_map_G867/D0_query_shape_map.json',
              '131ac32099d8c918c408e4845ff8ad1d07e3886abeb7492e7356a2c35b04afdb'),
    'counters': ('profile_capture_G804/actual_spec_counter_deltas.json',
                 '5c17848d994c37953454beb2c755ed21689220d2377c6e3d7a06f3217008b799'),
    'acceptance': ('profile_capture_G804/parent_diagnostic_raw.json',
                   'd2fb19ad7e0dd08a20e5a0de805795cdd04fd48543f68e4a9d5b98c616fd092b'),
    'manifest': ('profile_capture_G804/workload/manifest.json',
                 '5636ee98201020b59b7140d2962e1d56cbb9567282c7dedd2a1bf8eca0793edf'),
}
DATASET_SHA = '5cf5dbc8174409a627c8a3439f4b07e0ca99d53ac19c187ed4988eff354d0c00'


def read_evidence(root):
    loaded = {}
    for key, (name, digest) in FILES.items():
        raw = (root / name).read_bytes()
        assert sha256(raw).hexdigest() == digest, name
        loaded[key] = json.loads(raw)
    return loaded


def metric(roles, name):
    value, = (count for key, count in roles['decode'].items() if key.split('{', 1)[0] == name)
    assert int(value) == value
    return int(value)


def summarize(data):
    manifest = data['manifest']
    assert manifest['gate']['profile_epoch'] == 727 and manifest['dataset_sha256'] == DATASET_SHA
    accepted = data['acceptance']['report']
    assert data['acceptance']['accepted_diagnostic_workload_raw'] and accepted['fully_validated']
    assert accepted['configured_concurrency'] == accepted['observed_max_concurrency'] == 16
    assert accepted['requests'] == accepted['successful_requests'] == 16
    prompt, output = accepted['input_tokens'], accepted['output_tokens']
    assert (prompt, output) == (32768, 1024)
    start = prompt - 1
    stop = start + output
    rank_report = data['ranks']
    assert rank_report['all8_actual_CPU_logical_annotations_present']
    assert rank_report['all8_normalized_scheduler_sequences_identical']
    rank_rows = rank_report['ranks']
    assert sorted(row['rank'] for row in rank_rows) == list(range(8))
    assert len({row['normalized_logical_sequence_sha256'] for row in rank_rows}) == 1

    by_batch = defaultdict(Counter)
    seen = defaultdict(set)
    last_start = {}
    advances, slots = Counter(), Counter()
    total_steps = 0
    for total_steps, stored in enumerate(data['scheduler'], 1):
        step = stored['logical']
        assert step['iteration'] == total_steps and step['rank'] == 0 and step['schema'] == 1
        rows = step['requests']
        assert len({row['request_id'] for row in rows}) == len(rows)
        assert sum(row['scheduled_query_tokens'] for row in rows) == step['scheduled_query_tokens']
        if not rows:
            continue
        group = by_batch[len(rows)]
        group['steps'] += 1
        for row in rows:
            key = row['request_id']
            first = key not in last_start
            assert first == row['new_request'] == row['context_phase'] and not row['resumed']
            prefix = row['computed_tokens']
            query = row['scheduled_query_tokens']
            proposals = row['draft_proposals']
            assert type(prefix) is int and query == proposals + 1 == 6
            assert prefix == start + row['output_tokens_so_far']
            if first:
                assert row['prompt_tokens'] == prompt and prefix == start
            else:
                advance = prefix - last_start[key]
                assert 0 < advance <= query
                advances[advance] += 1
            slots['on_first_request_rows' if first else 'on_cached_request_rows'] += proposals
            last_start[key] = prefix
            endpoint = prefix + query
            group['logical_query_positions'] += query
            group['logical_prefix_tokens_sum'] += prefix
            group['logical_endpoint_tokens_sum'] += endpoint
            group['uncompressed_causal_query_token_pairs'] += query * prefix + query * (query + 1) // 2
            positions = seen[key]
            for pos in range(prefix, endpoint):
                if pos >= stop:
                    group['beyond_output_position_budget'] += 1
                elif pos in positions:
                    group['repeat_within_output_position_budget'] += 1
                else:
                    group['first_within_output_position_budget'] += 1
                positions.add(pos)
    assert len(seen) == accepted['requests'] and all(
        set(range(start, stop)).issubset(values) for values in seen.values())
    assert all(row['annotations'] == total_steps for row in rank_rows)

    graph = data['shape']
    assert graph['all1812_RI_calls_direct_device_correlated']
    assert graph['inside_scheduler_ranges_906_each_once']
    shape_by_batch = {row['logical_requests']: row for row in graph['inside_replay_query_shape_map']}
    assert set(shape_by_batch) == set(by_batch)
    for batch, group in by_batch.items():
        shape = shape_by_batch[batch]
        assert shape['replays'] == group['steps'] and shape['logical_query_tokens'] == 6 * batch
        total_lanes = shape['replays'] * shape['SparseFlashMla_input_first_dimension']
        assert total_lanes - group['logical_query_positions'] == shape['extra_query_positions_over_replays']
        group['actual_target_SparseFlashMla_shape_lanes'] = total_lanes
        group['extra_graph_shape_lanes'] = shape['extra_query_positions_over_replays']
    combined = sum(by_batch.values(), Counter())
    logical = combined['logical_query_positions']
    assert logical == sum(combined[name] for name in (
        'first_within_output_position_budget', 'repeat_within_output_position_budget', 'beyond_output_position_budget'))
    assert combined['actual_target_SparseFlashMla_shape_lanes'] == graph[
        'SparseFlashMla_input_query_positions_across_906_inside_replays']
    assert logical == graph['logical_query_positions_across_906_inside_replays']
    assert all(row['total_scheduled_query_tokens'] == logical and row['unique_requests'] == len(seen)
               for row in rank_rows)

    deltas = data['counters']['deltas']
    slots['server_draft_calls'] = metric(deltas, 'vllm:spec_decode_num_drafts_total')
    slots['server_draft_tokens'] = metric(deltas, 'vllm:spec_decode_num_draft_tokens_total')
    slots['server_accepted_draft_tokens'] = metric(deltas, 'vllm:spec_decode_num_accepted_tokens_total')
    slots['server_generation_tokens'] = metric(deltas, 'vllm:generation_tokens_total')
    assert slots['server_generation_tokens'] == output * len(seen)
    return {
        'capture': 'G804', 'vLLM_epoch': 727, 'diagnostic_dataset_SHA256': DATASET_SHA,
        'shape': {'requests': len(seen), 'concurrency': 16, 'input_tokens': prompt, 'output_tokens': output},
        'eight_TP_ranks_have_same_normalized_scheduler_sequence': True,
        'scheduler_annotations': total_steps, 'target_iterations_with_no_queries': total_steps - combined['steps'],
        'zero_based_request_input_positions_expected_to_produce_output': [start, stop - 1],
        'position_and_work_counts': dict(combined),
        'by_active_request_count': {count: dict(by_batch[count]) for count in sorted(by_batch)},
        'start_position_advances_between_successive_rows': dict(sorted(advances.items())),
        'proposal_slots_and_response_counters': dict(slots),
        'cached_proposal_slots_equal_server_draft_tokens': (
            slots['on_cached_request_rows'] == slots['server_draft_tokens']),
        'scope': ('Absolute scheduler positions and uncompressed logical pairs. Repeated positions can carry '
                  'different tokens. Graph input lanes are measured shape capacity. Neither count estimates '
                  'executed FLOPs or elapsed-time savings. Cross-framework matching still requires actual SGLang events.'),
        'input_evidence_SHA256': {name: digest for name, digest in FILES.values()},
    }


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('evidence', type=Path)
    args = cli.parse_args()
    print(json.dumps(summarize(read_evidence(args.evidence)), indent=2, sort_keys=True))
