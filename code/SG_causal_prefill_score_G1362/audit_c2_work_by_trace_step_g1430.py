"""Count C2 logical work from Prefill trace geometry or aggregate step markers."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from prefill_logical_score_work_g1422 import sum_causal_widths_through


COHORT = '5cf5dbc8174409a627c8a3439f4b07e0ca99d53ac19c187ed4988eff354d0c00'
NATIVE_CANONICAL_SHA256 = 'f608e9d94d43153b62198781180b4ef987f6369721a1cc6f2e7ba9214be066c7'
NATIVE_PROFILE_SHA256 = 'dcf76912c1af3b2ea521914de6d6bdf92ef6565ac4bd3e2e625b53f00a40995c'
MOMENTS = ('c_sq', 'c_sqsq', 'c_sk', 'c_sqsk')


def moment_bounds(batch, fields):
    """Give integer bounds without choosing a per-request decomposition."""
    count, squares, ends, weighted_ends = (fields[key] for key in MOMENTS)
    assert all(type(x) is int and x > 0 for x in (batch, count, squares, ends, weighted_ends))
    assert batch <= count <= ends and count <= squares <= weighted_ends
    assert count % 2 == squares % 2 and batch * squares >= count * count
    # A square is 1 modulo 4 exactly when the query length is odd.
    odd_lengths = batch - (batch - squares) % 4
    assert odd_lengths >= 0
    first_odd_count = (count - odd_lengths) // 2
    last_odd_count = (count + odd_lengths) // 2
    position_sum = (2 * weighted_ends - squares + count) // 2
    # The odd-position count has the same parity as the position sum.
    first_odd_count += (position_sum - first_odd_count) % 2
    last_odd_count -= (last_odd_count - position_sum) % 2
    assert first_odd_count <= last_odd_count
    eligible = [(position_sum - last_odd_count) // 2, (position_sum - first_odd_count) // 2]
    max_odd_end_weight = count - (count - weighted_ends) % 2
    dense = [(weighted_ends - max_odd_end_weight) // 2, weighted_ends // 2]
    assert 0 <= eligible[0] <= eligible[1] <= dense[1] and 0 <= dense[0] <= dense[1]
    return {'query_rows': count, 'C2_eligible_pair_bounds': eligible, 'C2_dense_pair_bounds': dense}


def exact_work(requests):
    query = [r['scheduled_query_tokens'] for r in requests]
    prefix = [r['computed_tokens'] for r in requests]
    ends = [before + q for before, q in zip(prefix, query)]
    assert all(type(q) is int and q > 0 for q in query)
    assert all(type(before) is int and before >= 0 for before in prefix)
    bounds = moment_bounds(len(requests), dict(zip(MOMENTS, (
        sum(query), sum(q*q for q in query), sum(ends), sum(q*k for q, k in zip(query, ends)),
    ))))
    eligible = sum(sum_causal_widths_through(end, 2) - sum_causal_widths_through(before, 2)
                   for before, end in zip(prefix, ends))
    dense = sum(q * (end // 2) for q, end in zip(query, ends))
    assert bounds['C2_eligible_pair_bounds'][0] <= eligible <= bounds['C2_eligible_pair_bounds'][1]
    assert bounds['C2_dense_pair_bounds'][0] <= dense <= bounds['C2_dense_pair_bounds'][1]
    return bounds, eligible, dense


def summarize_native(annotations, profile):
    assert profile['capture'] == 'G804' and (profile['role'], profile['rank']) == ('prefill', 0)
    by_iteration = {row['iteration']: row for row in profile['active_steps']}
    assert len(by_iteration) == len(profile['active_steps'])
    assert sum(row['device_tasks'] for row in by_iteration.values()) == profile['unique_exact_mappings']
    steps = []
    progress = {}
    for annotation in annotations:
        data = annotation['logical']
        requests = data['requests']
        if not requests:
            assert data['scheduled_query_tokens'] == 0 and data['iteration'] not in by_iteration
            continue
        indexer = by_iteration.pop(data['iteration'])
        assert data['rank'] == 0 and data['scheduled_query_tokens'] == indexer['scheduled_query_tokens']
        assert len(requests) == indexer['scheduled_requests'] and indexer['device_tasks'] == 8
        assert all(row['context_phase'] and row['output_tokens_so_far'] == 0 for row in requests)
        bounds, eligible, dense = exact_work(requests)
        for req in requests:
            key = req['request_id']
            assert req['computed_tokens'] == progress.get(key, 0)
            assert req['new_request'] == (key not in progress)
            if req['new_request']:
                assert req['prompt_tokens'] == 32767
            progress[key] = req['computed_tokens'] + req['scheduled_query_tokens']
        ns = indexer['summed_device_task_duration_ns']
        assert type(ns) is int and ns > 0
        steps.append({'iteration': data['iteration'], **bounds, 'C2_eligible_pairs': eligible,
                      'C2_dense_pairs': dense, 'native_fused_task_sum_ns': ns})
    assert not by_iteration and len(progress) == 16 and set(progress.values()) == {32767}
    assert len(steps) == profile['summary']['nonzero_query_steps']
    assert sum(s['native_fused_task_sum_ns'] for s in steps) == profile['summary']['summed_device_task_duration_ns']
    bands = []
    ordered = sorted(steps, key=lambda s: (s['C2_eligible_pairs'], s['iteration']))
    for band in range(4):
        rows = ordered[len(ordered)*band//4:len(ordered)*(band+1)//4]
        bands.append({'steps': len(rows),
                      'C2_eligible_pairs_min_max': [rows[0]['C2_eligible_pairs'], rows[-1]['C2_eligible_pairs']],
                      'native_fused_task_sum_median_ns': median(s['native_fused_task_sum_ns'] for s in rows),
                      'native_fused_task_sum_ns': sum(s['native_fused_task_sum_ns'] for s in rows)})
    return {'engine': 'vllm', 'capture': 'G804', 'rank': 0, 'exact_joined_steps': len(steps),
            'contiguous_prefill_positions_per_request': 32767, 'completed_request_spans': len(progress),
            'totals': {key: sum(s[key] for s in steps) for key in (
                'query_rows', 'C2_eligible_pairs', 'C2_dense_pairs', 'native_fused_task_sum_ns')},
            'C2_work_quartiles': bands, 'steps': steps,
            'scope': ('C2 pair counts are a shared logical reference on the native scheduler geometry. '
                      'Fused task duration sums use one actual rank. They do not measure physical '
                      'pair visits, wall-clock steps, or SGLang scoring-only cost.')}


def summarize_sglang(report):
    statuses = Counter()
    rows = []
    seen = set()
    for step in report['steps']:
        position = step['step']
        assert position not in seen
        seen.add(position)
        fields = step['logical_fields']
        if step['mode'] != 'EXTEND':
            statuses['outside_pure_prefill'] += 1
        elif not all(key in fields for key in MOMENTS):
            statuses['missing_context_moments'] += 1
        else:
            bounds = moment_bounds(step['batch_size'], fields)
            timing = step['device_attribution'].get('interval_union_ns', {}).get('index_score_marker')
            assert timing is None or (type(timing) is int and timing >= 0)
            statuses['measured_marker' if timing else 'no_attributed_score_marker'] += 1
            rows.append({'step': position, **bounds, 'device_score_marker_union_ns': timing})
    return {'engine': 'sglang', 'rank': report['device_id'], 'geometry_and_marker_statuses': dict(statuses),
            'steps': rows, 'scope': ('Rank-local pure EXTEND steps only. Report missing bounds and markers '
                                     'explicitly. Score marker intervals cover a different operation '
                                     'boundary from the native fused indexer.')}


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--native-annotations', type=Path)
    sources.add_argument('--sglang-attribution', type=Path)
    parser.add_argument('--native-profile', type=Path)
    args = parser.parse_args()
    if args.native_annotations:
        if args.native_profile is None:
            parser.error('--native-annotations requires --native-profile')
        annotations, annotation_sha = load(args.native_annotations)
        assert annotation_sha == NATIVE_CANONICAL_SHA256
        profile, profile_sha = load(args.native_profile)
        assert profile_sha == NATIVE_PROFILE_SHA256
        result = summarize_native(annotations, profile)
        result.update(cohort_sha256=COHORT, input_sha256={'annotations': annotation_sha, 'profile': profile_sha})
    else:
        if args.native_profile:
            parser.error('--native-profile belongs to --native-annotations')
        report, digest = load(args.sglang_attribution)
        result = summarize_sglang(report)
        result['input_sha256'] = {'attribution': digest}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
