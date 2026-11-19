"""Compare the logical work present in pinned vLLM and parsed SGLang profile data."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess

NATIVE_COMMIT = 'cff38e0021cd7f859dea4b8a403b06c3791eb7dd'
COHORT = '5cf5dbc8174409a627c8a3439f4b07e0ca99d53ac19c187ed4988eff354d0c00'
CAPTURE = 'evidence/profile_capture_G804'
ROLES = ('prefill', 'decode')


def digest(raw):
    return sha256(raw).hexdigest()


def local(path, sources):
    raw = path.read_bytes()
    sources[path.name] = digest(raw)
    return json.loads(raw)


def native(repo, path, sources):
    raw = subprocess.check_output(['git', '-C', str(repo), 'show', f'{NATIVE_COMMIT}:{path}'])
    sources[path] = digest(raw)
    return json.loads(raw)


def moment_keys(role):
    return ('requests', 'query_sum', 'query_square_sum', 'kv_end_sum', 'query_times_kv_end') if role == 'prefill' else (
        'requests', 'query_sum', 'query_square_sum')


def serial(histogram, role):
    return [{**dict(zip(moment_keys(role), key)), 'steps': count} for key, count in sorted(histogram.items())]


def exact_positive(shape):
    assert all(type(value) is int and value > 0 for value in shape), shape
    return shape


def native_steps(rows, role):
    histogram = Counter()
    request_tokens = defaultdict(list)
    blank = 0
    for row in rows:
        data = row['logical']
        reqs = data['requests']
        assert data['scheduled_query_tokens'] == sum(request['scheduled_query_tokens'] for request in reqs)
        if not reqs:
            blank += 1
            continue
        assert len({request['request_id'] for request in reqs}) == len(reqs)
        q = [req['scheduled_query_tokens'] for req in reqs]
        before = [req['computed_tokens'] for req in reqs]
        assert all(type(count) is int and count >= 0 for count in before)
        for req, tokens, prefix in zip(reqs, q, before):
            assert type(tokens) is int and tokens > 0
            request_tokens[req['request_id']].append((prefix, tokens))
        key = (len(reqs), sum(q), sum(tokens * tokens for tokens in q))
        if role == 'prefill':
            assert all(req['output_tokens_so_far'] == 0 for req in reqs)
            ends = [prior + tokens for prior, tokens in zip(before, q)]
            key += (sum(ends), sum(tokens * end for tokens, end in zip(q, ends)))
        histogram[exact_positive(key)] += 1
    assert len(request_tokens) == 16
    request_work = Counter()
    if role == 'prefill':
        for steps in request_tokens.values():
            prior_end = 0
            for before, q in steps:
                assert before == prior_end, ('Prefill work does not form one contiguous prefix', before, prior_end)
                prior_end = before + q
            assert prior_end == 32767
            request_work[(len(steps), prior_end)] += 1
    return histogram, blank, request_work



def device_reference(repo, sources, native_roles):
    p = native(repo, f'{CAPTURE}/all8_P_device_CPU_review.json', sources)
    d = native(repo, 'evidence/native_allrank_G909/all8_D_parent_review.json', sources)
    d0 = native(repo, 'evidence/logical_device_map_G867/D0_query_shape_map.json', sources)
    assert p['all8_P_parsed_device_and_CPU_coverage_reviewed']
    assert d['D_all8_original_device_and_CPU_coverage_reviewed']
    assert d0['all1812_RI_calls_direct_device_correlated'] and d0['inside_scheduler_ranges_906_each_once']
    assert {row['rank'] for row in p['reports']} == {row['rank'] for row in d['reports']} == set(range(8))
    prefill = []
    for row in p['reports']:
        device, = row['device_TASK']
        assert device['device_id'] == row['rank'] and device['model_id'] == 4294967295
        window = device['last_end_ns'] - device['first_start_ns']
        assert 0 < row['all_tasks_busy_union_ns_including_WAIT'] <= window
        prefill.append({'rank': row['rank'], 'device_event_window_ns': window,
                        'all_device_task_rows': device['rows'], 'compute_task_rows': row['compute_task_rows'],
                        'all_type_union_ns_including_WAIT': row['all_tasks_busy_union_ns_including_WAIT']})
    shape_rows = [{key: row[key] for key in ('logical_query_tokens', 'logical_requests', 'device_model_id',
                                            'replays', 'SparseFlashMla_input_first_dimension')}
                  for row in d0['inside_replay_query_shape_map']]
    query = sum(row['logical_query_tokens'] * row['replays'] for row in shape_rows)
    extra = sum((row['SparseFlashMla_input_first_dimension'] - row['logical_query_tokens']) * row['replays']
                for row in shape_rows)
    expected = native_roles['decode']['native_batch_size_histogram_positive_steps']
    assert {row['logical_requests']: row['replays'] for row in shape_rows} == expected
    assert query == native_roles['decode']['native_query_tokens'] and extra == d0['extra_input_positions']
    inner = {row['device_model_id'] for row in shape_rows}
    outer = {int(key) for key in d0['other906_RI_model_distribution']}
    assert inner.isdisjoint(outer)
    decode = []
    for row in d['reports']:
        counts = {model: count for rank, model, count in row['device_model_counts'] if rank == row['rank']}
        assert set(counts) == inner | outer | {4294967295}
        decode.append({'rank': row['rank'], 'CANN_execute_calls': row['RI_execute_calls'],
                       'compute_task_rows': row['compute_task_rows'],
                       'device_task_rows_on_D0_scheduler_enclosed_model_ids': sum(counts[key] for key in inner),
                       'device_task_rows_on_D0_other_graph_model_ids': sum(counts[key] for key in outer),
                       'device_task_rows_on_unassigned_model_id': counts[4294967295]})
    return {'native_prefill_device_counts': prefill, 'native_decode_device_graph_family_counts': decode,
            'native_D0_direct_scheduler_replay_work_shapes': shape_rows,
            'native_D0_extra_SparseFlashMla_input_positions': extra,
            'scope': 'P windows include all device task types. Only D0 graph families have direct scheduler connection evidence. Rank counts cannot be added as user tokens; intervals and task counts do not establish a cross-framework time difference.'}


def sglang_steps(report, role):
    accepted_modes = ('EXTEND',) if role == 'prefill' else ('VERIFY', 'TARGET_VERIFY')
    fields = ('c_sq', 'c_sqsq', 'c_sk', 'c_sqsk') if role == 'prefill' else ('g_sq', 'g_sqsq')
    histogram = Counter()
    missing = Counter()
    if report.get('unparsed_step_names'):
        missing['unparsed_step_names'] = len(report['unparsed_step_names'])
    for row in report['spans']:
        attrs = row['actual_query_KV_aggregate_fields']
        if row['mode'] not in accepted_modes or not all(field in attrs for field in fields):
            missing[row['mode']] += 1
            continue
        histogram[exact_positive((row['batch_size'], *(attrs[field] for field in fields)))] += 1
    return histogram, dict(sorted(missing.items()))


def sg_reports(args, sources):
    if args.sg_manifest is None:
        assert not (args.sg_prefill or args.sg_decode or args.sg_workload_audit)
        return None
    assert args.sg_workload_audit is not None and args.sg_workload_audit.parent.resolve() == args.sg_manifest.parent.resolve()
    manifest = local(args.sg_manifest, sources)
    audit = local(args.sg_workload_audit, sources)
    state = local(args.sg_manifest.with_name('state.json'), sources)
    assert state['phase'] == 'profile_workload_complete_pending_allrank_review' and state['actual_requests'] == 16
    assert manifest['dataset_sha256'] == COHORT and manifest['gate']['engine'] == 'sglang'
    assert audit['fully_validated'] and audit['requests'] == audit['successful_requests'] == 16
    assert audit['input_tokens'] == 32768 and audit['output_tokens'] == 1024
    assert audit['configured_concurrency'] == audit['observed_max_concurrency'] == 16
    groups = {}
    for role, paths in (('prefill', args.sg_prefill), ('decode', args.sg_decode)):
        assert len(paths) == 8
        role_results = {}
        for path in paths:
            raw = path.read_bytes()
            report = json.loads(raw)
            rank, = report['actual_device_ids']
            assert type(rank) is int and 0 <= rank < 8 and rank not in role_results
            histogram, missing = sglang_steps(report, role)
            role_results[rank] = histogram, missing
            sources[f'SGLang {role} rank {rank} parsed report'] = digest(raw)
        groups[role] = role_results
    return groups


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--native-repo', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--sg-manifest', type=Path)
    cli.add_argument('--sg-workload-audit', type=Path)
    for role in ROLES:
        cli.add_argument(f'--sg-{role}', type=Path, nargs='*', default=[])
    args = cli.parse_args()
    sources = {}
    metadata = native(args.native_repo, 'metadata/profile_epoch_G727.json', sources)
    assert metadata['profile_input_dataset_sha256'] == COHORT and metadata['actual_allrank_profile_accepted']
    audit = native(args.native_repo, f'{CAPTURE}/workload/workload_raw_audit.json', sources)
    assert audit['fully_validated'] and audit['requests'] == audit['successful_requests'] == 16
    assert audit['input_tokens'] == 32768 and audit['output_tokens'] == 1024
    assert audit['configured_concurrency'] == audit['observed_max_concurrency'] == 16
    sg = sg_reports(args, sources)
    roles = {}
    for role in ROLES:
        anchor = f'{CAPTURE}/logical/{role}'
        rows = native(args.native_repo, f'{anchor}/canonical_annotations.json', sources)
        summary = native(args.native_repo, f'{anchor}/all8_summary.json', sources)
        assert summary['all8_normalized_scheduler_sequences_identical']
        assert len({entry['normalized_logical_sequence_sha256'] for entry in summary['ranks']}) == 1
        histogram, empty, request_work = native_steps(rows, role)
        queries = sum(key[1] * n for key, n in histogram.items())
        assert {entry['rank'] for entry in summary['ranks']} == set(range(8))
        assert all(entry['annotations'] == len(rows) and entry['total_scheduled_query_tokens'] == queries for entry in summary['ranks'])
        occupancy = Counter()
        for key, count in histogram.items():
            occupancy[key[0]] += count
        roles[role] = {
            'native_positive_query_steps': sum(histogram.values()),
            'native_zero_query_annotations': empty,
            'native_query_tokens': queries,
            'native_batch_size_histogram_positive_steps': dict(sorted(occupancy.items())),
            'native_work_signature_histogram': serial(histogram, role),
        }
        if role == 'prefill':
            roles[role]['native_contiguous_prompt_sequences'] = [
                {'positive_steps_per_request': steps, 'query_tokens_per_request': tokens, 'requests': count}
                for (steps, tokens), count in sorted(request_work.items())
            ]
        if sg is not None:
            roles[role]['SGLang_shape_support_against_native_reference_by_rank'] = [
                {'rank': rank, 'SG_annotated_steps': sum(weights.values()),
                 'SG_outside_annotation_comparison': missing,
                 'steps_with_matching_signatures_multiset_intersection': sum((histogram & weights).values()),
                 'SG_signatures': serial(weights, role)}
                for rank, (weights, missing) in sorted(sg[role].items())
            ]
    device = device_reference(args.native_repo, sources, roles)
    value = {'native_commit': NATIVE_COMMIT, 'cohort_dataset_sha256': COHORT,
             'cohort': {'requests': 16, 'concurrency': 16, 'input_tokens': 32768, 'output_tokens': 1024},
             'roles': roles, 'device_reference': device, 'input_sha256': sources,
             'SG_matched_capture_data_supplied': sg is not None,
             'prefill_key_definition': 'Uses query lengths and post-extend KV lengths. Four moments plus batch size identify a work bucket.',
             'decode_key_definition': 'Uses target verification query widths and batch size. KV reference differs between scheduler and forward annotations and is omitted.',
             'scope': 'Histogram matching only. Equal moments can describe different per-request rows. This does not match timestamps, device work, or prove latency comparability.'}
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
