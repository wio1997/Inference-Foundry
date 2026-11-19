"""Bind the native Prefill tail window to the actual G804 P0 indexer shapes."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


OPERATOR = 'VllmAscendQuantLightningIndexerV2'
REPLAY_SOURCE_FILES = ('decoder_replay_layers.py', 'decoder_replay_state.py', 'model.py')
REPLAY_LOG = re.compile(
    r'Ascend MRV2 decoder bounded replay: layers (\d+)-(\d+) keep each prefill\x27s last (\d+) tokens\.'
)
SERVER_LOG_SHA = 'ba601dc171eda005733b7383fdcb6ce0be36b4f4fdcbf5314d65a3846b55301a'
LAUNCH_SHA = '941f8fb4e98db9a6d2d57aca13ea25dd4908788fdd1c0ce86d32b3d912b8a636'
INPUTS = {
    'scheduler': ('evidence/profile_capture_G804/logical/prefill/canonical_annotations.json',
                  'f608e9d94d43153b62198781180b4ef987f6369721a1cc6f2e7ba9214be066c7'),
    'bridge': ('evidence/profile_capture_G804/derived/P0_indexer_scheduler_bridge_G1419.json',
               'dcf76912c1af3b2ea521914de6d6bdf92ef6565ac4bd3e2e625b53f00a40995c'),
    'shapes': ('evidence/profile_capture_G804/derived/P0_indexer_operand_shapes_G1418.json',
               '5c7f667efb6f4c58370b40db7108aeb4fba438d3d0cf04494afb0171de5d453e'),
    'pins': ('code/native_PD_candidate_G630/actual_image_source_prerequisite.json',
             'aec38bfcc44238184e72f1a71af0496a71c4756e1b83d9ab79a912105f108dd5'),
}


def read_pinned(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f'Input hash mismatch: {path.name}')
    return raw


def runtime_sources(repository, pins):
    prefix = 'vllm_ascend/models/deepseek_v41/'
    source_root = repository / 'code/fixed_native_attention_G1130/models/deepseek_v41'
    by_path = {row['path']: row for row in pins['selected_files']}
    result = {}
    for name in REPLAY_SOURCE_FILES:
        entry = by_path[prefix + name]
        assert len(read_pinned(source_root / name, entry['sha256'])) == entry['bytes']
        result[name] = entry['sha256']
    return result


def summarize(scheduler, trace, shapes, window, first_tail, last_layer):
    assert trace['capture'] == shapes['capture'] == 'G804'
    assert trace['source_epoch'] == shapes['source_epoch'] == 727
    assert trace['rank'] == shapes['rank'] == 0
    assert trace['role'] == shapes['role'] == 'prefill'
    assert (first_tail, last_layer, window) == (21, 39, 128)
    logical = {row['logical']['iteration']: row['logical'] for row in scheduler}
    active = {iteration for iteration, step in logical.items() if step['requests']}
    assert len(logical) == 198 and len(active) == 65
    bridge = trace['active_steps']
    assert {item['iteration'] for item in bridge} == active and len(bridge) == len(active)

    full_calls, tail_calls, tail_batches = Counter(), Counter(), Counter()
    full_queries = tail_queries = 0
    for item in bridge:
        step = logical[item['iteration']]
        requests = step['requests']
        assert step['rank'] == 0 and all(req['context_phase'] for req in requests)
        lengths = [req['scheduled_query_tokens'] for req in requests]
        assert all(length > 0 for length in lengths)
        full_rows = sum(lengths)
        kept_rows = sum(min(length, window) for length in lengths)
        assert kept_rows < full_rows == step['scheduled_query_tokens'] == item['scheduled_query_tokens']
        assert len(requests) == item['scheduled_requests']
        assert item['device_tasks_by_query_rows'] == {str(full_rows): 4, str(kept_rows): 4}
        full_calls[full_rows] += 4
        tail_calls[kept_rows] += 4
        tail_batches[(len(requests), kept_rows)] += 1
        full_queries += full_rows
        tail_queries += kept_rows

    indexer, = (row for row in shapes['operators'] if row['operator'] == OPERATOR)
    durations, observed_calls = Counter(), Counter()
    assert not full_calls.keys() & tail_calls.keys()
    for signature in indexer['signatures']:
        query_shape = signature['Input Shapes'].strip('"').split(';', 1)[0]
        rows = int(query_shape.split(',', 1)[0])
        observed_calls[rows] += signature['calls']
        durations[rows] += signature['summed_device_task_duration_ns']
    assert observed_calls == full_calls + tail_calls
    assert indexer['calls'] == sum(observed_calls.values()) == 520
    assert sum(durations.values()) == sum(item['summed_device_task_duration_ns'] for item in bridge)

    reference_layer_rows = (last_layer + 1) * full_queries
    replay_layer_rows = first_tail * full_queries + (last_layer - first_tail + 1) * tail_queries
    return {
        'capture': 'G804',
        'epoch': 727,
        'prefill_physical_rank': 0,
        'nonzero_scheduler_steps_verified': len(active),
        'scheduler_request_chunks_verified': sum(n * steps for (n, _), steps in tail_batches.items()),
        'full_query_rows_summed_once_per_step': full_queries,
        'tail_query_rows_summed_once_per_step': tail_queries,
        'tail_window_by_observed_batch': [
            {'requests': n, 'kept_query_rows': kept, 'steps': count}
            for (n, kept), count in sorted(tail_batches.items())],
        'indexer_device_tasks': {
            'full_query': {'count': sum(full_calls.values()),
                           'duration_sum_ns': sum(durations[q] for q in full_calls)},
            'tail_query': {'count': sum(tail_calls.values()),
                           'duration_sum_ns': sum(durations[q] for q in tail_calls)},
        },
        'source_implied_layer_row_geometry': {
            'full_query_layer_ids_inclusive': [0, first_tail - 1],
            'tail_query_layer_ids_inclusive': [first_tail, last_layer],
            'with_tail_replay_layer_rows': replay_layer_rows,
            'all_layers_full_query_layer_rows_reference': reference_layer_rows,
            'reduction_fraction': 1 - replay_layer_rows / reference_layer_rows,
        },
        'scope': ('Runtime log plus pinned decoder replay sources and exact per-step P0 device query shapes. '
                  'Layer-row geometry is source-implied, without individual device task layer IDs. '
                  'Task duration sums and layer rows do not determine end-to-end savings.'),
        'SGLang_tail_replay_runtime_or_equivalent_work_verified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('private_G789_P_server_log', type=Path)
    parser.add_argument('private_G789_P_launch', type=Path)
    args = parser.parse_args()
    docs = {key: json.loads(read_pinned(args.repository / name, digest))
            for key, (name, digest) in INPUTS.items()}
    log = read_pinned(args.private_G789_P_server_log, SERVER_LOG_SHA).decode()
    launch = json.loads(read_pinned(args.private_G789_P_launch, LAUNCH_SHA))
    assert launch['all113_actual_source_pins_match'] and not launch['native_source_overlay']
    match, = REPLAY_LOG.finditer(log)
    first, last, window = map(int, match.groups())
    result = summarize(docs['scheduler'], docs['bridge'], docs['shapes'], window, first, last)
    result['actual_reported_replay_configuration'] = match.group()
    result['actual_runtime_input_SHA256'] = {'G789_P_server_log': SERVER_LOG_SHA, 'G789_P_launch': LAUNCH_SHA}
    result['runtime_source_SHA256'] = runtime_sources(args.repository, docs['pins'])
    result['archived_input_SHA256'] = dict(INPUTS.values())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
