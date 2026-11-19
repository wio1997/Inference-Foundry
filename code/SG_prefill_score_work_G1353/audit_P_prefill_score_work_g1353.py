"""Quantify the actual Prefill score shape and eligible causal work from installed source."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import re
from statistics import median

SCORING = 'srt/layers/attention/dsv4/v41_indexer/scoring.py'
SPARSE = 'srt/layers/attention/dsv4/dsv41_sparse.py'
UTILIZATION = re.compile(
    r'^\|\s*\d+\s*\|\s*[\da-fA-F:.]+\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|$'
)


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def actual_shape(records, source_sha):
    assert {row['device_index'] for row in records} == set(range(8)) and len(records) == 8
    shapes = {(tuple(row['query_shape']), tuple(row['key_shape']), row['chunk_rows'], row['budget_bytes']) for row in records}
    shape, = shapes
    assert all(row['native_score_source_sha256'] == source_sha and row['native_code_object_preserved'] for row in records)
    (qrows, heads, width), (krows, keywidth), row_tile, cap = shape
    assert width == keywidth == 128 and heads == 32
    assert row_tile == cap // (heads * krows * 2)
    assert all(row['chunk_score_shape'] == [row_tile, krows] for row in records)
    return qrows, heads, width, krows, row_tile, cap


def device_point(snapshot):
    values = [tuple(map(int, match.groups())) for line in snapshot['npu_smi'].splitlines()
              if (match := UTILIZATION.match(line))]
    assert len(values) == 8
    cores = [core for core, _, _ in values]
    available = [capacity - used for _, used, capacity in values]
    return {'at': snapshot['at'], 'AICore_percent_across_eight_devices': cores,
            'HBM_remaining_MB_across_eight_devices': available,
            'scheduler_main_thread_wait_channels': {
                rank: state['wchan'] for rank, state in snapshot['scheduler_main_threads'].items()}}


def causal_scenario(qrows, heads, base_k, cap, prompt_tokens):
    # This calculation models an uninterrupted query lane that starts at position zero.
    assert prompt_tokens % qrows == 0 and qrows % base_k == 0
    stride = qrows // base_k
    scenarios = []
    for start in range(0, prompt_tokens, qrows):
        keys = (start + qrows) // stride
        rows_per_call = max(1, cap // (heads * keys * 2))
        dense = qrows * heads * keys
        # Original order and chunk boundaries are fixed. The removable suffix is
        # common to each query tile, so trimming cannot remove an eligible key.
        tile_prefix = sum((stop - offset) * heads * ((start + stop) // stride)
                          for offset in range(0, qrows, rows_per_call)
                          for stop in (min(offset + rows_per_call, qrows),))
        causal = sum(heads * ((position + 1) // stride) for position in range(start, start + qrows))
        assert causal <= tile_prefix <= dense
        scenarios.append({'query_context_after_step': start + qrows, 'key_rows_from_stride': keys,
                          'dense_score_cells': dense, 'causally_eligible_score_cells': causal,
                          'cells_after_whole_query_tile_suffix_trim': tile_prefix,
                          'score_tiles': ceil(qrows / rows_per_call)})
    return scenarios


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('receipt','scoring-source','sparse-source','output'):
        cli.add_argument('--'+key, type=Path, required=True)
    args = cli.parse_args()
    observed = json.loads(args.receipt.read_bytes())
    assert observed['epoch'] == 1317
    sources = observed['current_p_sources']
    assert sha(args.scoring_source) == sources[SCORING]['sha256']
    assert sha(args.sparse_source) == sources[SPARSE]['sha256']
    score_source = args.scoring_source.read_text()
    sparse_source = args.sparse_source.read_text()
    assert 's = indexer.scores(q[tok_c], index_k, weights[tok_c])[:, :lc]' in score_source
    assert 's.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)' in score_source
    assert 'torch.einsum("bhd,nd->bhn", q, k)' in sparse_source
    assert '(s.relu() * weights.unsqueeze(-1)).sum(dim=1)' in sparse_source
    shape = actual_shape(observed['host_only_native_score_receipts']['large_score_call'], sources[SCORING]['sha256'])
    qrows, heads, width, base_k, row_tile, cap = shape
    logs = observed['batch_lines']
    assert len(logs) >= 2 and all('#new-seq: 1, #new-token: 8192, #cached-token: 0' in line for line in logs)
    stamps = [datetime.fromisoformat(re.match(r'^\[([^]]+) TP0 EP0\]', line)[1]) for line in logs]
    intervals = [(after-before).total_seconds() for before,after in zip(stamps,stamps[1:])]
    assert min(intervals) > 0

    scenarios = causal_scenario(qrows, heads, base_k, cap, 32768)
    dense = sum(row['dense_score_cells'] for row in scenarios)
    eligible = sum(row['causally_eligible_score_cells'] for row in scenarios)
    trimmed = sum(row['cells_after_whole_query_tile_suffix_trim'] for row in scenarios)
    report = {
        'epoch': observed['epoch'], 'P_source_receipt_sha256': sha(args.receipt),
        'score_source_sha256': sha(args.scoring_source), 'score_math_source_sha256': sha(args.sparse_source),
        'observed_score_operand_shapes_per_rank': {'query': [qrows,heads,width], 'key': [base_k,width]},
        'observed_first_tile_query_rows': row_tile, 'observed_index_score_cap_bytes': cap,
        'logical_score_tiles_for_full_observed_query_shape': ceil(qrows/row_tile),
        'logical_dense_score_cells_one_indexer_invocation': qrows * heads * base_k,
        'logical_BF16_score_bytes_one_indexer_invocation': qrows * heads * base_k * 2,
        'P_chunk_log_count':len(stamps), 'P_log_interval_seconds':{'min':min(intervals),'median':median(intervals),'max':max(intervals)},
        'non_continuous_device_points': [device_point(sample) for sample in observed['samples']],
        'P_current_queue_observation': observed['queues'],
        'contiguous_32K_causal_work_scenario': {
            'assumption': 'Four 8192-row chunks, zero cached prefix, ratio two inferred from the first operand shape.',
            'steps': scenarios, 'all_causal_invalid_fraction_of_dense_cells': (dense-eligible)/dense,
            'whole_tile_suffix_trim_fraction_of_dense_cells': (dense-trimmed)/dense,
        },
        'scope': 'Actual first large-input operand receipts, source-derived work and two device readings. Score cells are logical work per rank, not measured HBM traffic, kernel time, or full-impact speedup.',
    }
    with args.output.open('x') as f:
        json.dump(report,f,indent=2)
        f.write('\n')
    print(json.dumps({'chunk_log_intervals':report['P_log_interval_seconds'],
         'full_observed_shape_score_tiles':report['logical_score_tiles_for_full_observed_query_shape'],
         'theoretical_causal_fraction':report['contiguous_32K_causal_work_scenario']['all_causal_invalid_fraction_of_dense_cells'],
         'tile_trim_fraction':report['contiguous_32K_causal_work_scenario']['whole_tile_suffix_trim_fraction_of_dense_cells'],
         'device_points':[(min(point['AICore_percent_across_eight_devices']),
                           max(point['AICore_percent_across_eight_devices']),
                           min(point['HBM_remaining_MB_across_eight_devices']),
                           max(point['HBM_remaining_MB_across_eight_devices'])) for point in report['non_continuous_device_points']]}))


if __name__ == '__main__':
    main()
