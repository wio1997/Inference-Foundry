"""Read the original G804 Prefill op_statistic exports for the eight physical ranks."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import io
import json
from pathlib import Path
from statistics import median


OPERATORS = {
    'VllmAscendQuantLightningIndexerV2': 'fused_indexer',
    '_quantize_indexer_query_kernel': 'query_quantization',
    '_prepare_indexer_indices_kernel': 'index_write_prepare',
    'VllmAscendQuantLightningIndexerV2Metadata': 'indexer_metadata',
    'SparseFlashMla': 'sparse_attention',
    'SparseFlashMlaMetadata': 'sparse_attention_metadata',
}


def original_statistics(directory, rank):
    record_path = directory / 'result.json'
    worker = record_path.read_bytes()
    exported = json.loads(worker)
    assert exported['returncode'] == 0 and exported['original_raw_SHA256_unchanged']
    assert exported['op_statistic_present']
    export = Path(exported['output']).resolve(strict=True)
    assert export.is_relative_to(directory.resolve()) and export.name == 'ASCEND_PROFILER_OUTPUT'
    table = export / 'op_statistic.csv'
    assert 0 < table.stat().st_size < 256 * 1024
    body = table.read_bytes()
    records = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
    assert {int(row['Device_id']) for row in records} == {rank}

    metrics = defaultdict(lambda: {'calls': 0, 'total_us': Decimal(0)})
    total = Decimal(0)
    for row in records:
        count = int(row['Count'])
        time = Decimal(row['Total Time(us)'])
        assert count > 0 and time.is_finite() and time >= 0
        total += time
        if name := OPERATORS.get(row['OP Type']):
            metrics[name]['calls'] += count
            metrics[name]['total_us'] += time
    assert set(metrics) == set(OPERATORS.values()) and total > 0
    return {
        'rank': rank, 'operation_statistics_SHA256': sha256(body).hexdigest(),
        'worker_export_receipt_SHA256': sha256(worker).hexdigest(), 'statistics_rows': len(records),
        'summed_time_all_operator_records_ms': float(total / 1000),
        'operators': {name: {'calls': values['calls'], 'summed_task_ms': float(values['total_us'] / 1000),
                             'fraction_of_reported_operator_duration': float(values['total_us'] / total)}
                      for name, values in metrics.items()},
    }


def summarize(root):
    rows = []
    for rank in range(8):
        directory = root / 'offline_rank0_g809' if rank == 0 else root / 'offline_remaining_g816' / f'rank{rank}'
        rows.append(original_statistics(directory, rank))
    assert len({row['operation_statistics_SHA256'] for row in rows}) == 8
    stats = {}
    for label in OPERATORS.values():
        per_rank = [row['operators'][label] for row in rows]
        times = [item['summed_task_ms'] for item in per_rank]
        fractions = [item['fraction_of_reported_operator_duration'] for item in per_rank]
        stats[label] = {'counts_by_rank': [item['calls'] for item in per_rank],
                        'min_rank_summed_task_ms': min(times), 'median_rank_summed_task_ms': median(times),
                        'max_rank_summed_task_ms': max(times),
                        'min_rank_fraction_of_operator_sum': min(fractions),
                        'max_rank_fraction_of_operator_sum': max(fractions)}
    return {'capture': 'G804', 'epoch': 727, 'role': 'prefill',
            'observed_at_UTC': datetime.now(timezone.utc).isoformat(),
            'ranks': rows, 'rank_range_by_operator': stats, 'new_model_requests': 0,
            'scope': 'Per-rank operator duration sums from closed native device exports. Overlapping tasks and '
                     'ranks are never summed into elapsed time or used to estimate speedup.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile_root', type=Path)
    args = parser.parse_args()
    print('G1451_NATIVE_PREFILL_OP_STATS_JSON ' + json.dumps(summarize(args.profile_root), sort_keys=True))
