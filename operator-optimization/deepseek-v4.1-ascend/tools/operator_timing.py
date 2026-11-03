#!/usr/bin/env python3
"""Read-only whole-task timing summary from SQLite profiles or saved native envelopes."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics


SQL = '''SELECT sh.value, dt.value, od.value, c.blockNum, c.mixBlockNum,
    CAST(t.endNs AS INTEGER)-CAST(t.startNs AS INTEGER), t.rowid
    FROM TASK t JOIN COMPUTE_TASK_INFO c ON c.globalTaskId=t.globalTaskId
    LEFT JOIN STRING_IDS sh ON sh.id=c.inputShapes
    LEFT JOIN STRING_IDS dt ON dt.id=c.inputDataTypes
    LEFT JOIN STRING_IDS od ON od.id=c.outputDataTypes
    WHERE c.opType=?'''


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {'count': len(ordered), 'min_us': ordered[0] / 1000,
            'median_us': statistics.median(ordered) / 1000,
            'p90_us': ordered[math.ceil(.9 * len(ordered)) - 1] / 1000,
            'max_us': ordered[-1] / 1000, 'mean_us': statistics.mean(ordered) / 1000}


def database_summary(root, operator):
    files = sorted(root.glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db'))
    if not files:
        raise ValueError('No profile databases at the expected profile/output layout.')

    def read(path):
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            connection.execute('PRAGMA query_only=ON')
            connection.execute('PRAGMA temp_store=MEMORY')
            rows = []
            for (op_id,) in connection.execute('SELECT id FROM STRING_IDS WHERE value=?', (operator,)):
                duplicate = connection.execute('''SELECT globalTaskId FROM COMPUTE_TASK_INFO
                    WHERE opType=? GROUP BY globalTaskId HAVING COUNT(*)>1''', (op_id,)).fetchall()
                if duplicate:
                    raise ValueError(f'{path.name}: ambiguous static definition keys; do not guess joins.')
                rows.extend(connection.execute(SQL, (op_id,)).fetchall())
            if len({row[-1] for row in rows}) != len(rows):
                raise ValueError('Duplicate TASK join; do not double-count.')
            return rows
        finally:
            connection.close()

    groups = defaultdict(list)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for rows in pool.map(read, files):
            for row in rows:
                if row[5] < 0:
                    raise ValueError('Negative duration.')
                groups[row[:5]].append(row[5])
    output = []
    for signature, durations in sorted(groups.items(), key=lambda item: str(item[0])):
        output.append(dict(zip(['input_shapes', 'input_dtypes', 'output_dtypes', 'block_num', 'mix_block_num'], signature)) | distribution(durations))
    return {'source': str(root), 'database_count': len(files), 'operator_match_field': 'COMPUTE_TASK_INFO.opType via STRING_IDS',
            'task_count': sum(len(v) for v in groups.values()), 'groups': output,
            'quantiles': 'nearest-rank p90 of individual TASK durations; grouped by full recorded signature', 'query_sql': SQL}


def envelope_summary(path, operator, calls_per_envelope):
    raw = path.read_bytes()
    dataset = json.loads(raw)
    selected = []
    profile_names = set()
    for profile in dataset['profiles']:
        for envelope in profile['envelopes']:
            item = envelope['operators'].get(operator)
            if not item or (calls_per_envelope is not None and item['count'] != calls_per_envelope):
                continue
            selected.append(item)
            profile_names.add(profile['profile'])
    if not selected:
        raise ValueError('No matching envelopes.')
    count = sum(item['count'] for item in selected)
    means_ns = [item['sum_ns'] / item['count'] for item in selected]
    return {'source': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'operator_match_field': 'saved native envelope operators keys',
            'selected_profiles': len(profile_names), 'selected_envelopes': len(selected), 'task_count': count,
            'calls_per_envelope_filter': calls_per_envelope,
            'weighted_mean_per_call_us': sum(item['sum_ns'] for item in selected) / count / 1000,
            'distribution_of_envelope_means': distribution(means_ns),
            'quantiles': 'quantiles of envelope averages, NOT quantiles of individual kernel durations',
            'shape_or_Python_creator_binding': 'not inferred from envelope membership/count'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--db-root', type=Path)
    source.add_argument('--envelopes', type=Path)
    parser.add_argument('--operator', required=True)
    parser.add_argument('--cohort', required=True, help='Descriptive source label; not an independent hardware identity attestation.')
    parser.add_argument('--calls-per-envelope', type=int)
    args = parser.parse_args()
    if args.calls_per_envelope is not None and (args.db_root or args.calls_per_envelope < 1):
        parser.error('--calls-per-envelope requires envelope mode and a positive count.')
    summary = database_summary(args.db_root, args.operator) if args.db_root else envelope_summary(args.envelopes, args.operator, args.calls_per_envelope)
    result = {'operator': args.operator, 'cohort_label': args.cohort,
              'measurement_boundary': 'whole recorded named task; internal conversion time is not separately identified',
              'tool': 'tools/operator_timing.py', 'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'summary': summary,
              'limits': ['Different cohorts are not a before/after comparison.',
                         'Shape/dtype grouping does not establish all request/parameter/bank identity.',
                         'Do not sum rank/window task time as request wall time or removable critical-path time.',
                         'This command does not run kernels, load models, capture profiles, or write databases.']}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
