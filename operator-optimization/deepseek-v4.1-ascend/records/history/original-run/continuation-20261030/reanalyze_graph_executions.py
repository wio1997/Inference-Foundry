#!/usr/bin/env python3
"""Read official exports only. Detect repeated device model/stream signatures.

This does not label requests, assume a historical model ID, or access an NPU.
"""
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-20261030T084726Z-cb252d1d/analysis/operator-export/copies')


def ns(value):
    x = Decimal(value.strip()) * 1000
    assert x.is_finite() and x == int(x)
    return int(x)


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return sum(b-a for a, b in merged)


def describe(values):
    if not values:
        return None
    return {'count': len(values), 'min': min(values),
            'median': statistics.median(values), 'max': max(values)}


def signature(records):
    # IDs are local to an execution stream; include them for within-rank checks.
    return [(x['Task ID'], x['Type'], x['Input Shapes'], x['Output Shapes']) for x in records]


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


results = []
for directory in sorted(ROOT.glob('*_ascend_pt')):
    info = json.loads(next(directory.glob('profiler_info_*.json')).read_text())
    rank = int(re.search(r'_ep(\d+)_rank', directory.name).group(1))
    assert rank == info['rank_id']
    path = directory / 'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
    rows = []
    with path.open(newline='') as f:
        for x in csv.DictReader(f):
            duration = ns(x['Duration(us)'])
            if duration <= 0:
                continue
            x['s'] = ns(x['Start Time(us)'])
            x['d'] = duration
            x['e'] = x['s'] + duration
            rows.append(x)
    groups = collections.defaultdict(list)
    for x in rows:
        if x['Model ID'] != '4294967295':
            groups[(x['Model ID'], x['Stream ID'])].append(x)
    models = []
    for (model, stream), rr in sorted(groups.items()):
        rr.sort(key=lambda x: (x['s'], x['e']))
        first_key = (rr[0]['Task ID'], rr[0]['Type'])
        indices = [i for i, x in enumerate(rr) if (x['Task ID'], x['Type']) == first_key]
        chunks = [rr[a:b] for a, b in zip(indices, indices[1:]+[len(rr)])]
        sigs = [digest(signature(chunk)) for chunk in chunks]
        frequency = collections.Counter(sigs)
        dominant, dominant_count = frequency.most_common(1)[0]
        executions = []
        for index, (chunk, sig) in enumerate(zip(chunks, sigs)):
            types = collections.defaultdict(list)
            for x in chunk:
                types[x['Type']].append(x)
            moe = types.get('DispatchFFNCombine', [])
            experts24 = all(';24,5120,4608;24,2304,5120;' in x['Input Shapes'] and ';3,6;' in x['Input Shapes'] for x in moe)
            execution = {
                'ordinal': index, 'signature_sha256': sig,
                'dominant_signature': sig == dominant,
                'record_count': len(chunk),
                'unique_task_id_count': len({x['Task ID'] for x in chunk}),
                'start_epoch_ns': chunk[0]['s'], 'end_epoch_ns': max(x['e'] for x in chunk),
                'record_span_ns': max(x['e'] for x in chunk)-chunk[0]['s'],
                'record_union_ns': union([(x['s'], x['e']) for x in chunk]),
                'moe_count': len(moe),
                'main_shape_compatible_40_moe': len(moe) == 40 and experts24,
                'type_union_ns': {t: union([(x['s'], x['e']) for x in vv]) for t, vv in types.items()},
                'moe_tasks': [{'ordinal': i, 'task_id': x['Task ID'], 's_ns': x['s'], 'd_ns': x['d'], 'e_ns': x['e'], 'input_shapes': x['Input Shapes']} for i, x in enumerate(moe)],
            }
            if index+1 < len(chunks):
                execution['next_execution_start_minus_end_ns'] = chunks[index+1][0]['s'] - execution['end_epoch_ns']
            executions.append(execution)
        models.append({'model_id': model, 'stream_id': stream,
                       'first_task_key': first_key, 'signature_frequencies': dict(frequency),
                       'dominant_signature_execution_count': dominant_count,
                       'executions': executions})
    mc2 = [x for x in rows if x['Type'] == 'DispatchFFNCombine']
    shape_mc2 = collections.defaultdict(list)
    for x in mc2:
        shape_mc2[x['Input Shapes']].append(x['d'])
    results.append({'rank': rank, 'source_dir': directory.name,
                    'profiler_start_monotonic_ns': info['start_info']['start_monotonic'],
                    'kernel_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'model_streams': models,
                    'all_moe_shape_duration_ns': {k: describe(v) for k, v in shape_mc2.items()}})

per_rank = collections.defaultdict(list)
for x in results:
    per_rank[x['rank']].append(x)
for rank, rr in per_rank.items():
    for ordinal, x in enumerate(sorted(rr, key=lambda x: x['profiler_start_monotonic_ns']), 1):
        x['window'] = ordinal

print(json.dumps({'scope': 'same official capture, repeated device model/stream signatures',
                  'source_count': len(results), 'rank_count': len(per_rank),
                  'limits': ['Execution boundaries are inferred from recurring first task key and validated by full signatures.',
                             'A dominant full signature is not proof that the first/final capture boundary is a complete graph.',
                             'Model/shape compatibility does not identify requests, source-layer objects, or runtime replay IDs.',
                             'Cross-rank clocks have not been independently calibrated; no collective ordinal binding is asserted.',
                             'Duration, gaps and unions are device-export observations, not removable end-to-end latency.'],
                  'rows': sorted(results, key=lambda x: (x['window'], x['rank']))}, indent=2))
