#!/usr/bin/env python3
"""Run616: exact-flow current replay interval census, not a latency bound.

The first three scatter roles follow installed DSpark source order, but their
individual layer identity and cache generation are not exported by the trace.
All timings here are CANN Level0 observer-perturbed native *start* intervals.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from statistics import median

ROOT = Path('/data/wio/Inference_Foundry')
INP = ROOT / 'evidence/20260928_loop081_bound/run613/eager_flow_join.json'
OUT = ROOT / 'evidence/20260928_loop081_bound/run616/eager_interval_census.json'
PIN = 'dc17ab40f85d38d2d5f87d921d44d1b1f4debb51dea6aceddd9b9ae0ad9c99b9'


def need(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


need(hashlib.sha256(INP.read_bytes()).hexdigest() == PIN, 'flow input SHA drift')
source = json.loads(INP.read_text())
need(source['status'] == 'all8_eager_scatter_sparse_attention_exact_flow_join_pass',
     'flow admission drift')
need(len(source['rows']) == 24 and {r['rank'] for r in source['rows']} == set(range(8)),
     'rank/occurrence coverage')

records = []
for row in source['rows']:
    ops = row['ops']
    need(len(ops) == 9 and row['host_to_native_exact_flow_join_count'] == 9,
         'op cardinality drift')
    scatters = [op for op in ops if op['cpu_name'] == 'aclnnScatterNdUpdateSk']
    readers = [op for op in ops if op['cpu_name'] == 'aclnnSparseAttnSharedkv']
    need(len(scatters) == 6 and len(readers) == 3, 'role cardinality drift')
    need(all(op['layer_scope'] == [] for op in scatters[:3]),
         'first three scatter context scope drift')
    for idx, layer in enumerate((43, 44, 45)):
        context = scatters[idx]
        query = scatters[idx + 3]
        reader = readers[idx]
        label = [f'extreme::dspark_layer::{layer}']
        need(query['layer_scope'] == label and reader['layer_scope'] == label,
             'scoped query/reader layer drift')
        device_us = [Decimal(op['native_start_us']) for op in (context, query, reader)]
        host_us = [Decimal(op['host_start_us']) for op in (context, query, reader)]
        tasks = [op['native_task_id'] for op in (context, query, reader)]
        streams = [op['native_stream_id'] for op in (context, query, reader)]
        need(all(stream == 47 for stream in streams) and tasks == sorted(tasks) and
             device_us == sorted(device_us), 'same-stream native start order drift')
        need(len({op['flow_id'] for op in (context, query, reader)}) == 3,
             'nonunique exact flow')
        records.append({
            'rank': row['rank'], 'ordinal': row['ordinal_in_profile_by_host'],
            'nominal_cycle_if_window_exact': row['nominal_cycle_if_window_exact'],
            'layer': layer,
            'context_layer_scope': 'source_order_candidate_not_trace_tag',
            'context_to_query_native_start_us': float(device_us[1] - device_us[0]),
            'query_to_reader_native_start_us': float(device_us[2] - device_us[1]),
            'context_to_reader_native_start_us': float(device_us[2] - device_us[0]),
            'context_to_query_host_start_us': float(host_us[1] - host_us[0]),
            'query_to_reader_host_start_us': float(host_us[2] - host_us[1]),
            'context_host_to_native_start_us': float(device_us[0] - host_us[0]),
            'query_host_to_native_start_us': float(device_us[1] - host_us[1]),
            'reader_host_to_native_start_us': float(device_us[2] - host_us[2]),
            'native_task_ids': tasks,
            'flow_ids': [op['flow_id'] for op in (context, query, reader)],
        })


def distribution(key: str) -> dict[str, float]:
    values = sorted(r[key] for r in records)
    return {'min': values[0], 'median': median(values), 'max': values[-1]}


result = {
    'status': 'all8_exact_flow_current_native_start_intervals_only',
    'source_sha256': PIN,
    'scope': ('24 proposer occurrences x 3 layers; Run610/611 CANN Level0 observer-perturbed '
              'current start spacing. First three context scatter layer indices are source-order '
              'candidates; query scatter and attention have Host layer tags and exact CANN flow. '
              'No cache allocation/row generations, eligible-read proof, physical completion, '
              'removable delay, or strict Bound. No transfer to formal Run99.'),
    'record_count': len(records),
    'distributions_us': {key: distribution(key) for key in (
        'context_to_query_native_start_us', 'query_to_reader_native_start_us',
        'context_to_reader_native_start_us', 'context_to_query_host_start_us',
        'query_to_reader_host_start_us', 'context_host_to_native_start_us',
        'query_host_to_native_start_us', 'reader_host_to_native_start_us')},
    'records': records,
    'strict_scheduling_floor_s': None,
    'removable_wall_time_s': None,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'records': len(records),
                  'distributions_us': result['distributions_us']}))
