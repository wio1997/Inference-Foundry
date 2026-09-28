#!/usr/bin/env python3
"""Read-only source/Host/native cardinality ledger for DSpark eager scatter.

An equal ordered count is a candidate Host↔device match, not a tensor pointer
or generation certificate. In particular, no cycle-time partition is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
RUN611 = ROOT / 'evidence/20260928_loop081_bound/run611'
RUN613 = ROOT / 'evidence/20260928_loop081_bound/run613'
SOURCE = Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend')
PIN = {
    'spec_decode/llm_base_proposer.py': 'e3e1ff579e67f845f155c983294ae5546e6ee76ba330098d7867e870c3f9f6e4',
    'spec_decode/dflash_proposer.py': '6be46559f9f869861c676efb4531b0c01eef8a0445521ecadf5c1fec1c4412f8',
    'models/deepseek_v4_dspark.py': 'b459fa373e89da1722085cffa63a9959502dd29fb1a0cf4b8f073e431d811867',
    'device/device_op.py': '67dda24f28de847f9aee87db9714e5d2293ebb46309ad13152538cf63ca5f1b4',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, why):
    if not ok:
        raise ValueError(why)


for rel, expected in PIN.items():
    need(sha(SOURCE / rel) == expected, f'installed source drift {rel}')
manifest_path = RUN611 / 'offline_parse/manifest.json'
scope_path = RUN611 / 'scope_ledger.json'
native_path = RUN611 / 'native_graph_replay.json'
manifest = json.loads(manifest_path.read_text())
scope = json.loads(scope_path.read_text())
native = json.loads(native_path.read_text())
need(scope['manifest_sha256'] == sha(manifest_path) and
     native['host_scope_ledger_sha256'] == sha(scope_path) and
     len(manifest['parsed']) == 8, 'upstream identity')
scopes = {(r['rank'], r['ordinal_in_profile']): r for r in scope['rows']}
rows = []
for record in manifest['parsed']:
    rank = record['rank']
    path = ROOT / record['trace_view_json']
    need(sha(path) == record['trace_view_sha256'], f'trace SHA rank {rank}')
    events = json.loads(path.read_text())
    host = sorted((e for e in events if e.get('cat') == 'cpu_op' and
                   e.get('name') == 'aclnnScatterNdUpdateSk'),
                  key=lambda e: float(e['ts']))
    device = sorted((e for e in events if e.get('ph') == 'X' and
                     e.get('args', {}).get('Model Id') == 4294967295 and
                     'ScatterNdUpdateSk' in e.get('name', '')),
                    key=lambda e: float(e['ts']))
    layers = [e for e in events if e.get('cat') == 'cpu_op' and
              e.get('name', '').startswith('extreme::dspark_layer::')]
    need(len(host) == len(device) == 18, f'host/native eager scatter rank {rank}')
    task_ids = [e['args']['Task Id'] for e in device]
    need({e['args']['Physic Stream Id'] for e in device} == {47} and
         task_ids == sorted(set(task_ids)), f'eager stream/task order rank {rank}')
    for ordinal in range(3):
        six_host = host[ordinal * 6:(ordinal + 1) * 6]
        six_device = device[ordinal * 6:(ordinal + 1) * 6]
        proposer = scopes[(rank, ordinal)]['stages']['proposer']
        next_target = (scopes[(rank, ordinal + 1)]['stages']['target']['start_us']
                       if ordinal < 2 else None)
        tags = []
        for event in six_host:
            ts = float(event['ts'])
            need(proposer['start_us'] <= ts <= proposer['end_us'],
                 f'Host scatter outside proposer rank {rank} occurrence {ordinal}')
            tags.append(sorted({layer['name'] for layer in layers
                                if float(layer['ts']) <= ts <=
                                float(layer['ts']) + float(layer['dur'])}))
        need(tags == [[], [], [],
                      ['extreme::dspark_layer::43'],
                      ['extreme::dspark_layer::44'],
                      ['extreme::dspark_layer::45']],
             f'Host layer scope pattern rank {rank} occurrence {ordinal}')
        rows.append({'rank': rank, 'ordinal_in_profile_by_order': ordinal,
                     'nominal_cycle_if_window_exact': 64 + ordinal,
                     'host_scatter_count': 6, 'device_scatter_candidate_count': 6,
                     'host_layer_scope_tags': tags,
                     'host_scatter_start_us': [float(e['ts']) for e in six_host],
                     'device_scatter_start_us_by_order_candidate':
                     [float(e['ts']) for e in six_device],
                     'device_stream47_task_ids_by_order_candidate':
                     [e['args']['Task Id'] for e in six_device],
                     'device_starts_after_next_target_host_start_candidate':
                     None if next_target is None else
                     sum(float(e['ts']) >= next_target for e in six_device)})
need(len(rows) == 24, 'all8 three occurrences')
RUN613.mkdir(parents=True, exist_ok=True)
result = {
    'status': 'all8_eager_scatter_cardinality_and_host_scope_pass',
    'scope': 'All8 three Host proposer occurrences have six CPU aclnnScatterNdUpdateSk calls: first three outside draft layer scopes and last three within layer43/44/45 respectively. Each rank has 18 same-name Model UINT_MAX native events on stream47 with increasing task IDs; ordered six-by-six candidate pairing is NOT a proven flow, storage generation or first query reader. Rank5 first candidate device group crosses next Target Host start, so Host-cycle timestamp bins are invalid. No strict Bound or removable time.',
    'source_sha256': PIN,
    'manifest_sha256': sha(manifest_path),
    'host_scope_sha256': sha(scope_path),
    'native_graph_sha256': sha(native_path),
    'rows': rows,
}
out = RUN613 / 'eager_scatter_ledger.json'
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'rows': len(rows),
                  'rank5_cross_next_target_first_candidate':
                  rows[5 * 3]['device_starts_after_next_target_host_start_candidate']}))
