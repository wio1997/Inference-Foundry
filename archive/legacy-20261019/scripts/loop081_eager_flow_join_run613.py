#!/usr/bin/env python3
"""Exact CPU→NPU flow joins for Run611 eager Scatter/SparseAttn events.

This proves native event ownership, not tensor storage alias or first-read ABI.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
RUN611 = ROOT / 'evidence/20260928_loop081_bound/run611'
RUN613 = ROOT / 'evidence/20260928_loop081_bound/run613'
MANIFEST = RUN611 / 'offline_parse/manifest.json'
SCOPE = RUN611 / 'scope_ledger.json'
LEDGER = RUN613 / 'eager_scatter_ledger.json'
OUT = RUN613 / 'eager_flow_join.json'
NAMES = ('aclnnScatterNdUpdateSk', 'aclnnSparseAttnSharedkv')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, why):
    if not ok:
        raise ValueError(why)


manifest = json.loads(MANIFEST.read_text())
scope = json.loads(SCOPE.read_text())
ledger = json.loads(LEDGER.read_text())
need(scope['manifest_sha256'] == sha(MANIFEST) and
     ledger['manifest_sha256'] == sha(MANIFEST) and
     ledger['host_scope_sha256'] == sha(SCOPE) and
     len(manifest['parsed']) == 8, 'upstream identity')
scopes = {(r['rank'], r['ordinal_in_profile']): r for r in scope['rows']}
rows = []
for parsed in manifest['parsed']:
    rank = parsed['rank']
    path = ROOT / parsed['trace_view_json']
    need(sha(path) == parsed['trace_view_sha256'], f'trace SHA rank {rank}')
    events = json.loads(path.read_text())
    starts = defaultdict(list)
    ends = defaultdict(list)
    for event in events:
        if event.get('cat') != 'async_npu' or event.get('name') != 'torch_to_npu':
            continue
        if event.get('ph') == 's':
            starts[(event['pid'], event['tid'], Decimal(event['ts']))].append(event)
        elif event.get('ph') == 'f':
            ends[event['id']].append(event)
    natives = defaultdict(list)
    for event in events:
        if event.get('ph') != 'X' or event.get('args', {}).get('Model Id') != 4294967295:
            continue
        natives[(event['pid'], event['tid'], Decimal(event['ts']))].append(event)
    layers = [e for e in events if e.get('cat') == 'cpu_op' and
              e.get('name', '').startswith('extreme::dspark_layer::')]
    cpu = sorted((e for e in events if e.get('cat') == 'cpu_op' and
                  e.get('name') in NAMES), key=lambda e: Decimal(e['ts']))
    need(sum(e['name'] == NAMES[0] for e in cpu) == 18 and
         sum(e['name'] == NAMES[1] for e in cpu) == 9,
         f'CPU op count rank {rank}')
    joined = []
    for event in cpu:
        host_ts = Decimal(event['ts'])
        matches = starts[(event['pid'], event['tid'], host_ts)]
        need(len(matches) == 1, f'CPU→flow uniqueness rank {rank} {event["name"]}')
        flow = matches[0]
        finish = ends[flow['id']]
        need(len(finish) == 1, f'flow end uniqueness rank {rank}')
        f = finish[0]
        device = natives[(f['pid'], f['tid'], Decimal(f['ts']))]
        need(len(device) == 1, f'flow→native uniqueness rank {rank}')
        n = device[0]
        need(n['args']['Physic Stream Id'] == 47 and
             n['args']['Model Id'] == 4294967295 and
             (('ScatterNdUpdateSk' in n['name']) if event['name'] == NAMES[0]
              else n['name'] == 'SparseAttnSharedkv'),
             f'native type rank {rank} {event["name"]}')
        layer_tag = sorted({layer['name'] for layer in layers
                            if Decimal(layer['ts']) <= host_ts <=
                            Decimal(layer['ts']) + Decimal(str(layer['dur']))})
        owner = [ordinal for ordinal in range(3) if
                 Decimal(str(scopes[(rank, ordinal)]['stages']['proposer']['start_us'])) <=
                 host_ts <=
                 Decimal(str(scopes[(rank, ordinal)]['stages']['proposer']['end_us']))]
        need(len(owner) == 1, f'proposer owner rank {rank}')
        joined.append({'rank': rank, 'ordinal_in_profile_by_host': owner[0],
                       'cpu_name': event['name'], 'host_start_us': str(host_ts),
                       'layer_scope': layer_tag, 'flow_id': flow['id'],
                       'native_name': n['name'], 'native_start_us': f['ts'],
                       'native_task_id': n['args']['Task Id'],
                       'native_stream_id': n['args']['Physic Stream Id']})
    need(len({x['flow_id'] for x in joined}) == 27, f'flow reuse rank {rank}')
    for ordinal in range(3):
        block = [x for x in joined if x['ordinal_in_profile_by_host'] == ordinal]
        scatter = [x for x in block if x['cpu_name'] == NAMES[0]]
        attn = [x for x in block if x['cpu_name'] == NAMES[1]]
        need(len(scatter) == 6 and len(attn) == 3,
             f'Host owner cardinality rank {rank} ordinal {ordinal}')
        need([x['layer_scope'] for x in scatter] ==
             [[], [], [], ['extreme::dspark_layer::43'],
              ['extreme::dspark_layer::44'], ['extreme::dspark_layer::45']] and
             [x['layer_scope'] for x in attn] ==
             [['extreme::dspark_layer::43'], ['extreme::dspark_layer::44'],
              ['extreme::dspark_layer::45']],
             f'layer scope pattern rank {rank} ordinal {ordinal}')
        ordered_device = sorted(block, key=lambda x: Decimal(x['native_start_us']))
        need([x['native_task_id'] for x in ordered_device] ==
             sorted({x['native_task_id'] for x in ordered_device}),
             f'stream47 native order rank {rank} ordinal {ordinal}')
        next_target = (scopes[(rank, ordinal + 1)]['stages']['target']['start_us']
                       if ordinal < 2 else None)
        rows.append({'rank': rank, 'ordinal_in_profile_by_host': ordinal,
                     'nominal_cycle_if_window_exact': 64 + ordinal,
                     'host_to_native_exact_flow_join_count': 9,
                     'scatter_count': 6, 'sparse_attention_count': 3,
                     'native_scatter_after_next_target_host_start':
                     None if next_target is None else sum(
                         Decimal(x['native_start_us']) >= Decimal(str(next_target))
                         for x in scatter),
                     'ops': block})
need(len(rows) == 24 and sum(r['host_to_native_exact_flow_join_count'] for r in rows) == 216,
     'all8 exact flow joins')
result = {'status': 'all8_eager_scatter_sparse_attention_exact_flow_join_pass',
          'scope': 'Exact torch_to_npu s/f id plus CPU pid/tid/Decimal ts and native pid/tid/Decimal ts proves 144 ScatterNdUpdateSk and 72 SparseAttnSharedkv Host→native event pairs, including rank7. Host source/scope puts first three scatter calls outside draft layer scope and later scatter+attn in layers43/44/45. This does not export actual cache pointer/generation or prove first consumer/mandatory seriality; native timestamp crossing Target Host boundary is current async execution, not saving.',
          'manifest_sha256': sha(MANIFEST),
          'scope_ledger_sha256': sha(SCOPE),
          'eager_cardinality_ledger_sha256': sha(LEDGER),
          'rows': rows}
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'joins': 216,
                  'rank5_first_scatter_after_next_target':
                  rows[5 * 3]['native_scatter_after_next_target_host_start']}))
