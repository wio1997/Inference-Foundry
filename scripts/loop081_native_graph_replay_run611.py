#!/usr/bin/env python3
"""Admit three Model45 native Graph replays/rank from existing Run611 copies.

Use static task occurrence plus actual start anchor. connection_id is reused
across replays and is intentionally never a replay key.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
RUN = ROOT / 'evidence/20260928_loop081_bound/run611'
MANIFEST = RUN / 'offline_parse/manifest.json'
SCOPE = RUN / 'scope_ledger.json'
DEST = RUN / 'native_graph_replay.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, why):
    if not ok:
        raise ValueError(why)


manifest = json.loads(MANIFEST.read_text())
scope = json.loads(SCOPE.read_text())
need(scope['manifest_sha256'] == sha(MANIFEST) and
     manifest['status'] == 'all8_salvaged_raw_copy_offline_parse_pass' and
     scope['status'] == 'all8_instrumented_host_scope_ledger_pass',
     'upstream identity')
scope_by_rank = {(r['rank'], r['ordinal_in_profile']): r
                 for r in scope['rows']}
need(len(scope_by_rank) == 24, 'Host scope cardinality')
rows = []
for record in manifest['parsed']:
    rank = record['rank']
    path = ROOT / record['trace_view_json']
    need(sha(path) == record['trace_view_sha256'], f'trace SHA rank {rank}')
    events = json.loads(path.read_text())
    native = [event for event in events if event.get('ph') == 'X' and
              event.get('args', {}).get('Model Id') == 45]
    need(len(native) == 3 * 5412, f'Model45 native cardinality rank {rank}')
    key = lambda e: (e['args']['Physic Stream Id'], e['args']['Task Id'],
                     e['args']['Batch Id'], e['args']['Subtask Id'])
    static_counts = Counter(key(e) for e in native)
    need(len(static_counts) == 5412 and set(static_counts.values()) == {3},
         f'static task triple occurrence rank {rank}')
    start_events = [e for e in native if key(e)[:2] == (1, 0)]
    need(len(start_events) == 3, f'Graph task0 start rank {rank}')
    anchors = sorted(float(e['ts']) for e in start_events)
    buckets = [[], [], [], []]
    for event in native:
        start = float(event['ts'])
        buckets[bisect.bisect_right(anchors, start)].append(event)
    need(not buckets[0] and [len(b) for b in buckets[1:]] == [5412] * 3,
         f'temporal replay partition rank {rank}')
    for ordinal, replay in enumerate(buckets[1:]):
        need({key(e) for e in replay} == set(static_counts),
             f'static key equality rank {rank} replay {ordinal}')
        first = min(float(e['ts']) for e in replay)
        last = max(float(e['ts']) + float(e['dur']) for e in replay)
        need(first == anchors[ordinal] and
             (ordinal == 2 or last < anchors[ordinal + 1]),
             f'replay temporal separation rank {rank} replay {ordinal}')
        host = scope_by_rank[(rank, ordinal)]['stages']['target']
        need(host['start_us'] <= first <= host['end_us'],
             f'Graph anchor within Target Host scope rank {rank} replay {ordinal}')
        rows.append({'rank': rank, 'ordinal_in_profile': ordinal,
                     'nominal_cycle_if_window_exact': 64 + ordinal,
                     'model_id': 45, 'static_task_keys': 5412,
                     'native_event_count': len(replay),
                     'first_task_start_us': first,
                     'last_task_end_us': last,
                     'native_span_us_instrumented': last - first,
                     'target_host_scope_start_us': host['start_us'],
                     'target_host_scope_end_us': host['end_us'],
                     'last_device_after_target_host_scope_us': last - host['end_us'],
                     'hcom_named_native_events': sum('hcom' in e.get('name', '').lower()
                                                     for e in replay)})
need(len(rows) == 24, '24 Graph replay rows')
result = {'status': 'all8_three_native_model45_replay_occurrences_pass',
          'scope': 'Model45 native X-event completeness and temporal occurrence identity in this instrumented Level0 window. Connection_id is reused and not a replay key. Target Host scope contains each first device task but does not delimit replay completion. Nominal cycle labels depend on capture window. HCCL names count events, not bytes or necessary communication. No typed KV generation, legal overlap, observer-off time, or strict Bound.',
          'manifest_sha256': sha(MANIFEST),
          'host_scope_ledger_sha256': sha(SCOPE),
          'rows': rows}
DEST.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'rows': len(rows),
                  'native_events_per_rank': 16236}))
