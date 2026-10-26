#!/usr/bin/env python3
"""Extract rank-local Host scope order from the already parsed Run611 trace.

These are profiler CPU scopes, not device-ready timestamps or a legal DAG.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
OUT = ROOT / 'evidence/20260928_loop081_bound/run611'
MANIFEST = OUT / 'offline_parse/manifest.json'
ADMISSION = OUT / 'raw_recovery_admission.json'
DEST = OUT / 'scope_ledger.json'
NAMES = ('prepare_target', 'target', 'acceptance', 'state_advance',
         'dspark_model', 'proposer', 'draft_commit')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value, why):
    if not value:
        raise ValueError(why)


manifest = json.loads(MANIFEST.read_text())
admission = json.loads(ADMISSION.read_text())
need(manifest['raw_recovery_admission_sha256'] == sha(ADMISSION) and
     manifest['status'] == 'all8_salvaged_raw_copy_offline_parse_pass' and
     len(manifest['parsed']) == len(admission['records']) == 8, 'manifest/admission')
rows = []
for parsed in manifest['parsed']:
    rank = parsed['rank']
    path = ROOT / parsed['trace_view_json']
    need(sha(path) == parsed['trace_view_sha256'], f'trace SHA rank {rank}')
    trace = json.loads(path.read_text())
    by_name = {}
    for name in NAMES:
        scopes = sorted((float(event['ts']), float(event['dur'])) for event in trace
                        if event.get('cat') == 'cpu_op' and
                        event.get('name') == 'extreme::' + name)
        need(len(scopes) == 3 and all(duration > 0 for _, duration in scopes),
             f'3 scoped cycles rank {rank} {name}: {len(scopes)}')
        by_name[name] = scopes
    for ordinal in range(3):
        stages = {name: {'start_us': by_name[name][ordinal][0],
                         'end_us': sum(by_name[name][ordinal]),
                         'duration_us': by_name[name][ordinal][1]}
                  for name in NAMES}
        need(stages['prepare_target']['start_us'] <= stages['target']['start_us'] <
             stages['acceptance']['start_us'] < stages['state_advance']['start_us'] <
             stages['proposer']['start_us'] < stages['draft_commit']['start_us'] and
             stages['proposer']['start_us'] <= stages['dspark_model']['start_us'] <
             stages['dspark_model']['end_us'] <= stages['proposer']['end_us'],
             f'Host stage sequence rank {rank} ordinal {ordinal}')
        rows.append({'rank': rank, 'ordinal_in_profile': ordinal,
                     'nominal_cycle_if_window_exact': 64 + ordinal,
                     'stages': stages})
result = {'status': 'all8_instrumented_host_scope_ledger_pass',
          'scope': 'Three Host stage occurrences/rank align by ordinal with requested cycle64..66 capture, but profiler start/stop may include cycle63 tail and stop fence. CPU scope times describe instrumented Current Host scheduling, not device completion, per-layer KV ownership, legal overlap savings, or a finite Bound.',
          'manifest_sha256': sha(MANIFEST),
          'admission_sha256': sha(ADMISSION), 'rows': rows}
DEST.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'rows': len(rows),
                  'trace_sha_verified': 8}))
