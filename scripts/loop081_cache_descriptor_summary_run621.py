#!/usr/bin/env python3
"""Compact SHA-pinned Run621 guarded diagnostic admission for Git."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
LIVE = ROOT / 'evidence/20260928_loop081_bound/run621/live'
B = LIVE / 'b'
OUT = ROOT / 'evidence/20260928_loop081_bound/run621/admission_summary.json'
NAMES = {
    'warmup_client': ('warmup_client_admission.json', 'warmup48_client_admitted'),
    'client': ('client_admission.json', 'client_two_phase_admitted'),
    'basis': ('basis_admission.json', 'diagnostic_compact_basis_all8_admitted'),
    'product': ('product_admission.json', 'product_identity_pass'),
    'dispatch': ('dispatch_admission.json', 'scoped_dispatch_identity_pass'),
    'descriptor': ('descriptor_admission.json', 'all8_actual_W0_two_cycle_cache_descriptors_only_pass'),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


admissions = {}
for key, (name, expected) in NAMES.items():
    path = B / name
    payload = json.loads(path.read_text())
    need(payload['status'] == expected, f'{key} status drift')
    admissions[key] = {'path': str(path.relative_to(ROOT)),
                       'sha256': sha(path), 'status': expected}
descriptor = json.loads((B / NAMES['descriptor'][0]).read_text())
need(descriptor['rank_count'] == 8 and len(descriptor['rows']) == 8,
     'descriptor rank coverage')
cleanup = dict(line.split('=', 1) for line in
               (LIVE / 'cleanup_status.txt').read_text().splitlines())
need(set(cleanup) == {'run_exit', 'stop_exit', 'stop_verify_exit',
                      'restore_exit', 'source_sha_exit', 'source_compare_exit',
                      'script_sha_exit', 'script_compare_exit', 'final_exit'} and
     set(cleanup.values()) == {'0'}, 'cleanup gate drift')
need((LIVE / 'source_before.sha256').read_bytes() ==
     (LIVE / 'source_after.sha256').read_bytes(), 'source restoration drift')
need((LIVE / 'scripts_before.sha256').read_bytes() ==
     (LIVE / 'scripts_after.sha256').read_bytes(), 'script restoration drift')
need((B / 'server_post_count.txt').read_text().strip() == '96' and
     (B / 'server_post_count_after_stop.txt').read_text().strip() == '96',
     'frozen request cardinality drift')
for rank, row in enumerate(descriptor['rows']):
    need(row['rank'] == rank and row['cycles'] == [64, 65] and
         set(row['cache']) == {'43', '44', '45'}, 'rank/cache geometry drift')

result = {
    'status': 'run621_all8_guarded_cache_descriptor_only_admitted',
    'run_ts': descriptor['run_ts'],
    'admissions': admissions,
    'cleanup': cleanup,
    'cleanup_status_sha256': sha(LIVE / 'cleanup_status.txt'),
    'source_before_after_sha256': sha(LIVE / 'source_before.sha256'),
    'scripts_before_after_sha256': sha(LIVE / 'scripts_before.sha256'),
    'server_post_count_before_after_stop': [96, 96],
    'cache_geometry_all8': {'format_id': 2, 'dtype': 'torch.bfloat16',
                            'shape': [34090, 32, 1, 512], 'block_size': 32,
                            'row_logical_bytes': 1024,
                            'storage_nbytes_each_layer': 1117061120},
    'scope': ('Diagnostic frozen 48+48/c12 same W0, actual two-cycle descriptor '
              'samples only. Raw admissions and service logs retained remotely. '
              'No continuous allocation lifetime, slot values, actual reader, '
              'compulsory traffic, strict Bound or formal TPS change.'),
    'formal_current_tps': 571.681,
    'strict_resource_floor_s': None,
    'strict_scheduling_floor_s': None,
    'numeric_current_to_credible_limit_gap': None,
}
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'rank_count': 8}))
