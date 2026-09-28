#!/usr/bin/env python3
"""Rebind offline parse manifest to final Run611 raw admission, without reparsing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
ARM = ROOT / 'evidence/20260928_loop081_bound/run610/live/b'
RUN611 = ROOT / 'evidence/20260928_loop081_bound/run611'
OUT = RUN611 / 'offline_parse'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise ValueError(message)


admission_path = RUN611 / 'raw_recovery_admission.json'
admission = json.loads(admission_path.read_text())
manifest_path = OUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
need(admission['status'] == 'all8_cohort5_raw_profile_salvaged_after_invalid_validator_clock_gate', 'admission')
need(manifest['status'] == 'all8_salvaged_raw_copy_offline_parse_pass', 'parse status')
need(len(admission['records']) == len(manifest['parsed']) == 8, 'rank count')
for record, parsed in zip(admission['records'], manifest['parsed']):
    rank = record['rank']
    need(parsed['rank'] == rank, 'rank order')
    source = ARM / record['raw_session']
    copy = OUT / 'input' / f'rank{rank}' / source.name
    for path in source.rglob('*'):
        if not path.is_file():
            continue
        key = str(path.relative_to(ARM))
        need(digest(path) == admission['raw_sha256'][key], f'original drift: {key}')
        need(digest(copy / path.relative_to(source)) == admission['raw_sha256'][key],
             f'copy drift: {key}')
    for kind in ('task_time', 'trace_view'):
        path = ROOT / parsed[f'{kind}_{"csv" if kind == "task_time" else "json"}']
        need(digest(path) == parsed[f'{kind}_sha256'], f'parse output drift: rank {rank} {kind}')
manifest['raw_recovery_admission_sha256'] = digest(admission_path)
manifest['reconciled_after_clock_gate_correction'] = True
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'status': manifest['status'], 'ranks': 8,
                  'admission_sha256': manifest['raw_recovery_admission_sha256']}))
