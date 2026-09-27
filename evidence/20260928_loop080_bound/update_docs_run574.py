#!/usr/bin/env python3
"""Append the narrowly scoped Run574 Bound checkpoint once."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
APPENDS = {
    'ACHIEVABLE_BOUND.md': '''
## Run574 native row-coordinate conditional certificate

The Ascend910B sparse-attention source/build/package audit pins BF16/F16 kernel objects and their embedded hashes, SCFA/SWA TND Query-prefix and output-offset formulas, and the source template's `FLASH_DECODE=0` allowlist. For the reviewed local native path, Query and attention output use the same token/head coordinate. The generated compiler target is Ascend910B1; actual 910B3 production object/tiling selection and source→object reproducible-build provenance remain unproved. This narrows one Run573 row-identity subgate but does not certify production DSA A2A, wo_b, MoE or 43-layer composition. No compulsory-work numerator, capacity C⁺/B, Scheduling floor, Product ceiling or numeric Current→Bound distance is promoted. Current Formal remains571.681tok/s. See Run574 findings and independent Astra review.
''',
    'PERFORMANCE_MAP.md': '''
## Run574 source-level native attention row order

Reviewed SCFA/SWA TND `FLASH_DECODE=0` source writes attention output at the Query token/head offset; package objects match build objects and the container imports the audited source tree. This is a conditional local mapping only. Actual loaded B1-targeted binary on 910B3, tiling/metadata, Graph generation, DSA TP8 A2A/wo_b and all43 router rows are still open. Resource numerator and Current→credible-limit numerical gap remain unknown; next evidence should bind real loaded branch and one same-trajectory first-post-parking row/weight witness. See PK-075 and Run574.
''',
    'PROJECT_STATE.md': '''
## Run574 conditional native row-order checkpoint

Run574 offline audit and Astra High independent replay PASS at source/build/package scope. SCFA/SWA TND native output shares Query coordinates under supported `FLASH_DECODE=0`; the compile command targets Ascend910B1, and real 910B3 loaded object/tiling plus source→object build provenance remain open. No live service or formal E2E was run. All strict finite Bound endpoints and Current→credible-limit distance stay null; Formal Current571.681tok/s. Next bind actual loaded op/branch and implement a reversible same-trajectory first-post-parking all43 row/expert/packed-format witness; separately continue mixed-resource and exact-board C⁺/B tracks.
''',
    'RESULTS.md': '''
## Run574 native Query coordinate audit (2026-09-28)

Source/build/package hash and negative gates passed; independent Astra replay reproduced the output byte-for-byte and rejected eight extra mutations. The reviewed TND SCFA/SWA native path preserves Query token/head output coordinates, conditional on actual production dispatch. There was no model or NPU run and no E2E measurement. The Ascend910B1 compiler target and unresolved live binary/tiling binding preclude a production row-identity claim. Formal Current remains571.681tok/s; all finite Bound endpoints remain unknown. Evidence: `evidence/20260928_loop080_bound/run574/`.
''',
    'HANDOFF.md': '''
## Run574 Bound checkpoint

Run574 source/build/package native row-coordinate audit and independent Astra replay PASS with scope limits. Script SHA5db08117…, JSON SHAff86c527…; build generators target Ascend910B1 and production 910B3 object/tiling selection is unbound. The `FLASH_DECODE=0` TND SCFA/SWA source preserves Query coordinates, but it does not provide the first-post-parking all43 row map. Service remains stopped; borrowed sources remain pristine. Next bind actual op-api/tiling/kernel load and replay generation, branch/prefix/group metadata, then a guarded same-trajectory row/expert/weight collector; retain independent all8 mixed-resource and strict C⁺/B work. Formal Current571.681tok/s, every finite strict endpoint null.
''',
}

for name, section in APPENDS.items():
    path = ROOT / name
    old = path.read_text()
    marker = section.splitlines()[1]
    if marker in old:
        raise ValueError(f'already appended: {name}')
    path.write_text(old.rstrip() + '\n\n' + section.strip() + '\n')

path = ROOT / 'performance_knowledge/entries.jsonl'
old = path.read_text()
if '"id": "PK-075"' in old:
    raise ValueError('PK-075 already exists')
entry = dict(
    id='PK-075', topic='Native sparse-attention source preserves TND Query coordinates under conditional dispatch',
    status='conditional_source_contract',
    environment='DeepSeek V4 Flash W4A8 DP1TP8, Ascend910B3 server, CANN9.1; installed vLLM-Ascend custom-op source/package with Ascend910B1 compiler target; no live service run.',
    mechanism='SCFA/SWA kernel uses cu_seqlens_q prefix and Query tile offset for both input and attention output; supported template set and host tiling source select FLASH_DECODE=0.',
    observed='Run574 pins source/build/package hashes, two packaged objects with embedded SHA parity, container source bind/import path and negative gates. Independent Astra replay is byte-identical and rejects eight additional mutations.',
    failure_or_limit='Source→object reproducible build, actual B3 loaded op-api/tiling/kernel/Graph generation, selected layout/metadata and TP8 DSA/MoE all-layer composition are unproved. This is not a router-row semantic certificate, compulsory traffic, latency or Product Bound.',
    revalidate_when='Loaded custom-op binary, tiling branch, source hash, CANN build or device changes; bind real runtime object/tiling and one all8 first-post-parking trajectory before semantic expert incidence.',
    extreme_relation='Narrows one Resource numerator identity uncertainty while leaving formal Current571.681tok/s and every finite Hardware/Resource, Scheduling/Execution and Product endpoint null.',
    source=[dict(repository='Inference_Foundry', path='evidence/20260928_loop080_bound/run574/native_row_contract.json', ref='source/build/package audit'),
            dict(repository='Inference_Foundry', path='evidence/20260928_loop080_bound/run574/astra_review.md', ref='independent scoped review')],
)
with path.open('a') as f:
    f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + '\n')
print('Run574 docs and PK-075 appended')
