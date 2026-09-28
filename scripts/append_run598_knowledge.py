#!/usr/bin/env python3
"""Append two scoped Run597–598 Performance Knowledge entries."""
import json
from pathlib import Path

path = Path('/data/wio/Inference_Foundry/performance_knowledge/entries.jsonl')
entries = [json.loads(line) for line in path.read_text().splitlines() if line]
assert entries[-1]['id'] == 'PK-094'
new = [
    dict(id='PK-095',
        topic='New fixed-DSpark7 W0 compact lineage can join Host parks and real Draft input roles',
        mechanism='Capture initial state, every accepted/masked count and next-draft state, explicit Host park events, and sparse actual Target/DSpark/Draft inputs after ordinary cohort drain.',
        environment='DeepSeek V4 Flash W4A8, 8×910B3 DP1TP8, DSpark7, 48 warmup + 48 measured/c12/1024; Run597 observer-perturbed new diagnostic W0.',
        observed='96 client, 64 Runtime and 32 all8 basis files join; 1200 cycles, 99976 active-class Target-8 rows and 115200 current physical rows; sparse postpark witness in all four measured cohorts; 13 CPU fault controls and all source/idle cleanup gates pass.',
        failure_or_limit='Observer changes timing; new W0 is not formal Run99 same-state. Role/input identity does not prove fresh necessary arithmetic, prefill/seed/KV, compulsory bytes or a legal timed DAG.',
        revalidate_when='Pair with semantic reuse and actual Target/Draft/KV state readiness; a formal same-state acquisition is needed before transferring row cardinality to Run99.',
        extreme_relation='Narrows diagnostic workload and dependency identity; formal Current571.681 tok/s and strict Resource/Scheduling/Product endpoints remain unchanged.',
        source=[dict(repository='Inference_Foundry',ref='run597',path='evidence/20260928_loop081_bound/run597/findings.md'),
                dict(repository='Inference_Foundry',ref='run597',path='evidence/20260928_loop081_bound/run597/live/b/basis_admission.json')],
        status='current_diagnostic'),
    dict(id='PK-096',
        topic='Target and Draft current row geometry is a workload ledger, not compulsory work',
        mechanism='Reduce the Run597 admitted basis with a source-static Q7 and sparse live Q7/context96/anchor check, separate issued, active-slot, parked, accepted and padding classes.',
        environment='Run598 offline reducer on Run597 new observer-perturbed W0; frozen DP1TP8 DeepSeek V4 W4A8 DSpark7.',
        observed='Target issued115200/active99976/parked15224 rows; Draft query issued100800/active87479 conditional rows; staged accepted49523 includes371 beyond product output; 48 Host park events.',
        failure_or_limit='The three-layer Draft context KV 345600 layer-row figure is source-model conditional. Active rows may include reused semantic work; issued rows are not actual FMA or HBM bytes. No formal Run99 W0 equivalence, complete compulsory census or numeric Bound.',
        revalidate_when='Prove same-state fresh semantic work and model-global versus TP-replicated operations; measure matching exact-board C+/B and mixed scheduling dependencies before any time/TPS endpoint.',
        extreme_relation='Narrows new-W0 cardinality but leaves Formal Current571.681 tok/s and finite Resource/Scheduling/Product endpoints null.',
        source=[dict(repository='Inference_Foundry',ref='run598',path='evidence/20260928_loop081_bound/run598/summary.json'),
                dict(repository='Inference_Foundry',ref='run598',path='evidence/20260928_loop081_bound/run598/findings.md')],
        status='conditional_resource_relaxation'),
]
with path.open('a') as output:
    for entry in new:
        output.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
print('appended', [entry['id'] for entry in new])
