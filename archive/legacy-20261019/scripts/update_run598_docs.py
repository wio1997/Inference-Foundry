#!/usr/bin/env python3
"""Append the Run597–598 scoped Bound checkpoint and fix prior patch markers."""
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')

BLOCKS = {
    'ACHIEVABLE_BOUND.md': '''
## Run597–598 new-W₀ execution-work cardinality checkpoint

Run597 guarded 48 warmup + 48 measured/c12/1024 collected a **new diagnostic fixed-DSpark7 W₀**: all96 clients, all64 Runtime, all32 measured basis, 97 source hashes, full Target-state replay and sparse actual Target/DSpark/Draft model-input checks pass; tagged stop, all8 idle and five-source restore pass. The observer perturbs timing. Astra independently checked the live identity and cleanup. Four measured cohorts total1200 cycles, 115,200 current physical Target-8 rows, 99,976 active-slot class rows and 15,224 parked-slot rows. Internal staged accepted tokens total49,523, including371 beyond the 49,152 product outputs due to Host mirror lag. The first postpark cycle is witnessed in every cohort.

Run598 pins the installed source-static Q7 and sparse live Q7/context96/anchor inputs, then records conditional current Draft query100,800 issued /87,479 active-class rows, context combine115,200 issued rows and three-layer context KV345,600 layer-row incidences. These are current execution geometry, **not compulsory fresh evaluations or bytes**. For Target alone, Run569 coefficients yield active-class conditional BF16 `wo_a`288.499 T conventional ops and routed MoE1,298.245 T W4A8 GEMM-equivalent ops; neither is a strict W-minus. The study narrows new-W₀ workload cardinality and role/dependency identity, while Run99 same-state, full prefill/seed/KV/communication necessary work, exact-board cumulative `C⁺/B` and an all8 timed legal DAG remain open. Resource/Hardware, Scheduling/Execution, Product finite endpoints and numeric Current→credible-limit distance stay null. Formal Current571.681 tok/s. See Run597–598 findings and PK-095/096.
''',
    'PERFORMANCE_MAP.md': '''
## Run597–598 fixed-work ledger and Bound priority

One new guarded measured W₀ has a validated all-cycle accepted/draft/Host-park basis and sparse actual Target/DSpark inputs, with full client→Runtime joins and all8 restoration. It records1200 cycles, 99,976 active-class /115,200 submitted Target-8 rows and source-static Q7 conditional Draft geometry. This closes a new-trajectory cardinality/identity uncertainty only; current rows cannot become compulsory fresh work or a time Bound. The next Bound gate is semantic necessity plus actual mixed resource capacity and a typed, timed all8 Target/Draft/KV/Host dependency cut. Historical R21 and Run579/580 warn that source-independent overlap can lose to contention. Formal Current571.681 tok/s; strict numerical Resource/Scheduling/Product endpoints remain null. See PK-095/096.
''',
    'PROJECT_STATE.md': '''
## Run597–598 Bound-first checkpoint

Run597 live guarded W₀ acquisition passed 96-client/64-Runtime/32-basis all8 joins, sparse input checks and clean service/source restoration. Run598 reduced 1200 cycles into current Target and conditional Draft row classes; 99,976 active-class versus115,200 physical Target-8 rows, with49,523 staged accepted and48 explicit Host parks. This is a new observer-perturbed diagnostic W₀, not Run99 same-state or formal TPS. Continue compulsory-work/traffic and exact-board attainable-capacity acquisition in parallel with all8 timed dependency/mixed-contention evidence; do not promote active rows to W-minus. Formal Current571.681 tok/s, numerical credible-limit gap unknown.
''',
    'RESULTS.md': '''
## Run597–598 diagnostic fixed-work ledger

Run597 all8 guarded live W₀ passed client/Runtime/basis admission, source restore and idle. Run598 offline current execution ledger:1200 cycles; Target physical115,200 / active-class99,976 / parked15,224 rows; conditional Draft query issued100,800 / active-class87,479 rows. Internal staged49,523 includes371 tail overshoot. No formal performance measurement or strict finite Bound; Current Formal571.681 tok/s. See Run597–598 findings and independent review.
''',
}

FIX = {
    'ACHIEVABLE_BOUND.md': '+## Run593–596 broad coverage',
    'PERFORMANCE_MAP.md': '+## Run593–596 Bound uncertainty update',
    'PROJECT_STATE.md': '+## Run593–596 Bound-first checkpoint',
    'RESULTS.md': '+## Run593–596 offline Bound calibration',
}

for name, block in BLOCKS.items():
    path = ROOT / name
    original = path.read_text()
    if block.strip() in original:
        raise ValueError(f'already appended: {name}')
    marker = FIX[name]
    if original.count(marker) != 1:
        raise ValueError(f'prior literal patch marker mismatch: {name}')
    before, after = original.split(marker, 1)
    after = marker + after
    after = ''.join(line[1:] if line.startswith('+') else line
                    for line in after.splitlines(keepends=True))
    path.write_text(before + after.rstrip() + '\n' + block)
    print(name, 'updated')
