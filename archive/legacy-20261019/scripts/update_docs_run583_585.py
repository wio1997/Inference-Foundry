from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
notes = {
    'ACHIEVABLE_BOUND.md': '''## Run582–585 fixed-W₀ dual-bound checkpoint

Run582 was INVALID before readiness or any POST: its compiled decoder consumer hook invoked `os.getenv`/logging under Dynamo fullgraph. Tagged stop, all8 idle and five-source restoration succeeded; it is an instrumentation incompatibility, not a model/performance result. Run583 removed that hook and admitted the frozen 48+48/c12/1024 diagnostic, all64 eight-rank Runtime files and eight selected FULL96 capture rows. The loaded layer0 path is `SequenceRowParallelOp` with BF16 `[96,4096]` local partial (786,432B), `[12,4096]` RS result (98,304B), and a distinct attention destination. All8 branch, geometry, alias and stop/restore gates pass. The capture has no unique cohort/request join and its first replay mark is submission only; native RS ordinal, typed last writer, HC-post consumer and readiness times remain unproved. This closes a structural Scheduling branch uncertainty but gives no finite time floor.

Independent Astra High Run584 keeps Run569 conditional `wo_a` arithmetic, Run576 operand geometry, Run578 current read traffic and Run579–580 attained mixed service distinct from compulsory work and exact-board maximum cumulative capacity. Run585 designs one measured logical Target→BF16 `wo_a` group-row witness under a declared ordinary dense online class; one such fresh row would conditionally contribute 8,388,608 conventional ops, but has not been captured. Rated/sampled clock and further ordinary service tests do not substitute for `C⁺/B`. Formal Current remains Run99 median **571.681 tok/s**. Strict Resource/Hardware, Scheduling/Execution and Product E2E endpoints and numerical Current→credible-limit distance remain unknown. Next obtain native producer/RS/copy/HC-post identity and one scoped positive W⁻ witness; only then measure legal ready/completion timing or promote conditional resource work. See Run582–585 evidence and PK-083/084.
''',
    'PERFORMANCE_MAP.md': '''## Run582–585 fixed-work Bound map

Run583 all8 FULL96 capture resolves the actual layer0 `wo_b` implementation branch and capture-time partial→RS-result→attention-destination object lineage. Its `[96,4096]` partial and `[12,4096]` result are typed geometry, not compulsory link bytes. The first replay was submitted, not device-completed, and capture rows are not joined to unique measured requests. Run582's compiled hook failure is invalid instrumentation. Native collective, actual last writer, HC-post input and timing remain the Scheduling frontier; no Product gain is implied.

Run584 independent Resource review identifies a narrower positive W⁻ acquisition: one logical, measured Target evaluation→actual BF16 `wo_a` group-row under declared cache/reuse and ordinary dense online assumptions. Run585 is source-only preflight; no W⁻ is yet promoted. Strict exact-board cumulative `C⁺/B` remains missing, so the finite Resource/Scheduling/Product Bound and numerical Current→limit gap remain unknown. Formal Current571.681 tok/s. See PK-083/084 and Run582–585 evidence.
''',
    'PROJECT_STATE.md': '''## Run582–585 checkpoint (2026-09-28)

Run582 invalid startup instrumentation was stopped/restored without requests. Run583 diagnostic completed 96 client requests, all64 Runtime cohort records, all8 selected FULL96 layer0 branch/alias captures and final stopped/source-SHA admission. Astra independently rechecked 82 hashes, validator negatives and the important absence of capture→unique request and native last-writer proof. No latency or formal E2E result was promoted. Run584 independent Resource review and Run585 source preflight target one conditional logical `wo_a` W⁻ witness; exact-board `C⁺/B` remains unresolved. Formal Current571.681 tok/s; all finite strict Resource/Scheduling/Product endpoints and numeric Current→limit gap remain unknown. Next inspect existing Graph/profiler correlation for native RS/HC-post identity, then collect only missing production DAG fields; pursue the one-row Resource witness in parallel. See `evidence/20260928_loop081_bound/run582`–`run585`.
''',
    'RESULTS.md': '''## Run582–585 Bound evidence

Run582 INVALID before health/POST due to Dynamo fullgraph rejecting an inserted compiled decoder logging hook; clean stop/restore. Run583 structural PASS: frozen diagnostic96 requests, all64 Runtime files, eight FULL96 capture rows, exact stop/restore. Actual layer0 loaded branch is BF16 `SequenceRowParallelOp`/FlashComm1 RS; partial `[96,4096]` 786,432B, result `[12,4096]` 98,304B, distinct attention destination. No unique capture→request, native writer/consumer or device completion/time identity. Run584 Astra Resource review and Run585 source-only `wo_a` witness design add no numerical performance result. Formal Current Run99 median571.681 tok/s; strict finite Bound endpoints and Current→credible-limit gap remain unknown.
''',
}
for name, body in notes.items():
    path = ROOT / name
    s = path.read_text()
    heading = body.splitlines()[0]
    if heading in s:
        raise RuntimeError(f'duplicate heading: {name}')
    path.write_text(s.rstrip() + '\n\n' + body)

pk_path = ROOT / 'performance_knowledge/entries.jsonl'
existing = pk_path.read_text()
ids = {json.loads(line)['id'] for line in existing.splitlines() if line.strip()}
entries = [
    {
        'id':'PK-083',
        'topic':'Run583 FULL96 loaded DSA wo_b row-parallel branch is a scoped Scheduling DAG identity witness',
        'mechanism':'A capture-time typed input/partial/reduced/destination association can select the real TP branch before assigning native tasks or proposing overlap. Source order or identical buffer address alone cannot prove a device last writer or completion.',
        'environment':'DeepSeek V4 Flash W4A8, 8×Ascend910B3, DP1TP8, fixed DSpark7; instrumented 48 warmup+48 measured c12/1024; CANN9.1 FULL96 selected Graph; Run583.',
        'observed':'All8 final admission: 96 unique requests, 64 Runtime files, 8 capture rows, stopped/idle and four-source/script SHA restoration. Loaded SequenceRowParallelOp/AscendUnquantizedLinearMethod, DSA-CP/SP/FlashComm1 on, MMRS/OTP off, pad0, TP ranks0–7. Local BF16 partial [96,4096] 786432B, RS result [12,4096] 98304B, projected=result, distinct attention destination. Astra independently rechecked 82 admission hashes and validator negatives.',
        'failure_or_limit':'Capture rows are not joined to a unique measured cohort/request; ordinal1 records replay submission, not device completion. Native HCCL ordinal, actual last writer, copy task, HC-post consumer, timing and same-W0 marker perturbation remain unknown. Run582 compiled decoder hook was invalid before POST. Geometry is not compulsory communication bytes or Product time.',
        'revalidate_when':'First seek existing Graph/profiler source-to-native task correlation; if absent, capture one selected measured FULL96 with cohort/request/generation, native RS/copy/HC-post typed storage and rank-local ready/completion, with marker and same-state controls before timing or a legal schedule claim.',
        'extreme_relation':'Closes loaded branch uncertainty for a one-layer Scheduling slice. Strict Resource/Hardware, Scheduling/Execution and Product E2E endpoints and numeric Current-to-limit gap remain null; Current Formal571.681tok/s.',
        'source':[{'repository':'Inference_Foundry','ref':'run583','path':'evidence/20260928_loop081_bound/run583/final_admission.json'},{'repository':'Inference_Foundry','ref':'run583','path':'evidence/20260928_loop081_bound/run583/findings.md'},{'repository':'Inference_Foundry','ref':'run582','path':'evidence/20260928_loop081_bound/run582/findings.md'}],
        'status':'scoped_observation',
    },
    {
        'id':'PK-084',
        'topic':'Fixed-W0 Resource work needs one semantic fresh group-row plus matching cumulative capacity',
        'mechanism':'Source arithmetic, observed physical rows, current traffic and attained service are separate proof classes. A conditional W-minus can be established for one semantic fresh BF16 wo_a group-row under an explicitly declared ordinary dense online evaluation class; a strict time floor additionally needs exact-board cumulative C-plus/B.',
        'environment':'DeepSeek V4 Flash W4A8, 8×Ascend910B3, DP1TP8 and frozen DSpark7; Astra High Run584 review of Run569/576/578/579–580 and source-only Run585 preflight.',
        'observed':'Run569 formula is 8,388,608 conventional ops per fresh wo_a group-row. Run585 locates active BF16 npu_transpose_batchmatmul in dsa_cp, with conditional [96,1,4096] input, [1,4096,1024] weight, [96,1,1024] output and one rank-local group; no live witness yet.',
        'failure_or_limit':'Observed execution or a physical96 row does not establish mandatory fresh W0 work. Pre-window reuse, parked/padding, rank/group ownership, Graph generation and semantic evaluation membership must be checked. Run578 core-side read is not compulsory HBM. Ordinary GEMM/HCCL service and rated/sampled clock are not exact-board capacity upper certificates. No finite strict Resource time endpoint.',
        'revalidate_when':'Capture exactly one measured logical Target evaluation joined to its wo_a group-row, actual replay completion, source/weight identity, freshness and permitted reuse with isolated scratch; reject stale/padded/alias cases. Independently obtain authoritative exact-board cumulative C+/B for the same class before time conversion.',
        'extreme_relation':'Narrows the next Resource numerator uncertainty without changing acceptance/cycles or claiming a time gain. Formal Current571.681tok/s and numeric Current-to-credible-limit distance remain unknown.',
        'source':[{'repository':'Inference_Foundry','ref':'run584','path':'evidence/20260928_loop081_bound/run584/astra_resource_review.md'},{'repository':'Inference_Foundry','ref':'run585','path':'evidence/20260928_loop081_bound/run585/preflight.md'},{'repository':'Inference_Foundry','ref':'run569','path':'evidence/20260928_loop080_bound/run569/findings.md'}],
        'status':'conditional_source_contract',
    }
]
for entry in entries:
    if entry['id'] in ids:
        raise RuntimeError(f'duplicate PK: {entry["id"]}')
with pk_path.open('a') as f:
    for entry in entries:
        f.write(json.dumps(entry, ensure_ascii=False, separators=(',', ':')) + '\n')
print('docs and PK entries appended')
