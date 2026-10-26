from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
blocks = {
    'PROJECT_STATE.md': '''
## Run567–569 Bound-first checkpoint (2026-09-28)

Run567's realistic DSpark group-map CPU gate and independent Astra source review passed. Run568 guarded one-arm pause diagnostic completed warm48+measured48/c12/1024 and all8 admission. At measured cohort5 natural first completion cycle187 slot4, the enumerated Target/Host/Draft/Serving state was unchanged across the pause-only transition on all8 (60 tensors, 175,975,620 cloned bytes/rank). Full physical KV and opaque metadata, fixed-W₀ continuation, early API release and mixed successor work remain unproven. Client96 POST, tagged stop, eight idle NPUs, source restore and SHA gates all passed. No formal E2E or numeric Bound update.

Run569 independently reviewed V0 fixed-work census records 344 BF16 `wo_a` groups, 258 W4A8 expert incidences and current Compressor geometry, with 16 separate Run247–249 counter windows. Its Run566 A0 trace is diagnostic W₀ only. All compulsory HBM, complete work, exact-board C⁺/B and finite Resource/Hardware, Scheduling/Execution and Product endpoints remain null. Formal Current is Run99 median **571.681 tok/s**. Next prioritize a formal-window fresh-required-row/expert-ownership witness and matching capacity/residency, plus all8 concurrent attained service in an Engineering column; keep pause/legal-release as a conditional schedule question rather than the automatic next live run. See Run568/569 findings and PK-069/070.
''',
    'PERFORMANCE_MAP.md': '''
## Run568/569 Current→Bound classification

Run568 eliminates one narrow same-S* pause mutation concern: the 60 enumerated tensors and captured scalars/identities were unchanged on all8 at cohort5 cycle187 slot4. It does **not** measure removable execution time or certify full KV/fixed-W₀ continuation, normal API release, successor ADD or all8 mixed service. The legal schedule and Product E2E gap remain unknown.

Run569 separates per-fresh-row BF16 `wo_a` and W4A8 routed MoE formulas, current 96-row geometry, checkpoint operand footprints, and 16 instrumented current counter windows. The 18.965GB-class current read cannot be promoted to compulsory traffic; HCCL link bytes remain unavailable. The largest Bound uncertainty is now broad formal W₀ required work plus exact-board cumulative capacity/residency and mixed resource service/critical-path timing. Formal Current571.681tok/s; numerical Current→credible-limit gap and every finite strict endpoint remain null.
''',
    'ACHIEVABLE_BOUND.md': '''
## Run568/569 scoped Scheduling gate and Resource census V0

The Run568 same-cohort natural S* control passed eight-rank captured-field noninterference at cycle187 slot4, after fences and pending-work checks. This is a scoped scheduling state certificate, not a latency floor, legal release schedule or fixed-W₀ continuation proof. Full Target/Draft KV content and opaque metadata were not certified. The guarded controller passed client/admission/stop/restore/source SHA gates.

Run569's reviewed table enumerates 344 BF16 `wo_a` groups (8,388,608 conventional ops per fresh group-row; 2,885,681,152 per fresh full-model row), 258 W4A8 routed expert incidences (50,331,648 GEMM-equivalent ops per expert-row; 12,985,565,184 per fresh full-model row), and 58,384,711,680 nominal Compressor projection ops/rank-cycle at current 96-row geometry. These are conditional formulas and current geometry, not complete formal necessary work. BF16 `wo_a` checkpoint footprint 2,885,681,152 bytes and one active W4A8 expert-layer packed footprint 12,582,912 bytes are operands, not compulsory HBM reads. Run247–249's 16 counter windows remain separate current-path observations, with missing HCCL link bytes. Source/admission/runtime SHA, config/header identity and per-window finite/count/byte reconciliation gates passed, with Astra independent review.

No positive formal-window W-minus, compulsory-HBM lower, exact-board cumulative C⁺/B or complete all8 fixed-W₀ dependency DAG is yet certified. Thus all finite strict Hardware/Resource, Scheduling/Execution and Product E2E endpoints and numeric Current→credible-limit distance stay null; Current Formal remains Run99 median571.681tok/s. The next Resource step is fresh required rows and expert ownership for a broad formal-window subset, then matched capacity/residency. Engineering attained all8 service is a separate empirical column, never an absolute hardware maximum. See Run568/569 findings and PK-069/070.
''',
    'RESULTS.md': '''
## Run568/569 diagnostic evidence (2026-09-28)

Run568 no-publication single-arm diagnostic passed warm48+measured48/c12/1024, all8 client/runtime/trace admission and cleanup. At cohort5 cycle187 slot4, 60 enumerated tensors and captured state were unchanged across pause-only handling; full KV/fixed-W₀ and numerical performance implications are unproven. Run569 offline conditional work/traffic census passed expanded negative gates and Astra independent review. Neither is a formal E2E run. Formal Current remains **571.681 tok/s**; all finite strict Bound endpoints remain unknown.
''',
}
for name, block in blocks.items():
    path = ROOT / name
    text = path.read_text()
    marker = block.strip().splitlines()[0]
    assert marker not in text, f'already appended {name}'
    path.write_text(text.rstrip() + '\n\n' + block.strip() + '\n')
print('updated', ', '.join(blocks))
