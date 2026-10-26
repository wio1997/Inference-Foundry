# Run331 — refill granularity and warmed prefill prior

## Scope

Read-only, all-rank checked reanalysis of original Run287, Run188 and Run239 evidence. The frozen product is 48×32K→1024, c12, 8×910B3 DP1×TP8, DSpark7. Formal Current remains Run99 571.681 tok/s. This run does not change code in the serving path and is not a benchmark or an attainable throughput bound.

## Conditional schedule

Run330's original four-wave trajectory totals 1203 target cycles. Holding every observed request's decode-cycle duration fixed, preserving wave order, and assigning new requests to slots only when a microbatch of slots is free gives:

| Free slots gathered per refill | Zero-incremental-cost cycles | Cycles below 1203 | Refill batches |
|---:|---:|---:|---:|
| 1 | 1122 | 81 | 36 |
| 2 | 1138 | 65 | 18 |
| 3 | 1141 | 62 | 12 |
| 4 | 1149 | 54 | 9 |
| 6 | 1177 | 26 | 6 |
| 12 | 1203 | 0 | 3 |

These numbers are only scheduling sensitivity. A refill prefill may pause or compete with all 8 ranks and change every remaining request's cycle duration, route and acceptance; batching also changes the measured prefill shape. In particular, Run330's 24-cycle **slot-only** delay result is not a global prefill budget.

## Existing cost evidence

Run188's all-eight forward JSONL calls have identical token/request/mode signatures. Its warmed A/A2 no-wait cohorts used 7/8 prefill forwards with 1328/1352 scheduled tokens total; each forward handled 88–264 tokens. Summed per-call maximum-rank **Host-inclusive** forward walls were 2.737/2.969 s. The one-wait B cohort combined 12 requests into one 1000-token forward, 0.3775 s. Those calls are existing cohort preparation, not an incremental slot refill measurement. B changed decode cycles from 299 to 331, so the gross forward-wall difference was not an E2E saving.

Run239 all-eight boundaries agree in scheduled-token sequence for the four measured warmed cohorts. The current-wave execute-call sequences after the prior cohort's `[96,0]` tail have 7/11/12/10 calls, and first execute to handoff is 2.356/3.748/4.116/3.318 s. This includes scheduling, prefill and seed/handoff work; it does not isolate device occupancy or show which work may overlap decoding. The 32K logical input length cannot be treated as a new 32K prefill for each refill, but no per-request prefix-hit record is present here.

## Decision and next measurement

Keep refill as an architecture candidate. Prefer screening b=2–4 because it retains 54–65 conditional cycles with 9–18 preparations rather than 36, but do not build live slot reset from this alone. The next discriminating experiment is low-overhead all-eight phase evidence on the **original legal warmed path**: request arrival, prefix cached/residual token count, scheduled shapes, prefill device completion with collective and side-stream joins, DSpark seed ready, Host metadata/submit, and steady Graph readiness. Compare complete incremental preparation with the three existing cohort preparations and with decode resource contention; then decide whether serial insertion or prefill/decode overlap warrants a one-microbatch state/parity gate. Request arrival order must be retained; do not precompute an unarrived prompt.

Primary artifact: `granularity.json`. Source runs: Run287, Run188, Run239; historical early-refill prior is PK-018. Astra High independently reviewed the conditional model and supported measuring the complete preparation DAG before live state changes. Resource/Hardware, Scheduling-aware and Product E2E numerical bounds remain unknown.
