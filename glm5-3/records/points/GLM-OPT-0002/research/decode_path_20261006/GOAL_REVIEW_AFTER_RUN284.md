# Independent Goal Review after Run284

2026-10-07. Trigger: H11/H12/H13 have not produced a new repeatable complete-PD gain. Offline source/raw review only; no service, device, request, or frozen-artifact change. This is a route recommendation, not PERF_KEEP. Retain verified H6/H5 and target FULL; H11/H12/H13 remain off. Final product remains dynamic, complete GLM-5.3 standard PD using existing correct kernels. Unfinished API/SLA validation is not grounds to clear the research stack or stop research.

## Facts and evidence limits

Personally read Run284 measurement_reduced.json, complete A1/B1/A2/B2 SSE/events/results/workload/cache evidence, RESET_H13_CURRENT_FULL.md, Run276 offline frontier/rank0.json, actual runner batch dispatch, scheduler KV-ready promotion, and upstream graph dispatcher. Run283 correctness and Run284 terminal restoration/identity are supporting scope, not performance proof.

Run284 is raw-valid but comparison-inadmissible INCONCLUSIVE. A1/B1/A2 have 11 drafts/11 draft tokens/11 accepted/11 position-0 accepted; B2 has 12/12/11/11. All complete requests have the same 23 output IDs and stop, and matched prompt-cache accounting: P local 58/0 and external 58/0; D local 58/0 and external 58/58. Aggregate counters do not locate the extra draft or prove an unnecessary final step.

Independent token-bearing event arithmetic:

| Phase | First-to-last token arrival span, ms | D wall, ms | TPOT, ms |
|---|---:|---:|---:|
| A1 | 600.4214 | 898.215137 | 27.291881 |
| B1 | 556.4932 | 938.197469 | 25.295144 |
| A2 | 601.4989 | 896.845528 | 27.340859 |
| B2 | 602.4491 | 901.330175 | 27.384048 |

For the work-matched first pair, B1 saves 43.9282ms after first arrival, but first arrival is approximately 83.934ms later; total D wall is 39.982332ms worse. B1 successive output arrivals are mostly about 50ms versus A1 about 54ms. This is a useful steady-path signal, not repeatable complete-PD gain. B2's approximately 50ms intervals cannot be selected post hoc as a second matched success. SSE grouping/arrival intervals are not an engine-step trace. First-arrival time includes HTTP admission, KV readiness, scheduling, model execution and output publication; its difference is not an eager-compute budget.

Run276 current-FULL steady evidence is incompatible with carrying forward Run249's large per-layer eager starvation: about 52.3ms non-wait device union within a roughly 55.4ms period, with largest interior gap about 137us in the cited rank0 window. Union, inclusive task time and host synchronization duration are not removable time. Rank0 is not always the last rank. Run276 is profiled and a different epoch; these values cannot calibrate Run284 savings.

## Source-supported first-step question

Actual `pd_graph_sources/vllm__vllm__v1__core__sched__scheduler.py:2668–2674` caches successful remote blocks and, for a full prompt hit, reduces `num_computed_tokens` from `num_tokens` to `num_tokens - 1` so the last token is recomputed for sampling. This is the general scheduler path, not the Mamba-specific truncation helper.

Actual `post_full_bank_sources/model_runner_v1.py:3205–3220` requires, under DCP, computed tokens at least prompt length, query length equal to the speculative uniform length (2), and a uniform batch. The current captured descriptor is `(2, 1, uniform=True)`. The actual upstream dispatcher checks exact eligible FULL keys and returns NONE when no FULL/PIECEWISE key matches. Thus remote-KV resume has a concrete reason to take a different target execution path from steady K1 decode. This is source-supported inference; Run284's first-step descriptor and its critical-path duration were not directly witnessed by the complete-request metrics.

Do not remove the last-token recomputation, override `uniform_decode`, or submit the one-token state to the existing two-token graph merely to obtain a hit. A safe first-step graph needs the actual DCP/SFA initial-state branch, one-query metadata, KV writes, sampling and MTP consumers captured consistently. It may require code rather than a size-list change; that boundary is not yet proven closed.

## Gap ranking and route

1. **Highest-value unresolved question: PD-resume first target step and first-output critical path.** Concrete graph-coverage boundary, appreciable first-arrival envelope, and direct relevance to short complete requests. Removable amount and largest-global status remain unknown. PIVOT evidence work here before another local steady-kernel removal.
2. **Steady device framework materialization.** H13 has a plausible and observed first-pair steady benefit, but no repeatable full-request win. Park the unchanged candidate; do not treat inclusive Pad/MemSet totals as savings. Required model compute/communication occupies much of the measured steady period.
3. **Host metadata/sample/output frontier.** Still mixed and unpriced in the current FULL epoch; no source-proven larger deletion. Do not manufacture zero cost or a large new rewrite from old inclusive Python samples.

KEEP H6/H5 research stack and the evidence-led framework goal. STOP unchanged H11/H12/H13 comparisons. PIVOT the single question to whether first-step eager submission, rather than transfer/admission/publication, is a material removable part of complete-PD latency. If restarting today, I would inspect that existing trace prefix before writing another patch or building another comparison mechanism.

## Top 3 concrete code questions (ordered, not concurrent candidates)

1. At `_try_promote_blocked_waiting_request` → runner dispatch → SFA builder, what exact one-token PD-resume path executes before the first FULL replay, and is its host submission critical? This decides whether a dedicated initial-step capture contract is worthwhile.
2. In that initial state, which SFA/DCP metadata and KV/index-cache consumers differ from steady qlen2? A dedicated graph must preserve these branches and dynamic addresses; a forced descriptor is not a fix.
3. Where does the first sampled token become externally publishable relative to eager MTP and async output staging? Preserve CPU staging, ownership, error/abort and connector aggregation. Inspect only if the existing first-step prefix leaves a substantial publication interval; do not revive an early-copy patch on assumption.

## Smallest distinguishing evidence

First use the already collected Run276 full trace prefix **before the first `aclmdlRIExecuteAsync`**, not only replay-to-replay periods. Correlate the earliest target/MLA/MoE submission, connector load completion if exported, sampling/MTP and first output-copy readiness, across the relevant slow ranks. Separate device-active model work, host supply gaps, and periods whose admission/KV origin is unexported. Source-audit a synthetic full-hit scheduler state through the actual dispatch AST to confirm the NONE descriptor without a model request. This can falsify the proposed first-step mechanism or bound it using existing evidence.

If the trace lacks the necessary request/step markers, report the missing boundary rather than filling it with D_first. Only then consider one separately reset, bounded diagnostic recording KV-ready, scheduled qlen/computed count, selected mode, target begin/end and first-output-ready for the same request. No parameter scan, unchanged toggle repeat, or performance promotion follows from this review.

## Stop list / likely misreadings

- No unchanged H11/H12/H13 rerun to seek a favorable pair; no retroactive change to the frozen work gate.
- No inference of a useless trailing step from B2 counters or SSE chunks.
- No interpretation of +83.934ms first-arrival difference as candidate overhead, all eager cost, or proven removable latency.
- No transfer of Run249 eager starvation, profiled union, event/peer waits, or current synchronization wall time into a current FULL savings budget.
- No forced uniform flag, deletion of required last-token recompute, kernel rewrite, speculative offloading/early-publication change without consumer evidence.
- No new graph-bank infrastructure work merely because the existing diagnostic was expensive to establish.

The largest globally removable gap remains unknown. Recent experiments narrowed mechanisms and validated correctness, but did not establish a new complete-PD gain. That warrants a targeted change of question, not abandonment of the retained research stack or a claim that framework optimization is exhausted.
