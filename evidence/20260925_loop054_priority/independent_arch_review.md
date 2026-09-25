# Loop054 independent architecture and priority review

Date: 2026-09-25. Advisory input for Sol; no KEEP/PIVOT verdict is made here.

## Scope and provenance

Configured reviewer: `gpt-6-astra`, reasoning effort `high`, `fork_turns=none`, confirmed by the parent agent's spawn configuration. Actual backend response model ID is not exposed to this reviewer and is **unverified**. Configuration is not proof of backend identity.

Read-only repository/source/evidence review on SSH host `61.241.77.34-60008`; observed main HEAD `9a1aa6e86fb309fa336bc6504c61b25997fecc18`. The working tree already contains active TaskCtl changes and historical untracked evidence. This review changes only this requested advisory file. No service, device experiment, source edit, TaskCtl mutation, Run creation, or Git commit was performed.

Read AGENTS.md, MISSION.md, current HANDOFF.md through Run209, resume-pack generated 2026-09-25T11:06:00Z, the state/bound/performance/result documents, and the specific evidence below. Several state documents retain older opening summaries; the latest HANDOFF and resume-pack establish current chronology.

The governing product is DeepSeek V4 Flash W4A8, 8×910B3, DP1×TP8, DSpark7, frozen warm-cache 48×32K→1024/c12 protocol. Run99's formal Extreme median 571.681 tok/s versus historical Stock 543.655 remains the current recorded point. No later diagnostic supersedes it. The achievable full-product bound remains unknown.

## Recommendation

The highest expected-value *next bounded experiment* is an **exact-input, single-layer, single-shape prefill MoE subgraph replay feasibility and latency test**, subject to a short source-closure preflight. Start with the first padded88 eager prefill signature already captured in Run208. Keep DSA and all attention/KV/compressor/indexer operations outside this boundary.

This is an experiment proposal, not a claim that a stateless or capturable boundary has already been proven. The selected MoE body's hidden context, routing inputs, communication mode, multistream dependencies and possible mutable load counters must first be enumerated. Failure of that small closure gate is a useful negative result; it must not expand into whole-prefill ownership redesign.

I challenge the proposed generic return to “distinguish target GMM from exposed communication.” That question has already been investigated repeatedly. A new target run is justified only by a named changeable mechanism that existing evidence cannot resolve. Another decomposition of the same Run107 windows or another duration-spread measurement has low decision value.

## What the existing evidence actually establishes

| Evidence | Observation | Decision implication |
| --- | --- | --- |
| Run116 | Profiled GMM sum 9.966 ms; communication not overlapped by compute 10.248 ms; non-GMM compute union lower screen 29.979 ms | These are exposure descriptions, not separately removable product costs. |
| Run145 | First reduce-scatter starts differ 20.533/8.650 ms across ranks, ends differ only 0.010/0.0135 ms; latest entrant spends about 0.035 ms; remaining 259 HCCL calls sum median 5.205 ms | Shortening early-rank waiting does not advance collective completion. There is no established independent 10 ms transfer opportunity. |
| Run148/150 | One-card product-shape GMM reads 1.055× packed active W1 and 1.075× W2 bytes, at about 1118/995 GB/s effective in-kernel read bandwidth | Empty-expert skipping and packed-weight traffic are already largely effective in these sampled cases. This is not hardware peak proof, but removes the obvious easy GMM hypothesis. |
| Run151 | 62 compressor calls equal 21 c4×2 +20 c128; attention/indexer state products are distinct | Mandatory work counts cannot be promoted to redundant work. A fusion needs an actual intermediate/layout mechanism. |
| Run152 | Adjacent BF16/FP32 collective pairs exist, but second FP32 calls sum 0.257 ms/window | Coalescing has a small gross screen and dtype/layout costs. It cannot explain a large product gap. |
| Run195 | Steady per-cycle rank duration spread median 0.090 ms target, 0.063 ms proposer; target p90 spread 2.329 ms | Large profiled rank skew is not established as a persistent product imbalance. Duration equality still does not prove zero absolute arrival skew. |
| Run186/203 | Nine eager forwards sum 3.424 s maximum-rank wall versus 3.416 s thread CPU; MoE median about 3.52 ms/layer, 43 bodies/forward | Active submission remains a sizeable, observed product-stage exposure. It still needs an intervention to measure its removable fraction. |
| Run200/202 | Prefill same-stream variant was locally slower; isolated event-pair Host cost makes a few redundant pairs only about 12 ms over nine forwards | Event deletion is exhausted at this scale. It does not falsify graph replay of the entire MoE dispatch body. |
| Run208/209 | First padded88/one-request call repeats, all 107 observed addresses/layouts stable, eight selected integer hashes change; whole-prefill capture crosses real state writes | Whole-prefill capture requires a difficult state contract. This does not establish that a smaller MoE-only graph has the same KV ownership burden. |

The “GMM versus communication” chain includes Runs117–119, 129, 136–145 and 192–195. Run148/150 added the missing traffic evidence. None proves target work is optimal, but reopening these families needs new information: a legal replacement kernel, measured avoidable traffic, a supported tiling/layout change, or a specific dependence that can overlap. Their large historical sums alone are no longer a selection argument.

## Ranked candidates

1. **Prefill MoE-only replay feasibility.** Largest remaining directly observed Host family with a narrower state surface than DSA. A mechanism changes execution organization while retaining required kernels. A single real-weight layer tests graph/collective/multistream feasibility and exact input refresh before multiplying work. Confidence: moderate in experiment value, low in realized product gain until measured.
2. **One specific target fusion/layout intervention, if a source-backed implementation exists.** HC-pre/RMS or output-layout chains may remain possible, but no current audit establishes a replaceable sequence and parity-preserving candidate. Collect concrete operator support and byte/dependency evidence before a service run; do not start with an arbitrary 5 ms family threshold.
3. **Broader prefill native submission or whole-forward graph.** Potential gross envelope is sizeable, but Run209 correctly identifies broad state and capture-warmup costs. Reconsider after the smaller MoE experiment demonstrates savings and graph memory/amortization viability.
4. **Target GMM retuning or arrival/communication remeasurement without a new mechanism.** Low expected information gain after the existing route, traffic, collective, and timing audits.
5. **Already screened small mechanisms:** static expert remap, c12 tail compaction, c128 compressor scatter elimination, admission wait, and event/stream simplification. Their current screens or diagnostics do not justify the next broad implementation.

These are experiment priorities, not permanent exclusions. A fixed 5 ms per-family hurdle can reject every useful composable change. Conversely, small kernel savings should not trigger expensive architecture work without a credible whole-product return.

## Concrete next experiment and gates

### 1. Bounded source-closure preflight

Select one real layer's existing `vllm::moe_forward_shared` implementation boundary in the first warmed padded88 eager forward. Record the *actual local MoE tensor shapes*, not an assumption that every rank's MoE input has88 rows.

Source anchors inspected:

- Borrowed `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:152` delegates the custom op to a resolved layer's `_forward_impl`; `:641` onward retains transforms, reductions, output scaling and combination around this custom-op boundary.
- Borrowed `vllm_ascend/ops/fused_moe/fused_moe.py:1000` enters `_sequence_parallel_context` then shared/routed implementations.
- `:719–803` consumes forward context and `_EXTRA_CTX`, prepares/finalizes communication, routes and computes; `:779–797` can mutate EPLB counters.
- `:823–951` has required shared/routed stream waits and final stream completion; `:953–998` creates events and returns the shared/routed outputs.
- Upstream `moe_runner.py:598–610` can change DP/SP local-size context; `:71–89` layer lookup can increment the context's MoE layer index.

Therefore do **not** equate “no attention KV arguments” with “pure function.” Freeze/refresh the complete actual input set, including hidden states, router logits, shared input, token IDs or hash-routing inputs if reachable, padding/local-size fields, comm-mode flags and layer identity. Confirm which optional EPLB/LoRA/static-index branches are active. Keep outer MoERunner transforms/reductions in their original order. Explicitly account for output aliasing and lifetimes.

The preflight passes only if all device reads/writes and Host control fields can be bounded within the one layer and scratch buffers. If it requires DSA metadata ownership or live KV restoration, stop this candidate rather than silently enlarging it.

### 2. One-shape correctness and replay feasibility

Use one legal warmed Extreme service, full48 warmup and legal12×1024 cohorts, all eight ranks. Allocate one graph per rank for one chosen layer/signature. Preserve the original multistream/HCCL dependencies; do not combine this with event removal, changed routing, different weights, or capture of DSA.

Capture from isolated copies of real inputs with persistent output/scratch storage. Complete all graph-associated work before a compared output is read or scratch reused. Restore any enumerated mutable counter/context/scratch state affected by capture warmup. Capture itself must never become an extra live model transition.

Test at least two real input snapshots of the same signature with different tensor values/routing. Replay B between eager A and restored eager A-prime, then repeat the first input after the second. Require:

- Same expert IDs/counts and all discrete routing/padding fields, with complete snapshots/restores.
- Shared and routed output shapes, dtype, alias contract and values compared to eager self-replay; quantify any floating-point differences, and reject an unexplained candidate-only divergence above that floor.
- Changed input values demonstrably change replay outputs and match their corresponding eager outputs; pointer reuse alone is insufficient.
- Identical collective order and participation across all8 ranks; no deadlock and no stale event/output dependency.
- No live KV writes in the captured subgraph and no change to weights, request state or outer layer-index progression.

This experiment may be diagnostic-only until these checks pass. It should preserve the normal serving result by using the eager output during initial shadow testing.

### 3. Measure the actual mechanism before expanding

Use warmed, repeated A/B/A-prime timing on identical copied real inputs and the same collective order. Report Host wall/thread CPU, device event completion including both streams, and latest-rank end-to-end segment latency. Charge persistent-buffer input/output copies and replay overhead to B. Collect timing outside the correctness-copy interval. Do not insert a per-operator sync or torch profiler into the serving comparison.

A first padded88 call occurs once per measured cohort in Run208. Even removing all43×3.52 ms of its MoE Host envelope is only roughly0.151 s gross, **not** a demonstrated >=0.5 s gain. The one-layer88 test is a feasibility discriminator, not itself a formal E2E candidate.

Advance only if replay saves a repeatable amount above its own A/A-prime variation and does not slow device completion. Then compute a conservative expansion screen from actual shape frequency and measured savings on additional representative signatures. Do not multiply one layer's result across all43 layers and every shape as proof. Roughly43×7×2 ms would be0.602 s/cohort *if* two milliseconds survives at each occurrence; that is a conditional sizing example, not an expected gain.

Bound graph memory/capture time and bank size before expansion. No hundreds-of-graphs implementation on the strength of one successful graph; collect allocation/pool sizes and shape recurrence first. Unsupported/missing signatures fall back to eager without delaying admission.

### 4. Integration and final validation

Only after the small experiment passes, test enabled/disabled/enabled order controls on matched prefill signatures and full legal cohorts. Record latest-rank prefill completion, client TTFT/envelope, runtime cycles, useful accepted tokens and all8-rank Host/state gates. Preserve the DSpark7 contract and account for acceptance/cycle variation; do not interpret output-hash differences between independent cohorts as a standalone failure or success.

A local Host reduction that remains hidden behind required device work fails the mechanism's product test. An effect that needs broad DSA/KV redesign fails the intended bounded scope and requires a new explicit architecture decision.

For a real candidate, correctness must pass before frozen formal warm-cache48-request repeated E2E. Compare against current Extreme control, not merely historical Stock. Include warmup, graph preparation amortization under the frozen protocol, TTFT/TPOT and run spread. Sol retains the final KEEP/REJECT/PIVOT decision.

## Sources and evidence limits

Primary paths, relative to repository unless otherwise stated:

- `AGENTS.md`, `MISSION.md`, `HANDOFF.md`, `PROJECT_STATE.md`, `PERFORMANCE_MAP.md`, `ACHIEVABLE_BOUND.md`, `RESULTS.md`, `tasks/deepseek-extreme-p0/resume-pack.json`.
- `evidence/20260925_loop039_gmm/run116/priority_audit.json`.
- `evidence/20260925_loop044_target/run145/comm_gmm_audit.json`.
- `evidence/20260925_loop044_target/run148/counter_analysis.json`; `run150/counter_analysis.json`; `run151/dsa_chain_audit.json`; `run152/tp_collective_audit.json`.
- `evidence/20260925_loop050_target_dependency/run195/analysis.json`.
- `evidence/20260925_loop053_native_prefill/run203/analysis.json`; `run208/analysis.json`; `run208/interpretation.md`; `run209_source_audit.md`.
- `evidence/20260925_loop052_prefill_submission/run198/source_audit.md`; Run200/202 conclusions and Run184/186/188, Run180/191/197 summaries in current HANDOFF and resume-pack.
- Borrowed source roots: `/data/wio/vllm_ascend_26/framework/vllm` and `/data/wio/vllm_ascend_26/framework/vllm-ascend`.

Assumptions: frozen flags and restored borrowed source are the intended control; no unreviewed candidate in the active working tree invalidates these source anchors; small-subgraph graph capture and HCCL/stream support remain unproven until the proposed experiment. No external hardware peak specification is used. The report ranks information and implementation value under present evidence; it does not infer a throughput ceiling from failed local hypotheses.
