# Run545 — independent Bound review of the retained Run542 trajectory

2026-09-27. **Admit the following as posthoc instrumented Host/work accounting, using Run544's independent admission. Run542's original controller remains FAIL.** Run543 replays the retained ledger with corrected enum-string validation; Run544 independently reproduces both the original failure and corrected admission. This review would be conditional without that Run544 acceptance; its accepted report SHA is pinned in `host_metrics_review.json`. No service, NPU workload, source installation, formal E2E or optimization intervention was performed here. Formal Current remains **571.681 tok/s**.

Independent reducers `analyze.py` and `review_host_metrics.py` read the original JSONL/client/runtime data. `same_trajectory_analysis.json` contains all per-request/per-cohort results and input hashes; `host_metrics_review.json` separately checks Run546 and Host clock declarations. These scripts do not replace Run544's footer/identity/phase admission.

## 1. The output-work ambiguity actually reduced

| Same-trajectory quantity | Warmup48 | Measured48 |
|---|---:|---:|
| Rank0 cohort cycles |282,321,298,304|289,299,322,308|
| Total cycles |1205|1218|
| Effective staged q count across slots/cycles |49,527|49,462|
| Runtime retained R |49,152|49,152|
| Ordinary Scheduler tokens before bulk, G |485|401|
| G per request min/median/max |1/10/26|1/8/16|
| Scheduler-admitted Runtime bulk |48,667|48,751|
| Ordinary + admitted bulk |49,152|49,152|
| q excess above Runtime R |375|310|
| Runtime tail additionally clipped by Scheduler |485|401|

The raw q/token histories, initial positions, retained sequences and request mapping match across all8 ranks for all8 cohorts. Independent prefix reconstruction verifies `R=first1024 staged tokens` and `admitted_bulk=R[:1024-G]` for all96 requests. The measured Runtime contributes **40.02545 Scheduler-admitted tokens/cycle**, versus40.35468 Runtime-retained tokens/cycle. The401-token distinction closes a real Current useful-output accounting uncertainty; it is not401 tokens of compulsory work or a demonstrated speedup.

**The clipping direction matters:** Scheduler removes the Runtime suffix. The first Runtime token remains admitted in all96 requests and maps to complete server raw offset G (zero-based). There is no requirement to choose Runtime ordinal≥G for a witness. A Runtime ordinal j survives when `j<1024-G`, mapping to server ordinal `G+j`. The source-conditional first-position Target-argmax lemma can therefore still be used for a selected j=0. It still lacks a fresh required semantic-state witness. Exact API/client SSE correspondence establishes representation receipt; it does not make text an injective encoding of token IDs or prove literal publication of parser-suppressed tokens.

Prebulk G is not automatically G at handoff. Under the checked common Host clock declarations, ordinary Scheduler appends completed by rank0 handoff sum484 warmup and400 measured. In each phase one request's sole ordinary token is appended afterward (0.628ms /2.325ms). This narrows Host ordering only; Scheduler append, device completion and external publication remain different boundaries. Granting all eventual G as free initial output is an optimistic relaxation, not an observed initial device-ready state.

## 2. Recomputed conditional count relaxations

These calculations use only Run542's q trajectory, granting ordinary prefixes and omitting preparation/release/resource costs. They do not transfer Run287/Run239 durations.

| Cycle-domain relaxation | Warmup | Measured |
|---|---:|---:|
| Total remaining bulk /96, rounded up |507|508|
| Per-request ceil((1024−G)/8), then total /12 |509|510|
| Four fixed cohorts: sum of longest ideal8-token request chain |512|512|
| Observed q durations until1024−G, arbitrary-slot load /12 |1021|1035|
| Same observed durations, sum of longest chain in each fixed cohort |1194|1206|
| Observed q durations until full Runtime1024, slot-load /12 |1030|1043|
| Same full Runtime durations, fixed-cohort longest chains |1201|1214|
| Actually executed cycles |1205|1218|

Thus accounting for actual G changes the aggregate cardinality relaxation but leaves the fixed-cohort512 value in this instance. The measured1035 versus1206 difference identifies a conditional scheduling question, not171 removable cycles: requests cannot refill until legally released, preparation/seed/KV has a cost, batch shape and contention may change, and observed acceptance durations need not survive a new schedule. The actual1218 versus clipped fixed-chain1206 difference also cannot be multiplied by an unrelated cycle median to claim wall savings. The existing1118 FIFO and1011–1015 fixed-duration figures remain different-instance conditional results.

## 3. Preparation and Host release evidence now available

Every request has an OutputProcessor `prefill_stats` string. Parsing that **source-reported representation** gives:

- Measured:48×32768 cached prompt tokens;46 requests report83 computed residual prompt tokens, one85 and one84, totaling **3987**.
- Warmup: reported241,555 computed prompt tokens and1,335,296 cached tokens. This phase includes initial cold/partial cache conditions and is not the measured preparation workload.

These fields establish what the current output path reported. They are not native execution counters, physical traffic, proof of exact cache residency throughout the interval, or device-prefill/seed-ready timestamps. They substantially narrow the next measurement's relevant shape: the measured frontier is cached residual83–85-token preparation, not an assumed full32K prefill.

Run546's reported cross-process Host arithmetic independently matches. The ten ledger processes and both clients declare the same Linux boot ID, time namespace, zero offsets, CLOCK_MONOTONIC and1ns resolution; this supports these Host-domain comparisons under those declarations. It does not certify controller or NPU clock joins,1ns accuracy, passive timing or device readiness. The final Run546 script now enforces that Host clock identity; the independent review also checks it against original records. Final Run546 script SHA00fdad5ad3980e6c14183968e4ae3936f233f6d77aa9c4c71e98cb0ac08cebd5 and JSON SHA0c86ec1d23aae9e2cebf28d0555f05d711b56ab62cfdbf78c9651a41444763a6 are pinned in the supplementary review.

Measured request-weighted Host medians are client start→OutputProcessor add **0.669s**, add→rank0 handoff **2.004s**, handoff→rank0 done **17.242s**, and done→client end **0.369s**. Handoff→done is duplicated across12 requests/cohort; sums of these48 request durations are not Product makespan. Instrumented client wall is81.424915s; the resulting603.648 diagnostic TPS does not replace formal Current.

Within rank0's single process, previous cohort done→next handoff takes **2.904/2.900/3.366s**. All8 per-rank handoff→post-drain medians by measured cohort are16.367/16.895/18.336/17.543s. These observations contain preparation, ordinary inference, submission, synchronization, logging and waiting; they cannot be labeled pure compute, pure Host waste or removable bubbles.

Within the client alone, rank-matching the first36 completion timestamps to the36 later admission timestamps gives nonnegative delays1.709–25.032ms, median14.852ms. This is a chronology-based release relation, not a direct recorded semaphore-parent identity. It shows prompt subsequent admission on the observed path; it does not mean future request arrival can be frozen when simulating a faster publishing schedule. Delay sums overlap and must not be added as removable wall time.

The48 prompt hashes match across warmup/measured, but **zero of48 complete1024 server raw-output sequences match exactly**. Common prefixes range6–202 tokens, median13. This is not a correctness failure, a proof of fresh hidden work, or a prohibition on legitimate memoization. It only establishes that naively replaying the saved warmup raw sequence would not reproduce this measured Current trajectory; numerical/replay variability and the allowed equivalence contract remain separate questions.

## 4. Bound consequences and the next frontier

Algorithm/Resource uncertainty is reduced from unknown initial-output attribution to observed q→R→bulk clipping and ordinary-prefix counts for this acquisition. Scheduling uncertainty is reduced by actual cycles, prefix-cache/residual reports, observed client release chronology and Host preparation/publication windows. A sparse retained first Runtime token now has a joined server/raw/API representation path, subject to Run544's scope and the Run538 source lemma. **F[layer,group], compulsory traffic, strict C+/B, required communication cuts, device readiness, mixed service, feasible overlap and every finite overall Bound endpoint remain null.** No numerical Current→credible-limit distance follows yet.

**Select the all8 cached residual preparation→seed/KV-ready→first-useful-Target frontier, including its competition with a late prior-cohort Target/DSpark chain.** This selection is driven by the newly observed83–85 residual shape and repeated roughly3s inter-cohort Host transition. The Host gap's size is not its removable portion. A first measurement should distinguish actual residual/cache admission, last preparation producer completion, ordinary seed/state readiness, first required Target input readiness and consumer completion on all8 ranks, with shapes, buffers and ownership pinned. Reuse documented graph completion contracts where sufficient; do not require another exhaustive native dump by default.

The subsequent minimal mixed-service test must use the actual cached-residual/seed path and the relevant Target/DSpark active/parked state, compare compatible sequential and concurrent service under unchanged mathematical work, and validate KV/state/output ownership. A candidate successor may arrive only after legal earlier external completion; do not smuggle future inputs into an overlap model or assume publication is free. Record peak-live KV/activation/graph storage and HBM/HCCL/AICore contention. If the readiness acquisition shows preparation already hidden or dominated by a different dependency, pivot the frontier according to that result.

The measurement should return parameter envelopes for residual/seed service, legal overlap window and joint contention. Feed them into a same-trajectory resource-constrained model with endogenous c12 release, leaving unmeasured terms explicit. Strict C+/B work continues separately. Another isolated kernel census or replay of completed graph-identity experiments would not reduce the dominant uncertainty exposed here.

Historical priors remain Run537's pinned R31/R35 evidence: earlier communication launch can worsen co-running service, and shorter Host enqueue can move waiting without changing completion. Those old TP4/c1 outcomes do not decide this DP1TP8 cached-residual frontier. The required new information is whether the legal mixed frontier compresses full makespan under its actual resource and output dependencies.
