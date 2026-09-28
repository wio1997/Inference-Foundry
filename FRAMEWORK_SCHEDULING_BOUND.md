# Framework/Scheduling-Only Bound — V0

## Contract and answer status

This branch fixes the current correct DeepSeek V4 Flash W4A8 primitive/kernel implementations, their shape-dependent measured service costs, DP1×TP8 on 8×910B3, and the DSpark7 acceptance/output trajectory. It permits changes to execution order, buffering, prefetch, Graph organization, asynchronous orchestration, pipelining, overlap, request scheduling and Host/Device control. It excludes faster kernels, different algorithmic acceptance, fewer target/draft cycles and changes to model work. A fused replacement that changes primitive cost belongs to a separate operator branch; a Graph organization that changes launch/synchronization cost belongs here.

The recorded future trajectory is an **audit input**, never a Runtime oracle. A proposed legal schedule may prefetch only values available from prior state and declared inputs; it cannot use future accepted tokens, routing decisions or logits before their producer executes. Repeats of the frozen greedy product can produce different trajectories, so each acquisition is a separate W₀ unless all trajectory/output checks pass.

The requested full-Product numerical interval is **not yet admitted**. This V0 gives an executable makespan model and identifies the missing measurements. It does not wait for a compulsory-work proof or exact-board Hardware C⁺. The formal Current is Run99 **571.681 output tok/s**, 49,152 outputs in the paired median repeat, **85.977964 s** client wall and **1,206** Runtime cycles. The same repeat has 69.039655 s rank0 Runtime-scope sum and 16.938309 s arithmetic remainder. The latter is not a disjoint Host interval or removable time. See `evidence/20260928_loop081_bound/run630/product_hurdle.json`.

## Two legal schedule graphs

For each frozen cohort and cycle, instantiate Target, acceptance, state/KV advance, DSpark context/query, Host count-copy, Output and Product boundary work at the *existing* primitive implementations. Assign each instance a measured shape-conditioned cost interval and resource occupancy. Keep output count and the recorded acceptance/count/ID trajectory fixed for each compared W₀.

* `E_must` includes actual value/version/last-writer→first-consumer dependencies, TP collective order, buffer lifetime, prior draft/state→next Target, request release, and Product output order. A disputed edge remains unknown. An observed Python or Graph stream order alone does not become a mandatory dependency.
* `E_safe` retains current order wherever removability is unproved. A schedule using it is executable only after the actual Runtime/Graph and Product correctness gates pass. The formal Current is the initial complete witness; a simulated or synthetic schedule is not an executable upper witness.

For fixed primitive costs `d` and measured resource-contention policy `R`, calculate an optimistic makespan lower bound `L = max(longest_path(E_must,d), resource_load_lower(R,d), release_to_output_chain)`. Calculate a realizable upper witness `U` from a correct, repeated Product run of a legal schedule. Then `L ≤ T_best_legal ≤ U` and, for 49,152 outputs, `49,152/U ≤ TPS_best_legal ≤ 49,152/L`. Unknown edges and perturbed task costs are represented as alternative declared graph/cost cases, never as silently absent dependencies. A full-Product number is admitted only if one frozen W₀ has task identity/cost, all8 resource contention and Product timing in a compatible measurement scope.

## Current evidence and limits

| Evidence | Usable for schedule model | Limit |
|---|---|---|
| Run99 / Run630 | Formal complete Current witness and 85.978 s Product wall; 69.040 s rank0 Runtime-scope arithmetic component | Three formal repeats span 6.413 s in client wall; Runtime scope is not a device occupancy measure |
| Run625 DAG V0 | 82-node partial two-cycle value/conditional/current-order graph | Target internals, metadata ancestry, cache generations, all8 HCCL and Product boundary incomplete; node costs null |
| Run246 / Run592 | Real native task timing for a selected partial MatMul→RS→copy→HC-post chain | Historical Level1, separate W₀, only two occurrences/rank; no full-cycle extrapolation |
| Run579/580 | Fixed GMM/HCCL fixture cost under contention: about 10.3 ms GMM and 4.2 ms HCCL isolated; 14.4–14.6 ms serial, 16.2 ms concurrent | Synthetic activation/real weights/routes, fixture scope; `max(GMM,HCCL)` is not a proven joint service time |
| Run611–613 | Same-acquisition selected Target Graph and eager DSpark native task ownership/flow | Level0 observer, only a local W₀ window, HCCL completion and full critical path unresolved |
| Run602/606/633/634 | Same-W₀ Product/Runtime and preparation Host ownership, request join, actual Graph modes | Instrumented Current; Host intervals overlap other resources and are not removable time |
| Run638 | Controlled OFF/ON/OFF cycles63–65 current/side-stream packet, effective work, Runtime bulk IDs and client text/usage | Each arm admitted separately; cross-arm fixed-W₀ failed, so no observer-effect or timing transfer; stream markers do not certify native/HCCL completion |

Historical R26 (now PK-122) is a direct scheduling counterexample: a true-shape, one-layer DSA event schedule shortened local NPU time by 12.77–15.34% with exact Q/KV outputs, yet its old DP2×TP4 c1 short-service TPOT changed only 20.3→20.2 ms amid overlapping run distributions. This is a prior about E2E non-conversion, not a current DP1×TP8 verdict.

Run639 makes the first local fixed-observed-cost calculation from the 16 admitted Run592 historical rank/occurrence chains. MatMul+RS+copy observed duration sums are 28.241–32.181 µs (median 29.40075); matched current envelopes are 30.320–33.441 µs (median 31.7695); the **paired** envelope-minus-sum is 1.2195–3.6195 µs (median 2.1600). This is a conditional *maximum gap inside that local chain if those observed durations are frozen*, not an established saving: RS may include peer wait, other legal dependencies are unproved, and the trace is a different W₀. `evidence/20260928_loop081_bound/run639/local_fixed_cost.json` pins every row and the Run592 source SHA.

Run638 completed three 48+48/c12 arms with clean all8 stop and five-source restoration. Its cross-arm validator correctly rejected fixed-W₀ equivalence: measured cohort cycles were OFF_A 278/300/290/307, ON 311/300/314/302, OFF_B 291/289/301/298, and OFF_A versus ON had 0/48 identical client output texts. The post-run arm-scoped reducer independently read all client raw/SSE, all64 Runtime records per arm and all32 ON Event packets. The packet field named `product_output_sha256` hashes Runtime bulk token IDs, not final Scheduler/API token IDs; client raw/SSE text and usage pass independently, while final API token-ID parity was not re-proved. For ON's 96 selected rank-cycles, **instrumented current-stream** median cycle span is 55.538 ms, Target-forward span 46.901 ms, proposer span 6.346 ms, and paired residual outside those two spans 1.707 ms. The existing Host count-copy wait median is 0.013 ms. Paired Target+proposer span sums are 52.103–55.631 ms (median 53.331); the paired residual C−(T+D) is 1.605–4.871 ms (median 1.707). Keeping the two opaque spans serial would give only a *restricted, instrumented* local schedule bracket `[T_i+D_i, C_i]`, not a full-framework bound. These are one observed W₀'s early cycles, not primitive intrinsic service, exposed savings, whole-W₀ averages or formal Run99 timing. `evidence/20260928_loop081_bound/run638/arm_scoped_recovery.json` preserves the failure and per-arm admission.

Run640 independently replays rank-local Run611 Level0 task-time intervals across three selected Target FULL Graph occurrences and intervening work. Physical kernel/copy active unions occupy 145.593–202.083 ms of rank windows 182.684–209.446 ms; the interval complement ranges 6.907–37.090 ms. `EVENT_WAIT` unions overlap physical activity heavily: physical∩wait is 140.946–198.821 ms, wait outside physical only 6.173–7.103 ms. Adding wait durations to kernel durations would double count. These are instrumented Current rank-local intervals; they do not prove that the complement is removable, that a kernel held all resources, or that cross-rank clocks align. See `evidence/20260928_loop081_bound/run640/rank_local_current_timeline.json`.

The fixture contention result requires a resource policy with at least serial and jointly active alternatives. A model that substitutes `max(10.3,4.2)=10.3 ms` for the tested pair would contradict observed service. Conversely, freezing an HCCL event duration that includes peer wait as intrinsic communication service would hide a scheduling opportunity.

The following are **wall-saving hurdles**, not proposed bounds or available savings, for the formal 49,152-output median sample:

| Candidate TPS | Required complete Product wall | Required saving from 85.977964 s |
|---:|---:|---:|
| 600 | 81.920 s | 4.058 s |
| 650 | 75.618 s | 10.360 s |
| 700 | 70.217 s | 15.761 s |
| 750 | 65.536 s | 20.442 s |
| 800 | 61.440 s | 24.538 s |

This table lets each proposed schedule name the exact whole-Product saving it must explain. The 16.938 s arithmetic remainder alone is neither a source of that saving nor a hard limit on it; Runtime ordering can change as well.

## Same-W₀ Target fixed-stream relaxation (Run641)

Run641 pins all8 Run611 `trace_view.json` SHA and partitions three Model45 Graph occurrences per rank, each exactly5,412 tasks. With fixed observed physical task durations and stream assignment but all cross-stream waits/edges erased, the longest-stream physical sum is36.905–38.083ms against the current instrumented Graph span45.102–72.209ms; the *conditional* difference is8.193–34.186ms, median15.819ms across24 graphs. This is an intentionally optimistic restricted relaxation, not a legal schedule or Product saving. The longest physical stream is stream1 in all24. All HCCL-named physical tasks are on stream0, total5.792–31.714ms/Graph (median13.666ms); their intervals have zero overlap with stream1 physical work and are wholly covered by stream1 wait intervals in the recorded schedule. The conditional gap covaries with HCCL-named duration, largely through an algebraic identity (difference minus HCCL ≈2.398ms median), so that correlation is not independent causal evidence; peer wait may be included and dependencies/resource interference may force seriality. Run579/580 measured concurrent GMM/HCCL slower than serial in a separate fixture. The next decisive task is edge and service typing, not subtracting this gap from formal wall.

## Admission ladder and next calculation

1. Keep Run638's ON arm as an instrumented within-W₀ stage census only. Its three arms cannot support observer-effect transfer or cross-arm performance benefit because trajectory/output diverged.
2. Reuse existing Run246/611 native events and Run579/580 contention to construct a **local fixed-primitive schedule subproblem** with explicit `E_must` and `E_safe`, cost intervals, current native makespan, optimistic legal makespan and contention alternatives. Run639/640 are the first cost/replay checks; do not combine different W₀ as if they were one trace.
3. Expand task ownership over a representative complete cohort and its prefill/seed/Output boundary with low-overhead all8 evidence only where the local model shows a large uncertain makespan term. Keep a matched OFF/ON/OFF control. Then evaluate 49,152-output Product interval and convert to Framework-only TPS.

Current provisional answer: **the achievable Framework-only TPS interval is not identifiable from admitted evidence yet**. The most consequential uncertainties are full-cohort dependency coverage, intrinsic versus wait-contaminated HCCL service, Target/DSpark/Host side-stream overlap and the Product time outside Runtime scope. No kernel or acceptance optimization is required to close these scheduling uncertainties.

## Framework-only first-collective arrival and precursor — Run643–644

In the same Run611 Level0 diagnostic W₀, all8×3 first Model45 ReduceScatter events match high-level count49,152/BFP16/MESH-RING-NHR. Arrival spread is26.733/9.739/9.783ms, while finish spread is12–21µs; rank7 arrives last with a34–35µs native duration. The first collective begins49–55µs after each Graph entry. Early-rank HCCL task duration is thus strongly consistent with **peer wait**, not an intrinsic frozen communication cost. The first occurrence may include profiler startup disturbance. Cross-rank clock alignment is conditionally supported, but not independently proven to microsecond precision.

Run644 joins the two adjacent Graph transitions: prior exported Graph last-event spread52/56µs, next Graph entry spread9.741/9.785ms, rank7 latest both times. The skew reappears between observed Graph boundaries. Previous proposer Host-scope end and next Target Host start spread much more, but those Host scopes overlap native execution; rank5 proposer Host end precedes the prior exported Graph last event by15.567ms. No disjoint Host cost or removable time is inferred. **Framework-only full Product TPS remains unidentified**; neither early-rank waiting nor the Run641 conditional gap can be multiplied by cycle count. Next resolve rank7 pre-submit necessary predecessor/wait lineage from existing API/native trace, then only the missing bounded all8 device-ready/enqueue measurement. See Run643/644 findings and independent Astra reviews.
