# Run326 — private complete MoE Graph parked routed48 screen

## Frozen context and prior

- DeepSeek V4 W4A8, 8×910B3, DP1×TP8, DSpark7, c12 48×32K→1024. Formal Current remains Run99 median 571.681 tok/s. This run is a **12×1024 eager diagnostic carrier**, not formal E2E.
- Historical pinned R14/R35 were consulted before Loop072 (PK-013/014); their DP2×TP4 verdicts are not imported. Run322 confirmed current layer4 local12/global96 ALLGATHER geometry; Run323 constrained the local-only parked-row opportunity. Run325 eager B completed but A/A had intrinsic non-bit-exact drift, requiring this private Graph control.

## Experiment

- Opt-in reversible SHA-guarded bootstrap hook, source restored to `f644bd14ac1cb9c8365ba2464abbff989d7f2c4c2d58279a8e18a5bce918716f` after the run. All 8 Runtime ranks and 12/12 requests of 1024 output tokens completed; runner exit 0, service stopped.
- Same real layer4 prestate at cycle 253, active mask `[1,1,0,0,1,0,1,0,0,1,0,1]`, six active requests and local active rows `[12,4,4,4,8,0,8,8]`.
- A and A2 are independently captured original full-MoE Graphs; A_repeat replays A. B keeps original hidden/router AllGather, shared-expert path and finalize reduce-scatter, but selects the global48 active routed rows after the existing gather and scatters routed output back to global96 before original finalize. Four-arm sequence repeated 3 warmup plus 10 measured triplets. Complete layer4 MoE endpoint timed with NPU Events and rank-rendezvous Host timestamps. Active indices were prepared outside Graph, so dynamic mask/index and graph selection costs are excluded.

## Result

| Complete layer4 endpoint, median of 10 triplets | A | A_repeat | B | A2 |
|---|---:|---:|---:|---:|
| Max-rank device ms | 0.51219 | 0.53186 | **0.57209** | 0.51844 |
| Rank-rendezvous Host ms | 0.83833 | 0.84570 | **0.89574** | 0.83403 |

- B was strictly faster than all three original controls in **0/10** device triplets and **0/10** Host triplets. Independent paired reanalysis of the raw samples gives max-rank B−mean(A,A2) **+35.58 to +95.89 µs**, median **+49.585 µs**, slower in 10/10 triplets. At this actual parked state, this post-gather routed48 shape/scatter implementation worsened the complete MoE endpoint despite excluding dynamic selection cost.
- Active routing expert IDs and weights were exact for B versus A in all measured triplets; independent A2 routing IDs were also exact. All compared tensors were finite. Graph A versus eager A had max absolute difference 0.0078125. B active-output max absolute difference by rank was 0.00390625–0.0078125, but only **1/7 nonempty ranks** stayed within the predeclared A/A_repeat/A2 envelope jointly on max absolute, RMS and signed mean. Thus candidate correctness is **unestablished**. Zero-active rank5 is vacuous for active-output comparison.
- Private Graph capture/replay passed on all ranks; checker explicitly verified final sync and inactive capture state. The diagnostic output throughput of about 82.197 tok/s is **not** comparable to formal FULL Graph throughput.

## Decision and bound effect

**REJECT this specific parked post-gather routed48 implementation** for live Target integration: it has no complete-MoE endpoint benefit and its numerical screen does not establish correctness. This does not reject all parked-tail schedules or prove proximity to a Hardware/Resource, Scheduling-aware or Product bound. No numeric bound update; formal Current stays 571.681 tok/s. Pivot to a larger full-active MoE scheduling exchange: investigate moving replicated FP32 router computation after the already-required hidden AllGather to eliminate the logits AllGather, subject to weight parity, same-prestate routing and complete-endpoint tests. Historical R11/R35 remain hypothesis priors only.

Primary artifacts: `fixture_check.json`, `fixture/rank*_cohort1.json`, `runtime/rank*_cohort1.json`, `bench12.json`, `patch.json`, `restore.log`, `stop.log`.
