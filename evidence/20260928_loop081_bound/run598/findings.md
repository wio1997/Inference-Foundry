# Run598 — current execution row ledger from Run597

The offline reducer requires Run597's all8 basis admission and all-zero cleanup status, SHA-pins the four measured rank0 basis files, and pins the restored installed DSpark source. Source-static `num_query_per_req` plus all sparse live Q7/anchor/context96 witnesses justify a **conditional current execution geometry** for the 1,200-cycle new W₀:

| Row/step class | Current issued | Active-slot class | Interpretation |
|---|---:|---:|---|
| Target verification input | 115,200 | 99,976 | 15,224 parked-slot rows remain in current physical shape |
| Draft query input | 100,800 | 87,479 | One three-layer Draft forward per cycle, not seven complete Draft passes |
| Draft context combine input | 115,200 | 99,976 | Current fixed 96-row input geometry |
| Draft context KV projection, three layers | 345,600 | 299,928 | Layer-row incidences, not bytes or unique values |

The active context contains 49,523 staged accepted rows and 50,453 width-8 padding rows. The staged total exceeds 48×1,024 output tokens by 371 because the Host mirror can park after device overshoot. There are 48 Host park events. The 336 last-cycle Draft query rows across four cohorts are terminal issued work with no next Target cycle; their semantic removability has **not** been proven.

For the Target subset alone, applying the source-audited Run569 coefficients as a conditional illustration gives active-class `wo_a` **288.499 T conventional BF16 ops** and routed MoE **1,298.245 T W4A8 GEMM-equivalent ops**; current issued geometry gives **332.430 T** and **1,495.937 T**, respectively. These are separate arithmetic classes and omit other Target, Draft, KV, prefill, seed, shared expert and communication work. Active-slot rows are not a compulsory fresh-work numerator; the issued geometry is not actual kernel FMA count or compulsory traffic.

The result narrows **new-W₀ workload cardinality**, but not the historical Run99 W₀. It makes Target→Draft context, acceptance→next query seed and Draft→next Target state edges concrete enough to plan a fixed-work scheduling cut, while leaving readiness/completion timestamps and mixed resource contention unmeasured. Run579/580 and historical R21 already show that independent operations can slow under overlap; any new cut must be tested with real dependency readiness and a mixed-service makespan. Strict Resource/Hardware, Scheduling/Execution and Product E2E endpoints, plus numeric Current→credible-limit gap, remain null. Formal Current is **571.681 tok/s**.
