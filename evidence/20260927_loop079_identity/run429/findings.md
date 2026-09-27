# Loop079 Run426–429: reviewed Host cutoffs and V3.13 Bound calibration

## Validity and measured boundary

Run427 was a clean, instrumented **diagnostic** of the frozen 48+12, c12, exact-1024 streaming workload: 60 distinct server POSTs, five cohorts, 40/40 eight-rank FULL Graph reports, source restoration, idle NPUs and a stopped service. Four buffered Host hooks covered the ModelRunner pre-run probe, Scheduler ordinary and terminal appends, OutputProcessor receive/queue, and Chat raw-token consumption/generator yield. Every one of 376 ordered segments matched its `(length, SHA256)` across Scheduler→receive→queue→Chat, with causal timestamp order. Clock origins shared one boot ID, time namespace and `CLOCK_MONOTONIC` domain. Astra's independent Run428 raw review accepted these gates and the corrected reducer. This run's instrumentation perturbed Host timing, so it supplies no formal TPS.

At each cohort's **eight-rank H_probe envelope** (sum of five separate cohort cutoffs), Scheduler committed output `G` is **[795,796]** raw token IDs; terminal pre-bulk `G` is 796. Chat had consumed `A=[674,796]` raw IDs. Generator-level generated-output yield count is **[213,268]**. At least 50 requests had completed, in aggregate, at least 213 yields containing nonempty reasoning deltas before their cohort's earliest H_probe. This disproves an all-zero pre-H_probe generator-output premise for this run. It does not assign one visible token to each raw ID.

`H_probe` is a pinned Host hook **before** `FixedCohortServing.run`, not exact run entry or device-ready time. `G` is Scheduler commitment; `A` is raw Chat/parser consumption; `Y` is completed generator yield of chunks. None is device-completed reusable work `D_i(H)`, literal raw-token publication `p_i(H)`, ASGI send, or client receipt. The 60 client JSON files establish exact completion and 1024 output tokens, not time-joined client publication. Do not carry Run427's 796, trajectory (1488 total cycles) or cutoff to Run421, Run403 or formal Run99.

## Dual-Bound effect

- **Algorithm/Resource:** Host output accounting now has request-correlated committed and consumed intervals for this acquisition. The compulsory in-window device work numerator remains open because `D_i(H_run)`, all43 router row identity and Draft dependencies are unproved. Runtime-retained IDs cannot be treated as all new model work.
- **Hardware/Resource:** No new 910B3 matching capacity upper cap or compulsory traffic certificate. The Run423 `W⁻/C⁺` strict-ceiling gate remains the smallest logical route to a loose numerical endpoint.
- **Scheduling/Execution:** The 376-segment Host path has verified causal order, but it lacks Graph/device completion, all-rank resource contention and exposed wall-time. No overlap or scheduling saving is inferred.
- **Product E2E:** Formal Current is Run99 median **571.681 tok/s**. The V3.13 model leaves every finite latency floor and Product ceiling **null**; the achieved best single formal sample 612.962 tok/s remains a point, not a repeatability guarantee.

The next Bound experiment should first close the all43 row identity with current installed native CANN row-order/branch and selected FULL Graph tensor/address evidence, while pursuing a genuine SKU-matched capacity upper cap. Another Host ledger alone would mostly repeat facts already closed. If a narrower device/ASGI boundary is later required, specify which Bound certificate it changes before instrumenting it.

Executable calibration: `scripts/extreme_bound_calibration_v3_13.py` → `run429/bound_calibration_v3_13.json`. Source: Run426 design, Run427 raw+analysis, Run428 independent review. No performance intervention or formal E2E promotion occurred.
