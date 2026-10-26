# Run346 — Dual-Bound V3 conditional-frontier audit

Status: read-only model calibration, no service run or performance KEEP. Current Formal remains Run99 median **571.681 tok/s** (49,152 output tokens / 85.978 s median wall). Its three repeats used 1217/1212/1206 Runtime target cycles. The best supported Product ceiling interval is still **[demonstrated 571.681 tok/s, finite upper endpoint unknown]**; this interval is a statement about evidence, not a claim that 571.681 is the optimum.

Executable artifact: `scripts/extreme_bound_calibration_v3.py` → `run346/bound_calibration_v3.json`, pinned to Run287/329–331, Run339/340, Run99 and `dual_bound_v2.json`. It leaves the numeric Algorithm/Resource, Hardware, Scheduling and Product latency floors null when inputs are incomplete.

## Distinct cycle constraints

| Run287 diagnostic relaxation | Cycles | Meaning |
|---|---:|---|
| Fixed durations, immediate completion | 1011 | `ceil(12122 occupied slot-cycles / 12)`; capacity relaxation, no schedule constructed. |
| Fixed durations, Host-mirror release | 1015 | `ceil(12170 / 12)`; capacity relaxation, no schedule constructed. |
| Zero-cost FIFO, immediate completion | 1118 | Constructed refill trajectory under unchanged per-request duration and zero incremental preparation; neither a lower bound nor an implemented run. |
| Zero-cost FIFO, Host-mirror release | 1122 | Same constructed trajectory with one-cycle later release. |
| Observed Run287 warmup | 1203 | Actual four-cohort diagnostic trajectory, **not** Run99 formal. |

The separate **512** cardinality floor is four current zero-output handoff cohorts × `ceil(1024 / 8)` cycles. It does not assume perfect acceptance is attainable or establish cycle latency. A new handoff must recompute `r_i` from the real remaining product output and account for work already done before handoff. For a fixed cohort, `C >= max_i ceil(r_i/8)`; with arbitrary refill, `C >= max(max_i ceil(r_i/8), ceil(sum_i ceil(r_i/8)/12))` is only a capacity relaxation.

## Explicit missing cost and causality

The Run340 b2/b3/b4 cross-run sensitivity has gross time budgets 3.669–3.742/3.500–3.569/3.048–3.109 s at the Run99 observed cycle-time reference. If the same 6.935 s later-wave preparation as Run332 were required, at least 46.0–47.1% / 48.5–49.5% / 55.2–56.0% of it would have to be hidden merely to break even, before publication, state transaction, changed acceptance or contention. These are **screening inequalities**, not feasible throughput predictions. In the frozen client, each successor request can exist only after an earlier HTTP stream ends; the legal DAG includes completion → all-rank publication → client stream end → actual next arrival → admission → prefix-conditioned residual prefill and KV/state → DSpark seed-ready → first new Target. Run343 fence and Run344 offline protocol close only two implementation questions; Run345 live continuation remains unmeasured.

Resource evidence is partial: Run242's routed GMM sample 155.676 GFLOP/rank-cycle and Run256 Compressor 58.385 GFLOP are not the complete necessary arithmetic. Run247's 18.950 GB read/2.380 GB write per rank-cycle (whole-window median) are **current** graph traffic; the 18.965 GB read is the sum of family medians. Neither is compulsory traffic. Run250 reported collective payload is not physical link bytes. Run339 isolated official HCCL Test measures five warmed padded88 prefill shapes, not concurrent mixed-decode service or a physical bandwidth ceiling. Run99 client-minus-Runtime residual and Run266 HCCL event union cannot be subtracted as savings.

## Bound ladder and next evidence

For a specified legal architecture `a`, the strict Resource/Hardware time relaxation is the maximum of necessary compute/credible capacity **upper** bound, necessary HBM bytes/credible bandwidth **upper** bound, and necessary network-cut bytes/credible link capacity **upper** bound. Measured attained bandwidth is a feasible service point, not a capacity upper bound. A Scheduling relaxation adds the semantic critical path and all-rank collective join; an executable schedule must also satisfy resource sharing, stream, Graph address, storage lifetime, and event constraints. Product wall adds finite-request arrival, prefix/preparation, publication and drain. Different architectures may trade resources, so their individually best terms cannot be spliced into one fictional schedule.

The highest-value next measurement is a **single correlated formal 48-request work/dependency ledger**, rather than another isolated kernel timing: request ID and client arrival/publication; prefix hit and new prompt work; last prefill and full seed-ready; each cycle's useful accepted tokens, active shape, route and KV length; Target/DSpark resource counters; collective all-rank input-ready/service/join; and final client output. Use passive instrumentation without a new synchronization, then pair any timing intervention with correctness and repeated formal E2E. This ledger will identify the largest term preventing a finite credible Product interval. An independent Astra High review agrees that no finite whole-product TPS upper endpoint is presently defensible.
