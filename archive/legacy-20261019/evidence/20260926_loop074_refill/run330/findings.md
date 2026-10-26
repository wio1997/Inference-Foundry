# Run330 — fixed-cohort barrier versus conditional FIFO slot refill

## Source and historical prior

- Frozen Product remains 48×32K→1024, concurrency12, DeepSeek V4 W4A8, 8×910B3 DP1×TP8 DSpark7. Formal Current is Run99 571.681 tok/s; this run uses **Run287 diagnostic warmup48**, not a new formal benchmark.
- Current `FixedCohortServing.run` parks each completed slot but publishes all 12 outputs only after the slowest slot reaches the limit. The client semaphore submits the next request as the prior bulk wave returns. Run287 client wave edges confirm four 12-request waves.
- Pinned historical search of R06/R36/R37 found no direct early completion or slot refill intervention. R06 measured DP2 prefill padding, R36 removed a rarely hit SchedulerOutput deepcopy without stable service benefit, and R37 declined Draft Graph under an old eager-only flag. None decides current TP8 refill; see PK-018 and saved search JSON.

## Read-only validation

- Parsed original Run287 all8 `ownership/rank*_cohort0..3.jsonl`; every cycle has matching `active_mask` and `emitted_token_count`. Matched 8-rank FULL Graph Runtime records `cohort1..4` and warmup48 client 48/48×1024, c12.
- Four cohort durations were **294 + 297 + 297 + 315 = 1203 cycles**. First `emitted_token_count ≥1024` gives immediate completion; the observed Host mirror releases each slot one cycle later. The fixed execution has **2,266 / 14,436 parked slot-cycles** (15.70%). This is unused slot capacity, not removed compute/bytes or time.
- With all 48 observed per-request decode durations held fixed and **zero incremental** prefill/seed/Host/state/Graph cost, the abstract work floor is 1011 cycles for immediate completion and 1015 for the current one-cycle Host mirror. A specific Runtime-slot-order FIFO schedule is 1118 / **1122 cycles** respectively; the latter is 81 fewer than 1203 under these assumptions.
- Exact client `i`→Runtime slot mapping was not stored. 512 deterministic within-wave order permutations of mirror durations give zero-cost FIFO **1081–1188 cycles**, median sample1126; this sampled range is neither a rigorous bound nor a measured execution.
- Per-refill **slot-only incremental delay** sensitivity, preserving the same decode durations: delay0→1122, 8→1146, 16→1170, 24→1194, 32→1218 cycles. This delay is abstract; real 32K prefill can compete with all other slots and acceptance/routes may change. The model must compare incremental cost to existing cohort prefill, not count the entire new prefill twice.

## Interpretation and next gate

Run330 establishes a conditional Scheduling/Product opportunity large enough to investigate, but **no actual positive gain, TPS projection, Hardware Bound or Scheduling-aware attainable Bound**. The zero-cost FIFO case is an intentionally optimistic screen. Fixed durations, client order, rank-shared compute contention, next-request arrival, prompt prefill, KV/prefix ownership, DSpark seed/state, Graph capture constants and Host output publication all matter. The diagnostic Run287 trajectory differs from formal Run99.

Before a live refill implementation, measure or bound current 32K prefill marginal cost for the actual arrival pattern and audit a one-slot state generation change across Target/Draft/Host/Graph. A minimal live test is one completed slot early-return and one subsequent request seed while 11 slots continue, with all8 same-cycle decisions and unaffected-slot token/state parity. If conservative incremental cost exceeds the conditional window, pivot to another Gap without building a full scheduler.

Primary artifacts: `counterfactual.json`; source `evidence/20260926_loop064_cp/run287/{ownership,runtime,warmup48.json}`; pinned historical search JSON and PK-018.
