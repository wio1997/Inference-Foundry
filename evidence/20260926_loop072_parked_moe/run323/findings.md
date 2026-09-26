# Run323 — parked row balance screen from existing Run287 masks

Read-only reuse of Run287 original FULL Graph ownership masks, 1,496 sampled cycles across five cohorts; no new service, kernel or model output.

For each cycle, map 96 global Target token rows to contiguous TP8 blocks of 12 and eight token rows per request, then count active rows on each rank. The 678 cycles with at least one parked request include 579 cycles whose busiest rank still owns all 12 rows. Only 99/1,496 cycles have a busiest rank below 12. Every 5–11-active-request sample retains at least one full 12-row rank. The 6-active subset has 52 cycles, all max-local=12, with 1–3 ranks owning zero active rows. These facts challenge a local-only row-removal design when slowest-rank rendezvous dominates.

**Conditional screening proxy only:** If one MoE stage's wall time were strictly proportional to the busiest rank's local row count and all other work stayed fixed, local-only row deletion could reduce the sum of those stage times by 516/(1,496×12) = 2.875%. This is not a kernel, Target, cycle or Product bound: expert selection, weight traffic, capacity padding, other resources and dependencies need measurement. Compaction after the already-required MoE AllGather has a different resource/communication profile and is **not** limited by this 2.875% proxy.

Source: `evidence/20260926_loop064_cp/run287/ownership/rank0_cohort*.jsonl`. Script: `scripts/loop072_imbalance.py`; output: `imbalance.json`.
