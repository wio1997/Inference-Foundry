# Run592 — historical production slice timing from existing Run246 trace

**Scoped offline PASS.** No service, NPU workload or new profiler session was run. A new reducer reads the last of five existing Run246 Level1 MemoryAccess profile sessions per rank. In each of eight ranks, it finds two Model45 physical stream/task sequences `1/40 MatMulV2 → 0/13 BF16 RS → 1/43 MEMCPY_ASYNC → 1/44 HcPost candidate`, with the intervening event record/wait tasks. All 16 neighborhoods pass chronology; copy start/stop agrees with the same session's `task_time.csv`. Thirty-two source files are SHA-pinned, and four identity/order mutations are rejected. Astra High independently recomputed all rows, medians and hashes, cross-checked all eight roles against `task_time.csv`, and checked profiler metadata/session ordering.

| Rank-local interval in Run246 Level1 trace | Min | Median | Max |
|---|---:|---:|---:|
| Partial MatMul start → SDMA copy end | 30.320 µs | 31.7695 µs | 33.441 µs |
| Partial MatMul end → SDMA copy end | 15.780 µs | 17.32975 µs | 18.440 µs |
| Partial end → RS start | 0.580 µs | 1.13025 µs | 1.640 µs |
| RS end → copy start | 0.6395 µs | 1.06025 µs | 1.9795 µs |
| Copy end → following HcPost start | 0.020 µs | 0.020 µs | 0.0405 µs |

These are **historical instrumented Current intervals** of a local native task neighborhood. The profile has no direct cohort/request ID. Its final session is inside each rank's final profile window (`first_cycle=64`, count2), and the five-session/five-cohort order supports an ordinal association with Run246's diagnostic cohort5, but it is not a direct fixed-`W₀` join. Run589's source-labeled Model45 sequence belongs to a different acquisition; matching numeric task IDs/order/shape is a hypothesis prior, not same-generation semantic proof. Run246 Level1 MemoryAccess profiling and sync perturb schedule; there is no matched unprofiled timing control. The Graph debug dump's synthetic `ts/dur` were never used. Microsecond timestamps are compared only within rank; no cross-card arrival calculation is made.

The tiny copy→HcPost gap rejects a large idle gap at this *profiled adjacency*. It does not identify communication transfer versus peer wait inside RS, establish actual current unmarked duration, or make any portion removable. Do not multiply one slice by layer/cycle count, add it to other profiler spans, or infer a Product gain. Fixed acceptance/output/work and formal Run99 Current **571.681 tok/s** remain unchanged. Strict Resource/Hardware, Scheduling/Execution and Product E2E endpoints and numerical Current→credible-limit gap remain null.

Decision: reuse this local timing evidence as a conditional Current DAG cost and **defer a duplicate live Level0 slice acquisition** until current same-generation identity or lower perturbation would change a Bound or architecture decision. The next higher-value Bound work is a broad fixed-`W₀` logical-work/traffic witness and a full Target/Draft/KV/Host all8 dependency/overlap account, while exact-board compute/HBM/HCCS `C⁺/B` remains separate.

Evidence: `summary.json`, `astra_final_review.md`, reducer `scripts/loop081_historical_slice_run592.py`, original Run246 trace and `task_time.csv` under the 32 SHA-pinned paths in summary. This is not a fresh E2E benchmark.
