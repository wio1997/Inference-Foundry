# 执行日志

- `2026-09-24T14:23:03Z` Loop 已冻结。下一步：Implement opt-in cycle-scheduled profiling in ExtremeDecodeRuntime and launch one legal cohort.

- `2026-09-24T14:26:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run105`（profile）。

- `2026-09-24T14:27:47Z` Run `run105` 记录为 `invalid`；正确性为 `invalid`。Launch failed before service start: script lacked executable mode (bash: Permission denied, exit 126). No workload or profiler ran.

- `2026-09-24T14:28:22Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run106`（profile）。

- `2026-09-24T14:53:38Z` Run `run106` 记录为 `pass`；正确性为 `pass`。Legal 12x1024 c12 cohort completed 12/12 exact lengths and 8/8 Runtime gates at 289 cycles/rank. Cycle 64-65 profiler captured two target scopes on all eight ranks, 5.6 MB raw/rank, parsed successfully. Profiler target device-total median 51.144 ms is dominated by nested wait_event median 50.134 ms, while nested HCCL all-gather is 0.247 ms; asynchronous graph work makes CPU scope timestamp clipping invalid. Whole-trace grouped matmul kernel sum median 20.592 ms across two cycles includes target and proposer, so target-only optimization bound is not established. Diagnostic TPS 508.89 includes profiler overhead and is not comparable to formal E2E.

- `2026-09-24T14:55:22Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run107`（profile）。

- `2026-09-24T15:17:36Z` Run `run107` 记录为 `pass`；正确性为 `pass`。Legal 12x1024 c12 cohort passed 12/12 exact lengths and 8/8 Runtime gates at 301 cycles/rank. Two sampled target cycles per rank parsed; all eight cycle64 windows share 2836 target kernels. Rank1 cycle65 has 2979 kernels and is excluded. Across 15 canonical synchronized windows, target device interval union median 50.281 ms, compute union 39.932 ms, communication union 11.547 ms with 1.292 ms overlap; 86 grouped-matmul kernels sum 9.966 ms/cycle. Synchronization/profiler perturb timing; 494.03 tok/s is diagnostic only. This identifies a target grouped-matmul candidate, not an achieved optimization.

- `2026-09-24T15:17:36Z` 暂存知识变化 `loop038-target-graph-compute-profile`：In a synchronized two-cycle legal c12 TP8 target trace, 15 canonical target windows across all eight ranks contain 2836 kernels each; target device union median is 50.281 ms, compute union 39.932 ms, communication union 11.547 ms, overlap 1.292 ms, and 86 grouped-matmul kernels sum 9.966 ms per cycle. These are diagnostic upper bounds under profiler/synchronization overhead, not formal E2E gains.

- `2026-09-24T15:17:37Z` 主控结论为 `PIVOTED`。Cycle-scheduled profiling met the technical attribution goal in Run107: legal 8-rank cohort and 15 canonical synchronized target windows isolate 2836 kernels/cycle and quantify compute, communication, overlap, and a 9.966 ms grouped-matmul family. Run105 invalid launcher remains preserved, so the mixed-history Loop is pivoted; forced synchronization makes performance diagnostic only.
