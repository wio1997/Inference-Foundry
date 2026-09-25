# 执行日志

- `2026-09-25T09:46:03Z` Loop 已冻结。下一步：Run198 source and existing Run186/184 Host trace audit: quantify per-layer exclusive DSA/MoE Python/native dispatch call families and identify a repeatable specialization no wider than one layer segment.

- `2026-09-25T09:48:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run198`（review）。

- `2026-09-25T09:48:44Z` Run `run198` 记录为 `pass`；正确性为 `not-applicable`。Run181 profiler Event record/wait totals37.46ms per83-token forward but source uses them to overlap shared/routed MoE; no direct deletion. Run186 low-overhead CPU≈wall. Next inspect saved device overlap before same-stream A/B.

- `2026-09-25T09:50:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run199`（review）。

- `2026-09-25T09:50:00Z` Run `run199` 记录为 `pass`；正确性为 `not-applicable`。Run165 rank0 83-token profiled prefill: shared-expert stream36 172 kernels sum2.1ms in437ms DSA/MoE Host envelope; no observed overlap with main/comm kernel intervals. Profiler caveat; supports bounded prefill-only same-stream A/B, not gain.
