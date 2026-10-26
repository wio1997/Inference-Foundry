# 执行日志

- `2026-09-25T09:46:03Z` Loop 已冻结。下一步：Run198 source and existing Run186/184 Host trace audit: quantify per-layer exclusive DSA/MoE Python/native dispatch call families and identify a repeatable specialization no wider than one layer segment.

- `2026-09-25T09:48:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run198`（review）。

- `2026-09-25T09:48:44Z` Run `run198` 记录为 `pass`；正确性为 `not-applicable`。Run181 profiler Event record/wait totals37.46ms per83-token forward but source uses them to overlap shared/routed MoE; no direct deletion. Run186 low-overhead CPU≈wall. Next inspect saved device overlap before same-stream A/B.

- `2026-09-25T09:50:00Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run199`（review）。

- `2026-09-25T09:50:00Z` Run `run199` 记录为 `pass`；正确性为 `not-applicable`。Run165 rank0 83-token profiled prefill: shared-expert stream36 172 kernels sum2.1ms in437ms DSA/MoE Host envelope; no observed overlap with main/comm kernel intervals. Profiler caveat; supports bounded prefill-only same-stream A/B, not gain.

- `2026-09-25T09:53:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run200`（benchmark）。

- `2026-09-25T10:11:08Z` Run `run200` 记录为 `pass`；正确性为 `invalid`。Legal warmup48+A12/B12/A2 12 all84 pass; 8/8 B prefill-only candidate marks; B/A2 identical8-call shapes but B max-rank forward sum3.0750s vs A2 3.0566s (slower18.4ms). Output hashes unstable even A/A2 (0/12), so exact parity not established. Runtime 8-rank pass; candidate rejected, no formal E2E. Sources restored, service stopped.

- `2026-09-25T10:11:44Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run201`（benchmark）。

- `2026-09-25T10:12:26Z` Run `run201` 记录为 `invalid`；正确性为 `invalid`。One-card event loop executed but evidence write failed because docker exec cwd was not repository; no analyzable result. Process exited, no service running. Repeat with explicit -w as Run202.

- `2026-09-25T10:12:42Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run202`（benchmark）。

- `2026-09-25T10:13:33Z` Run `run202` 记录为 `pass`；正确性为 `not-applicable`。Single-card 12x1000 same-stream eager microbench: record+wait median15.696us/pair thread CPU, empty0.043us; illustrative 2 pairs*43 layers=1.346ms/forward gross, about12ms/9-forward cohort before critical-path effects. No service/source change; rejects deeper event rewrite priority.

- `2026-09-25T10:13:33Z` 主控结论为 `PIVOTED`。Run200 legal same-service prefill-only same-stream candidate activated8/8 but B shape-matched forward wall is18.4ms slower than A2; client changes confounded by decode cycles and response hashes unstable even between controls. Run202 one-card same-stream event pair Host cost15.65us, making even2 redundant pairs per43 layers ~1.35ms/forward gross, ~12ms/9-call cohort. Event/stream simplification has no material observed product value. Remaining prefill active CPU is broader operator submission; pursue bounded fixed-model native path only if its removable fraction can be demonstrated.
