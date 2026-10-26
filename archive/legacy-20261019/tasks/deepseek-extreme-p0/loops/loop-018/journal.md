# 执行日志

- `2026-09-21T01:09:46Z` Loop 已冻结。下一步：Inspect saved profiler trace schema and align collective/compute intervals across all eight ranks without treating Idle Time as network latency.

- `2026-09-21T01:13:01Z` 为用例 `mixed_32k_1024_c12` 创建 Run `saved-c12-collective-clock-audit-20260921`（profile）。

- `2026-09-21T01:13:01Z` Run `saved-c12-collective-clock-audit-20260921` 记录为 `pass`；正确性为 `not-applicable`。Aligned 48,960 identical TP8 collective types; raw start spread median0.6105ms and end spread0.583ms, rank6 earliest in48,829 calls. Median-end calibration estimates rank6 offset -579.172us, reducing start spread median to0.2175ms but leaving end spread0.206ms; no independently synchronized clocks or E2E causal attribution. No communication patch justified.

- `2026-09-21T01:13:02Z` 主控结论为 `PIVOTED`。Saved cross-rank timestamps are confounded by stable ~579us rank6 clock offset plus residual ~206us end spread. Communication elapsed is Idle-only, so collective arrival skew cannot be separated from clock/reporting artifacts or translated to a safe high-value change using current evidence.
