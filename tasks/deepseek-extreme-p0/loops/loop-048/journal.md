# 执行日志

- `2026-09-25T07:51:07Z` Loop 已冻结。下一步：Run181 offline correlate Run165 prefill CPU trace and HostToDevice flows to source callsites for the 83-token forward; estimate path-specific removable time and select a narrowly instrumented legal 8-rank follow-up

- `2026-09-25T07:55:59Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run181`（profile）。

- `2026-09-25T07:55:59Z` Run `run181` 记录为 `pass`；正确性为 `not-applicable`。Run165 profiled 83-token rank0 prefill has 43 DSA scopes totaling187.371ms and 43 MoE scopes totaling207.733ms, nonoverlapping; direct-child-uncovered 80.091+82.923ms includes profiler/Python overhead, no removable-time claim

- `2026-09-25T07:57:43Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run182`（profile）。

- `2026-09-25T08:00:21Z` Run `run182` 记录为 `invalid`；正确性为 `invalid`。Temporary custom-op wrappers had varargs; PyTorch infer_schema rejected startup before model load or benchmark. Source restored, service stopped, eight NPUs idle.

- `2026-09-25T08:01:19Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run183`（profile）。

- `2026-09-25T08:19:09Z` Run `run183` 记录为 `pass`；正确性为 `pass`。Typed-op preflight and legal 48+12 Stock-only run passed; 8 ranks x10 prefill forwards, 43 DSA and43 MoE each; rank medians forward0.339-0.376s, DSA+MoE0.299-0.332s. Extreme flags absent so no Extreme conclusion; source restored/service stopped.

- `2026-09-25T08:19:36Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run184`（profile）。

- `2026-09-25T08:36:30Z` Run `run184` 记录为 `pass`；正确性为 `pass`。Legal Extreme 48+12 passed60/60, 40 rank/cohort records and Host mirror exact. Five prefill shapes88/264/368/152/328 tokens all ranks; per-call max forward walls sum1.901s; DSA+MoE inclusive CPU scopes cover87.7-88.0% of forward wall, not removable-time proof. Source restored/service stopped.

- `2026-09-25T08:40:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run185`（review）。

- `2026-09-25T08:40:10Z` Run `run185` 记录为 `pass`；正确性为 `not-applicable`。Independent review recommends one legal Extreme wall-versus-thread-CPU prefill discrimination with same-service control; actual model ID unverified; no removable-time or E2E claim
