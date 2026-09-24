# 执行日志

- `2026-09-24T10:53:45Z` Loop 已冻结。下一步：Source-audit max_seqlen_k consumers and derive a safe per-cohort bound; first add an opt-in exact metadata shadow against the existing dynamic path in the dedicated Runtime, then measure matched c12 stage timing.

- `2026-09-24T10:58:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run94`（test）。

- `2026-09-24T11:15:36Z` Run `run94` 记录为 `invalid`；正确性为 `invalid`。Invalid 2048-token diagnostic control: Extreme serving guard requires max_tokens=1024; no metadata shadow checks executed; no semantic/performance conclusion.

- `2026-09-24T11:40:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run95`（test）。

- `2026-09-24T11:40:48Z` Run `run95` 记录为 `invalid`；正确性为 `invalid`。Host-side invocation failed before service launch because /usr/local/Ascend/ascend-toolkit/set_env.sh exists only inside the designated container; no metadata shadow or benchmark ran.
