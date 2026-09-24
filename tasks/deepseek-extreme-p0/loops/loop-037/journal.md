# 执行日志

- `2026-09-24T13:13:41Z` Loop 已冻结。下一步：Build a bounded, legal one-cohort Runtime-only target profile using PROFILER_DIR and the existing scope markers; verify parser mapping before any code optimization.

- `2026-09-24T13:15:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run100`（profile）。

- `2026-09-24T13:36:53Z` Run `run100` 记录为 `invalid`；正确性为 `invalid`。Legal warmup and sampled 12x1024 cohorts succeeded, but start_profile RPC completed after benchmark end; offline parsed 8-rank traces have zero target scopes and zero kernel rows. No target attribution or performance inference. Run101 must activate profiler before launching cohort.

- `2026-09-24T13:37:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run101`（profile）。

- `2026-09-24T13:41:38Z` Run `run101` 记录为 `invalid`；正确性为 `pass`。Legal 12x1024 cohort and Runtime path succeeded; profiler activated before benchmark, but 8-second trace produced approximately 11 GiB raw across 8 ranks and offline parse warned >30 min per rank. Analysis terminated during rank0; no trustworthy target kernel attribution. Next use subsecond profile window during 48-request decode.
