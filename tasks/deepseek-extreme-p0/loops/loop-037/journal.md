# 执行日志

- `2026-09-24T13:13:41Z` Loop 已冻结。下一步：Build a bounded, legal one-cohort Runtime-only target profile using PROFILER_DIR and the existing scope markers; verify parser mapping before any code optimization.

- `2026-09-24T13:15:02Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run100`（profile）。

- `2026-09-24T13:36:53Z` Run `run100` 记录为 `invalid`；正确性为 `invalid`。Legal warmup and sampled 12x1024 cohorts succeeded, but start_profile RPC completed after benchmark end; offline parsed 8-rank traces have zero target scopes and zero kernel rows. No target attribution or performance inference. Run101 must activate profiler before launching cohort.

- `2026-09-24T13:37:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run101`（profile）。

- `2026-09-24T13:41:38Z` Run `run101` 记录为 `invalid`；正确性为 `pass`。Legal 12x1024 cohort and Runtime path succeeded; profiler activated before benchmark, but 8-second trace produced approximately 11 GiB raw across 8 ranks and offline parse warned >30 min per rank. Analysis terminated during rank0; no trustworthy target kernel attribution. Next use subsecond profile window during 48-request decode.

- `2026-09-24T13:42:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run102`（profile）。

- `2026-09-24T13:48:04Z` Run `run102` 记录为 `invalid`；正确性为 `pass`。Legal 48x1024 profiled cohort completed 48/48 and parsed 8-rank kernel traces, but every rank has zero Extreme cycle and target scopes. Short window hit cohort boundary/non-target stage; no target attribution. Run103 will use one c12 cohort and activate profiler immediately at request launch.

- `2026-09-24T13:48:34Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run103`（profile）。

- `2026-09-24T13:52:33Z` Run `run103` 记录为 `invalid`；正确性为 `pass`。Legal c12 12x1024 cohort completed 12/12, but profiler start returned immediately and 0.5-second trace stayed in prefill. Eight parsed ranks each have zero Extreme cycle/target scopes despite kernels; no target attribution. Next request profiling 3 seconds after cohort launch.
