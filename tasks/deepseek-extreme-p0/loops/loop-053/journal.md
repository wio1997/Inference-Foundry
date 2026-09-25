# 执行日志

- `2026-09-25T10:13:50Z` Loop 已冻结。下一步：Run203 offline map Run181 profiled direct-child calls against Run186 low-overhead prefill layer costs and borrowed source; identify one precise repeated segment and quantify only an optimistic call-overhead screen.

- `2026-09-25T10:16:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run203`（review）。

- `2026-09-25T10:16:38Z` Run `run203` 记录为 `pass`；正确性为 `not-applicable`。Run186 legal 8-rank9-call prefill has43 DSA+43 MoE layers/call; c4 DSA median4.558ms/layer, c1283.105ms, MoE~3.52ms. Run181 direct-child-uncovered DSA80.1/MoE82.9ms is profiler-perturbed and not removable. No bounded single call-family gain established; next screen shape reuse for exact-state prefill replay.

- `2026-09-25T10:18:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run204`（review）。

- `2026-09-25T10:18:47Z` Run `run204` 记录为 `pass`；正确性为 `not-applicable`。Seven legal saved cohorts have53 eager prefill calls and20 unique (actual_tokens,num_reqs) signatures; 45/53 fall in repeated signatures. This is necessary shape reuse only, not exact-state replay feasibility; DSA-CP capture builder rejects prefill. Next inventory mutable input/write-set ABI.

- `2026-09-25T10:20:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run205`（review）。

- `2026-09-25T10:20:09Z` Run `run205` 记录为 `pass`；正确性为 `not-applicable`。Stock graph builder rejects prefill; repeated (tokens,reqs) signatures omit dynamic query/slot/block/CP metadata and KV write state. Next collect two real88-token call fingerprints across8 ranks before capture design.

- `2026-09-25T10:22:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run206`（profile）。

- `2026-09-25T10:23:52Z` Run `run206` 记录为 `invalid`；正确性为 `invalid`。Preflight direct import model_runner_v1 hit borrowed circular import DeviceOperator from partially initialized device_op before service startup; zero requests, no metadata. Patch exact restored SHA004dbd0, no service. Retry with validated import order as Run207.
