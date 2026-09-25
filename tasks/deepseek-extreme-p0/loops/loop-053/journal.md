# 执行日志

- `2026-09-25T10:13:50Z` Loop 已冻结。下一步：Run203 offline map Run181 profiled direct-child calls against Run186 low-overhead prefill layer costs and borrowed source; identify one precise repeated segment and quantify only an optimistic call-overhead screen.

- `2026-09-25T10:16:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run203`（review）。

- `2026-09-25T10:16:38Z` Run `run203` 记录为 `pass`；正确性为 `not-applicable`。Run186 legal 8-rank9-call prefill has43 DSA+43 MoE layers/call; c4 DSA median4.558ms/layer, c1283.105ms, MoE~3.52ms. Run181 direct-child-uncovered DSA80.1/MoE82.9ms is profiler-perturbed and not removable. No bounded single call-family gain established; next screen shape reuse for exact-state prefill replay.

- `2026-09-25T10:18:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run204`（review）。

- `2026-09-25T10:18:47Z` Run `run204` 记录为 `pass`；正确性为 `not-applicable`。Seven legal saved cohorts have53 eager prefill calls and20 unique (actual_tokens,num_reqs) signatures; 45/53 fall in repeated signatures. This is necessary shape reuse only, not exact-state replay feasibility; DSA-CP capture builder rejects prefill. Next inventory mutable input/write-set ABI.
