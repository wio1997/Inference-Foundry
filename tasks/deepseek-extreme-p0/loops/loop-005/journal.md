# 执行日志

- `2026-09-20T16:39:46Z` Loop 已冻结。下一步：Make DSA CP env-toggle default true, launch both false, run full correctness/warmup/benchmark script

- `2026-09-20T16:41:22Z` 为用例 `mixed_32k_1024_c12` 创建 Run `sp-dsacp-off-20260920`（benchmark）。

- `2026-09-20T16:55:31Z` Run `sp-dsacp-off-20260920` 记录为 `invalid`；正确性为 `invalid`。Valid service loaded, but frozen golden4 exact-text gate failed 4/4; same candidate repeated golden4 also differs 4/4, proving exact reasoning text is nondeterministic here. No performance benchmark ran.

- `2026-09-20T16:55:31Z` 主控结论为 `PIVOTED`。Strict golden4 exact-output gate is invalid: 4/4 mismatch even between repeat runs on unchanged candidate; cannot infer candidate correctness failure or performance
