# 执行日志

- `2026-09-25T11:12:33Z` Loop 已冻结。下一步：Run211 read-only source closure: actual custom-op call, context, mutable state and graph API contract; choose a minimal one-layer capture implementation or reject

- `2026-09-25T11:12:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run211`（design-check）。

- `2026-09-25T11:13:35Z` Run `run211` 记录为 `pass`；正确性为 `not-applicable`。MoE custom-op source closure excludes direct DSA/KV but has forward-context index, DP/SP sizes, routing, multistream/HCCL and possible mutable counters; live ABI/state probe required before capture

- `2026-09-25T11:17:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run212`（profile）。

- `2026-09-25T11:20:16Z` Run `run212` 记录为 `invalid`；正确性为 `invalid`。Stopped before requests after detecting probe BF16-to-NumPy hash incompatibility; dry import passed but no ABI data, no benchmark; exact source restore and service stop verified

- `2026-09-25T11:21:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run213`（profile）。

- `2026-09-25T11:41:10Z` Run `run213` 记录为 `pass`；正确性为 `pass`。72/72 legal requests and8-rank Runtime pass; first88 MoE layer0 per-rank [11,4096] BF16 stable alias/address/context with changing inputs and outputs; no graph/performance claim

- `2026-09-25T11:44:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run214`（design-check）。

- `2026-09-25T11:44:44Z` Run `run214` 记录为 `pass`；正确性为 `not-applicable`。Bound one layer/shape shadow graph capture with two distinct real inputs, eager self-noise controls, retained serving eager output and all8-rank collective gates; no performance claim
