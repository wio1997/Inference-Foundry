# 执行日志

- `2026-09-25T09:20:12Z` Loop 已冻结。下一步：Run189 offline align Run121 cycle64/65 per-expert count matrices with Run107/116 per-rank target/GMM/communication windows; quantify actual active-weight load imbalance and a safe static expert remapping screen

- `2026-09-25T09:23:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run189`（simulation）。

- `2026-09-25T09:23:29Z` Run `run189` 记录为 `pass`；正确性为 `not-applicable`。Run121 active expert sums nearly balanced by rank; impossible per-layer perfect balance removes161/175 active packed-weight reads, only2.026/2.202ms at representative1TB/s before costs. Separate Run107 profiled GMM rank-sum spread0.441/0.390ms; cross-run not causal. No remap benefit proven.

- `2026-09-25T09:27:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run190`（review）。

- `2026-09-25T09:27:46Z` Run `run190` 记录为 `pass`；正确性为 `not-applicable`。Static expert_map_path changes Ascend execution map but inspected checkpoint loader retains default physical slot mapping; nonidentity map unsafe without loader/reorder integration. No service/source change.

- `2026-09-25T09:29:42Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run191`（simulation）。

- `2026-09-25T09:29:42Z` Run `run191` 记录为 `pass`；正确性为 `not-applicable`。Fixed 32-expert/rank map trained on Run121 cycle64 gains57 active reads in train but only3 on held-out65; reverse gains78 in train and loses1 on held-out64. Static placement has no robust demonstrated balance gain; no service was started.

- `2026-09-25T09:29:43Z` 主控结论为 `PIVOTED`。Run190 finds static expert_map execution/weight-loader mismatch needing substantial correctness integration. Run191 fixed-map pair-swap simulation over two captured cycles gains57/78 active reads in sample but only+3/-1 on the other cycle; no robust transferable reduction. Run189 ideal2.0-2.2ms/cycle is unattainable arithmetic, and separate-service GMM rank spread<0.45ms. Deprioritize placement; no code or formal E2E.
