# 执行日志

- `2026-09-25T09:20:12Z` Loop 已冻结。下一步：Run189 offline align Run121 cycle64/65 per-expert count matrices with Run107/116 per-rank target/GMM/communication windows; quantify actual active-weight load imbalance and a safe static expert remapping screen

- `2026-09-25T09:23:29Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run189`（simulation）。

- `2026-09-25T09:23:29Z` Run `run189` 记录为 `pass`；正确性为 `not-applicable`。Run121 active expert sums nearly balanced by rank; impossible per-layer perfect balance removes161/175 active packed-weight reads, only2.026/2.202ms at representative1TB/s before costs. Separate Run107 profiled GMM rank-sum spread0.441/0.390ms; cross-run not causal. No remap benefit proven.

- `2026-09-25T09:27:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run190`（review）。

- `2026-09-25T09:27:46Z` Run `run190` 记录为 `pass`；正确性为 `not-applicable`。Static expert_map_path changes Ascend execution map but inspected checkpoint loader retains default physical slot mapping; nonidentity map unsafe without loader/reorder integration. No service/source change.
