# 执行日志

- `2026-09-25T01:28:26Z` Loop 已冻结。下一步：Run136: reconstruct per-rank absolute prepare/target/proposer event timing and first-collective intervals from Run106/107 and Run98; distinguish fixed offset from cycle growth and identify the latest rank.

- `2026-09-25T01:28:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run136`（profile）。

- `2026-09-25T01:29:41Z` Run `run136` 记录为 `pass`；正确性为 `not-applicable`。Run106 two profiled cycles show prepare and first collective arrival skew ~9.6-10.5 ms, collective end aligned <=0.012 ms, with latest rank changing between cycles. Run98 steady cycles64-255 rank duration spreads are tiny (target median 0.0812 ms, proposer 0.0603 ms) but lack shared absolute timestamps, so steady phase offset and removable critical path remain unidentified.
