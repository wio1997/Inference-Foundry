# 执行日志

- `2026-09-25T01:28:26Z` Loop 已冻结。下一步：Run136: reconstruct per-rank absolute prepare/target/proposer event timing and first-collective intervals from Run106/107 and Run98; distinguish fixed offset from cycle growth and identify the latest rank.

- `2026-09-25T01:28:45Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run136`（profile）。

- `2026-09-25T01:29:41Z` Run `run136` 记录为 `pass`；正确性为 `not-applicable`。Run106 two profiled cycles show prepare and first collective arrival skew ~9.6-10.5 ms, collective end aligned <=0.012 ms, with latest rank changing between cycles. Run98 steady cycles64-255 rank duration spreads are tiny (target median 0.0812 ms, proposer 0.0603 ms) but lack shared absolute timestamps, so steady phase offset and removable critical path remain unidentified.

- `2026-09-25T01:31:46Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run137`（profile）。

- `2026-09-25T01:50:50Z` Run `run137` 记录为 `pass`；正确性为 `not-applicable`。Legal 12/12 exact1024 TP8 diagnostic. Each rank produced 294 cycles, 192 steady cycles64-255 analyzed. Host begin skew median3.944ms, post-target skew0.991ms, post-proposer skew3.928ms; cohort host wall median58.130ms. Target graph catches up rank phase; proposer reintroduces skew. Timestamps are host asynchronous dispatch markers, not device collective interval or removable wall. Borrowed sources restored to original SHA and service stopped.
