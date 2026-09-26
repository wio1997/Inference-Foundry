# 执行日志

- `2026-09-26T09:23:56Z` Loop 已冻结。下一步：Inspect actual native CompressorMetadata/Compressor state and scatter writes at layer2; build private same-entry A/A and full96 versus owner16 fixture

- `2026-09-26T09:33:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run294`（test）。

- `2026-09-26T09:58:03Z` Run `run294` 记录为 `fail`；正确性为 `invalid`。8/8 private owner16 Compressor valid output slots exactly match full96; A/A/A3 stable; whole owner state page bytes mismatch on ranks0-6 (rank7 exact), so planned state gate fails and candidate semantics remain unproved. Eager bench12 and all8 Runtime gates pass; source restored; diagnostic TPS invalid for Product.

- `2026-09-26T10:00:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run295`（test）。

- `2026-09-26T10:19:28Z` Run `run295` 记录为 `fail`；正确性为 `invalid`。Runner exit1 on original whole-owner-page gate; attribution succeeds: 8/8 owner output exact; B prestate-changed bytes all match A (0 B-changed/A-different), A-only bytes explain every A/B mismatch; first difference per affected rank maps nonowner current write alias. Full live state/typed consumer/lifetime semantics remain unproved. Eager bench12 and all8 Runtime pass; sources restored.
