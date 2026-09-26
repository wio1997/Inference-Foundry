# 执行日志

- `2026-09-26T09:23:56Z` Loop 已冻结。下一步：Inspect actual native CompressorMetadata/Compressor state and scatter writes at layer2; build private same-entry A/A and full96 versus owner16 fixture

- `2026-09-26T09:33:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run294`（test）。

- `2026-09-26T09:58:03Z` Run `run294` 记录为 `fail`；正确性为 `invalid`。8/8 private owner16 Compressor valid output slots exactly match full96; A/A/A3 stable; whole owner state page bytes mismatch on ranks0-6 (rank7 exact), so planned state gate fails and candidate semantics remain unproved. Eager bench12 and all8 Runtime gates pass; source restored; diagnostic TPS invalid for Product.

- `2026-09-26T10:00:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run295`（test）。

- `2026-09-26T10:19:28Z` Run `run295` 记录为 `fail`；正确性为 `invalid`。Runner exit1 on original whole-owner-page gate; attribution succeeds: 8/8 owner output exact; B prestate-changed bytes all match A (0 B-changed/A-different), A-only bytes explain every A/B mismatch; first difference per affected rank maps nonowner current write alias. Full live state/typed consumer/lifetime semantics remain unproved. Eager bench12 and all8 Runtime pass; sources restored.

- `2026-09-26T10:29:27Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run296`（test）。

- `2026-09-26T10:51:22Z` Run `run296` 记录为 `pass`；正确性为 `pass`。Private real layer2 owner16 state/scatter/native QLI gate 8/8 bit exact; eager carrier 12/12x1024 and Runtime 8/8 pass; no Graph/E2E gain claim

- `2026-09-26T10:55:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run297`（benchmark）。

- `2026-09-26T11:14:03Z` Run `run297` 记录为 `pass`；正确性为 `pass`。Private owner16/full96 typed-chain parity all8 and repeated topk pass; eager paired events show no stable update or through-QLI gain, performance REJECT pending Graph disambiguation

- `2026-09-26T11:18:30Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run298`（benchmark）。

- `2026-09-26T11:37:21Z` Run `run298` 记录为 `pass`；正确性为 `pass`。Private A96/B16/A2 Graph capture/replay all8 semantic gates pass; B local replay signal ~4–6us below 8.24us control drift, no Product gain; pivot whole DSA producer alias closure
