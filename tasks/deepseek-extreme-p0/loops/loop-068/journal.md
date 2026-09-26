# 执行日志

- `2026-09-26T13:00:32Z` Loop 已冻结。下一步：Build fail-closed private one-bank full producer fixture from original layer2 _forward prestate; first static ABI/source preflight, then all8 A/A/B/A value run

- `2026-09-26T13:00:32Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run304`（design-check）。

- `2026-09-26T13:04:06Z` Run `run304` 记录为 `pass`；正确性为 `not-applicable`。Source SHA guards and hook anchors pass, static ABI and private one-bank fixture compiled; no device correctness claim

- `2026-09-26T13:04:43Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run305`（test）。

- `2026-09-26T13:22:01Z` Run `run305` 记录为 `pass`；正确性为 `pass`。8/8 real layer2 private one-bank same-prestate A/A/B/A WKV/SWA, main Compressor, indexer/QLI and Sparse exact on owner writes; client12/12 exact1024 and Runtime8/8 pass; no Product timing

- `2026-09-26T13:22:01Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run306`（design-check）。

- `2026-09-26T13:23:57Z` Run `run306` 记录为 `pass`；正确性为 `not-applicable`。Source SHA, Graph capture script syntax and single-bank A/B private replay preflight pass; actual Graph yet untested

- `2026-09-26T13:23:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run307`（benchmark）。

- `2026-09-26T13:43:13Z` Run `run307` 记录为 `pass`；正确性为 `pass`。Private Graph all8 rank Sparse/QLI exact, 12/12x1024 Runtime8 pass, 77/80 strict paired B faster; post-replay persistent cache values not checked in this run

- `2026-09-26T13:43:13Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run308`（test）。

- `2026-09-26T14:02:08Z` Run `run308` 记录为 `pass`；正确性为 `pass`。8/8 private Graph 39 replays/rank Sparse/QLI and owner SWA/main/indexer state/key/scale exact to eager A, 80/80 paired B faster, 12/12x1024 Runtime8 pass; source restored

- `2026-09-26T14:02:08Z` 暂存知识变化 `k_loop068_full_producer_owner_private`：Single real layer2 c4 private owner16 full WKV/SWA plus indexer/main Compressor through QLI and Sparse is same-prestate value equivalent to full96 across eight ranks and reproducibly shortens isolated Graph producer-to-Sparse duration by about 24-32us at the max-rank screen; Product effect unproved

- `2026-09-26T14:02:08Z` 主控结论为 `PIVOTED`。Private value and Graph cost gates pass; persistent live Target semantics and exposed multi-rank cycle effect require reversible one-layer integration
