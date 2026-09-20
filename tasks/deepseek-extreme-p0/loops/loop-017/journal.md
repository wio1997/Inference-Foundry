# 执行日志

- `2026-09-20T22:42:42Z` Loop 已冻结。下一步：Inspect existing msprof timeline and Compressor ratio4 callsite; design a prefill-only critical-path measurement before any optimization.

- `2026-09-20T22:45:57Z` 为用例 `cold_32k_128_c1` 创建 Run `compressor-cold-clusters-20260920`（profile）。

- `2026-09-20T22:46:29Z` Run `compressor-cold-clusters-20260920` 记录为 `pass`；正确性为 `not-applicable`。Existing msprof shape[8096,4096] ratio4 Compressor 336 tasks partition into four cold request clusters of84 calls each, about126.4ms summed task duration per request and ~1.503ms median/call. This is device occupancy, not proven TTFT critical-path saving; next attribute dependencies and operator internals.

- `2026-09-20T22:48:20Z` 为用例 `cold_32k_128_c1` 创建 Run `compressor-stream-order-20260920`（profile）。

- `2026-09-20T22:48:20Z` Run `compressor-stream-order-20260920` 记录为 `pass`；正确性为 `not-applicable`。All336 main ratio4 Compressor tasks on frozen original-source profile are immediately followed on same device stream by ScatterNdUpdateSk then SparseAttnSharedkv; median next-task gap2.87us and Compressor-to-attention start1.845ms. This establishes serial local dependency, not global E2E achievable savings.
