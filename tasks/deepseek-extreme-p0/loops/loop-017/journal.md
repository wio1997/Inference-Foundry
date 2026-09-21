# 执行日志

- `2026-09-20T22:42:42Z` Loop 已冻结。下一步：Inspect existing msprof timeline and Compressor ratio4 callsite; design a prefill-only critical-path measurement before any optimization.

- `2026-09-20T22:45:57Z` 为用例 `cold_32k_128_c1` 创建 Run `compressor-cold-clusters-20260920`（profile）。

- `2026-09-20T22:46:29Z` Run `compressor-cold-clusters-20260920` 记录为 `pass`；正确性为 `not-applicable`。Existing msprof shape[8096,4096] ratio4 Compressor 336 tasks partition into four cold request clusters of84 calls each, about126.4ms summed task duration per request and ~1.503ms median/call. This is device occupancy, not proven TTFT critical-path saving; next attribute dependencies and operator internals.

- `2026-09-20T22:48:20Z` 为用例 `cold_32k_128_c1` 创建 Run `compressor-stream-order-20260920`（profile）。

- `2026-09-20T22:48:20Z` Run `compressor-stream-order-20260920` 记录为 `pass`；正确性为 `not-applicable`。All336 main ratio4 Compressor tasks on frozen original-source profile are immediately followed on same device stream by ScatterNdUpdateSk then SparseAttnSharedkv; median next-task gap2.87us and Compressor-to-attention start1.845ms. This establishes serial local dependency, not global E2E achievable savings.

- `2026-09-20T23:55:53Z` 为用例 `cold_32k_128_c1` 创建 Run `compressor-shape-operator-20260920`（benchmark）。

- `2026-09-20T23:57:59Z` Run `compressor-shape-operator-20260920` 记录为 `pass`；正确性为 `pass`。One-NPU synthetic inputs matching frozen cold ratio4 Compressor shapes and configuration produce finite output [2025,512]; 25 timed calls device median1.630ms, minimum1.499ms. Isolated screen is not E2E or exact model-input parity. Initial oversized random activation gave nonfinite output and was discarded; corrected scaled input/weights is benchmark.

- `2026-09-21T00:03:10Z` 为用例 `cold_32k_128_c1` 创建 Run `compressor-mbase256-isolated-build-20260920`（build）。
