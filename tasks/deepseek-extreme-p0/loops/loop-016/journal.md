# 执行日志

- `2026-09-20T21:29:38Z` Loop 已冻结。下一步：Trace generation of DSA compressor slot mapping through scheduler and metadata, then decide legal fast-path guard and prototype isolated kernel.

- `2026-09-20T21:33:27Z` 为用例 `cold_32k_128_c1` 创建 Run `builtin-scatter-screen-20260920`（benchmark）。

- `2026-09-20T21:33:27Z` Run `builtin-scatter-screen-20260920` 记录为 `pass`；正确性为 `pass`。Captured-rank0 8096x2 unique indices, real cache/update shape: torch_npu.npu_scatter_nd_update_ output bit-equal; 25-call device median1.402ms vs SK1.429ms (1.9% faster), below practical E2E target and different loaded-service conditions. Screen only.

- `2026-09-20T21:43:37Z` 为用例 `cold_32k_128_c1` 创建 Run `index-copy-screen-20260920`（benchmark）。

- `2026-09-20T21:43:37Z` Run `index-copy-screen-20260920` 记录为 `pass`；正确性为 `pass`。Real captured indices and contiguous matching cache: index_copy_ bit-equal to SK, device median0.510ms vs SK1.428ms over25 calls. Flattened indices precomputed; screen only.

- `2026-09-20T21:43:50Z` 为用例 `cold_32k_128_c1` 创建 Run `strided-index-copy-screen-20260920`（benchmark）。

- `2026-09-20T21:43:50Z` Run `strided-index-copy-screen-20260920` 记录为 `pass`；正确性为 `pass`。Simulated first-axis stride32768 with real captured indices: as_strided physical-row index_copy_ bit-equal to SK, including per-call index flatten device median0.694ms vs SK1.528ms over20 calls. Isolated screen, not real cache layout or E2E.

- `2026-09-20T21:43:50Z` 为用例 `cold_32k_128_c1` 创建 Run `swa-prefill-index-copy-ab-20260920`（benchmark）。
