# 执行日志

- `2026-09-25T00:48:01Z` Loop 已冻结。下一步：Run131: parse Run107 bounded trace to map 96-token target quant-matmul device kernels to CPU operator calls and shapes; use source audit, no service restart.

- `2026-09-25T00:54:06Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run131`（profile）。

- `2026-09-25T00:54:57Z` Run `run131` 记录为 `pass`；正确性为 `not-applicable`。Run107 15 valid synchronized windows: exactly 236 quant-matmul and 43 GMM1 kernels/target. Layer-position bins match config ratios: first layers4/5, c4 layers6, c128 layers5, one tail kernel. Quant kernel sum median4.8223ms; c4 layer median120.28us vs c128 103.83us. Ordered association, no exact module/pointer proof.
