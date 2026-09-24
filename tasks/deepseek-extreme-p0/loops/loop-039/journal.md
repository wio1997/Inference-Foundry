# 执行日志

- `2026-09-24T15:17:37Z` Loop 已冻结。下一步：Read borrowed target grouped-matmul implementation and trace 86 calls to exact layer, shape, dtype and backend; rank candidate substitutions by device bound.

- `2026-09-24T15:20:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run108`（design-check）。

- `2026-09-24T15:21:10Z` Run `run108` 记录为 `pass`；正确性为 `not-applicable`。Read-only source/trace audit passed: DeepSeek V4 Flash config has 43 hidden layers; all eight rank cycle64 target windows have exactly 43 GMM1 SwigluQuantWeightNzV2 and 43 GMM2 WeightNz kernels. Rank0 sums 6.425 and 3.646 ms, respectively. Source maps GMM1 to DeviceOperator.npu_grouped_matmul_swiglu_quant and GMM2 to npu_grouped_matmul_gmm2 through moe_mlp; actual tensor shapes and replacement parity are not yet measured.
