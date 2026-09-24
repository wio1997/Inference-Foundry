# 执行日志

- `2026-09-24T15:17:37Z` Loop 已冻结。下一步：Read borrowed target grouped-matmul implementation and trace 86 calls to exact layer, shape, dtype and backend; rank candidate substitutions by device bound.

- `2026-09-24T15:20:48Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run108`（design-check）。

- `2026-09-24T15:21:10Z` Run `run108` 记录为 `pass`；正确性为 `not-applicable`。Read-only source/trace audit passed: DeepSeek V4 Flash config has 43 hidden layers; all eight rank cycle64 target windows have exactly 43 GMM1 SwigluQuantWeightNzV2 and 43 GMM2 WeightNz kernels. Rank0 sums 6.425 and 3.646 ms, respectively. Source maps GMM1 to DeviceOperator.npu_grouped_matmul_swiglu_quant and GMM2 to npu_grouped_matmul_gmm2 through moe_mlp; actual tensor shapes and replacement parity are not yet measured.

- `2026-09-24T15:29:11Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run109`（design-check）。

- `2026-09-24T15:29:43Z` Run `run109` 记录为 `pass`；正确性为 `not-applicable`。Corrected Run108 GMM1 source attribution: frozen quant group_size=0 makes W4A8 weights per-channel; trace GMM1 uses moe_mlp.py custom grouped_matmul_swiglu_quant_v2 branch, while GMM2 uses DeviceOperator.npu_grouped_matmul_gmm2. Checkpoint packed I8 layer0 expert0 shapes are w1/w3 [1024,4096] and w2 [2048,2048]. Run108 43+43 kernel counts remain valid; live routed input shapes and faster variant remain unmeasured.

- `2026-09-24T16:10:17Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run110`（review）。

- `2026-09-24T16:15:25Z` Run `run110` 记录为 `invalid`；正确性为 `invalid`。Bounded Zcode environment check exited 0 after 188.845s and 27 provider requests, but headless build mode denied docker exec, npu-smi and HTTP access (No permission client configured for Bash). Configured model deepseek/deepseek-flash; actual model was not independently observed in output, so model_verified=false. HEAD 1b7c0e0 and 8 profile windows matched direct Sol recheck; Sol directly verified all eight NPUs idle and service stopped. Delegated result is partial and excluded from product evidence.

- `2026-09-24T16:56:20Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run111`（review）。

- `2026-09-24T16:57:50Z` Run `run111` 记录为 `pass`；正确性为 `not-applicable`。Minimal Zcode plan-mode ping succeeded: exit 0, 14.735 s, one provider request, exact sentinel response. Independent local model I/O record identifies request provider deepseek/model deepseek-flash and response modelId deepseek-flash. This validates short text connectivity only; Run110 tool permissions remain blocked and this is not product performance evidence.
