# 执行日志

- `2026-09-25T01:20:35Z` Loop 已冻结。下一步：Run134: offline Run107 device kernel and source audit of HcPre, RMSNorm, clone/copy and cache scatter inside valid target windows; no service restart.

- `2026-09-25T01:23:09Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run134`（profile）。

- `2026-09-25T01:25:28Z` Run `run134` 记录为 `pass`；正确性为 `not-applicable`。Offline 15 valid synchronized Run107 target windows: 86 HC pre kernels median sum 2.962 ms, 86 HC post 0.688 ms, 126 cache scatter 2.344 ms, 130 RMSNorm 0.940 ms. Source clones 2/layer but only 1 CPU clone during graph replay; copy names cannot map to clones. hc_pre_inv_rms is standalone inverse RMS, not HC pre fusion. No removable time or safe edit established.

- `2026-09-25T01:26:10Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run135`（design-check）。

- `2026-09-25T01:27:41Z` Run `run135` 记录为 `pass`；正确性为 `not-applicable`。Run107 15 valid target windows show exact 126 scatter pattern: first two c0 layers 1 each, 21 c4 layers 4 each, 20 c128 layers 2 each. DSA CP source maps these to SWA cache, compressed KV, and c4-only indexer K/scale writes. No duplicate or unnecessary cache write established; no edit or E2E.

- `2026-09-25T01:28:26Z` 主控结论为 `PIVOTED`。Run134 HC/copy census and Run135 cache scatter map reveal no source-backed semantics-safe >=2ms edit. 126 cache scatters match SWA/compressed/indexer distinct writes by layer; clone device attribution is inconclusive. The larger remaining exposed HCCL/phase skew needs causal analysis before implementation.
