# 执行日志

- `2026-09-25T15:16:47Z` Loop 已冻结。下一步：Run230 read-only target profile/source reconciliation against pinned community MRV2 to rank specific changeable mechanisms, not generic kernel census

- `2026-09-25T15:16:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run230`（review）。

- `2026-09-25T15:27:37Z` Run `run230` 记录为 `pass`；正确性为 `not-applicable`。Read-only target bound: no proven >=1ms mechanism; W4A8 GMM gross conditional1.59-2.09ms, comm/cache limits; select official RMSNorm+cast fusion one-card gate >23.26us/call

- `2026-09-25T15:28:41Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run231`（benchmark）。

- `2026-09-25T15:30:50Z` Run `run231` 记录为 `invalid`；正确性为 `invalid`。Official npu_rms_norm_cast not registered in installed runtime; no numerical or timing data; no service started; alternate registered op deferred to Run232

- `2026-09-25T15:32:25Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run232`（benchmark）。

- `2026-09-25T15:33:51Z` Run `run232` 记录为 `pass`；正确性为 `pass`。One-card BF16 local parity maxabs0.00390625; eager paired saving39.33us/call, confounded by Host launch; graph replay screen required before product decision

- `2026-09-25T15:35:37Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run233`（benchmark）。

- `2026-09-25T15:36:35Z` Run `run233` 记录为 `pass`；正确性为 `pass`。One-card captured RMS+cast parity maxabs0.0009766; paired graph replay saving0.45us/call vs23.26us gate, reject product integration

- `2026-09-25T15:42:38Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run234`（review）。

- `2026-09-25T15:44:01Z` Run `run234` 记录为 `pass`；正确性为 `not-applicable`。Read-only Run107 target kernel-name census and official/shared graph-pool audit; shared pool may address Run227 OOM but first88 frequency makes product headroom modest; Run233 claim scoped to surrogate/global shape; exact-shape screen next
