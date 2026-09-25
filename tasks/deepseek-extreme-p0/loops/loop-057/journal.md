# 执行日志

- `2026-09-25T15:16:47Z` Loop 已冻结。下一步：Run230 read-only target profile/source reconciliation against pinned community MRV2 to rank specific changeable mechanisms, not generic kernel census

- `2026-09-25T15:16:47Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run230`（review）。

- `2026-09-25T15:27:37Z` Run `run230` 记录为 `pass`；正确性为 `not-applicable`。Read-only target bound: no proven >=1ms mechanism; W4A8 GMM gross conditional1.59-2.09ms, comm/cache limits; select official RMSNorm+cast fusion one-card gate >23.26us/call

- `2026-09-25T15:28:41Z` 为用例 `mixed_32k_1024_c12` 创建 Run `run231`（benchmark）。

- `2026-09-25T15:30:50Z` Run `run231` 记录为 `invalid`；正确性为 `invalid`。Official npu_rms_norm_cast not registered in installed runtime; no numerical or timing data; no service started; alternate registered op deferred to Run232
