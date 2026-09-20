# 执行日志

- `2026-09-20T16:32:38Z` Loop 已冻结。下一步：Make FlashComm1 a single env toggle; restart with false; warm, check golden, benchmark three passes

- `2026-09-20T16:35:57Z` 为用例 `mixed_32k_1024_c12` 创建 Run `flashcomm-off-20260920`（benchmark）。

- `2026-09-20T16:39:05Z` Run `flashcomm-off-20260920` 记录为 `error`；正确性为 `not-applicable`。Engine startup failed before model load: DSA CP requires SP/FlashComm1; no request or benchmark occurred

- `2026-09-20T16:39:05Z` 主控结论为 `PIVOTED`。FlashComm1-off alone violates vllm-ascend DSA CP requires SP constraint; config failed during worker init before performance measurement
