# 执行日志

- `2026-09-21T04:04:17Z` Loop 已冻结。下一步：Join exact async flow launches to their innermost CPU op parent inside draft_token, rank candidate chains by clipped device union and inspect the corresponding source callsites.

- `2026-09-21T05:07:33Z` 为用例 `mixed_32k_1024_c12` 创建 Run `tp0-cpu-device-parent-20260921`（profile）。

- `2026-09-21T05:07:33Z` Run `tp0-cpu-device-parent-20260921` 记录为 `pass`；正确性为 `not-applicable`。121038/121038 draft-owned async flows exact-matched. Index and Pad chains contribute only 0.545 ms and 0.337 ms median clipped union when present, below mixed-run noise. EVENT_WAIT dominates apparent union; semantic attribution places 26.595 ms median under moe_forward_shared and 51.169 ms under the outer draft scope, so event dependencies must be separated from actual compute before any fusion claim.

- `2026-09-21T05:07:33Z` 主控结论为 `REJECTED`。Exact flow attribution falsifies the proposed Index/Pad/Copy chain as a material standalone target: aclnnIndex clipped device union is 0.545 ms median when present and ConstantPadNd 0.337 ms, both below the 4.3 percent frozen TPS spread. The dominant apparent span is EVENT_WAIT, chiefly 51.169 ms at outer draft scope and 26.595 ms under moe_forward_shared; event waits are stream dependencies and cannot be counted as compute or removable time.
