# Loop 报告：loop-013

- 结论：`PIVOTED`
- 决定时间：`2026-09-20T20:26:35Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

Valid c12 trace isolates large draft host and TP8 HCCL/idle envelopes but does not prove either removable. The candidate DSpark draft graph path is hard-disabled in source; prior upstream graph attempt was reverted. Further source-level cause and numerical constraints are needed before an E2E optimization.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_decode_c12_profile/run4/device_c12_summary.json`
- `/data/wio/Inference_Foundry/evidence/20260920_decode_c12_profile/run4/host_scope_c12_summary.json`

## 下一步

Loop014: inspect repeated draft MoE/DSA host self time and cross-rank synchronization; identify a minimal correctness-preserving change with predicted measurable TPS benefit or pivot again.
