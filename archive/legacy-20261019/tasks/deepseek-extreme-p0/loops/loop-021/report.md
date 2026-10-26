# Loop 报告：loop-021

- 结论：`REJECTED`
- 决定时间：`2026-09-21T05:07:33Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`not-applicable`
- 因果结论：`falsified`

## 决定依据

Exact flow attribution falsifies the proposed Index/Pad/Copy chain as a material standalone target: aclnnIndex clipped device union is 0.545 ms median when present and ConstantPadNd 0.337 ms, both below the 4.3 percent frozen TPS spread. The dominant apparent span is EVENT_WAIT, chiefly 51.169 ms at outer draft scope and 26.595 ms under moe_forward_shared; event waits are stream dependencies and cannot be counted as compute or removable time.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_loop021_device_parent/cpu_parent_attribution_tp0.json`
- `/data/wio/Inference_Foundry/evidence/20260921_loop021_device_parent/semantic_leaf_attribution_tp0.json`

## 下一步

Map MoE EVENT_WAIT producer/consumer streams and event source lines; measure whether waits serialize the main critical stream or only represent intended asynchronous overlap.
