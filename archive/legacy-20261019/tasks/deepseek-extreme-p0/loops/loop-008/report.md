# Loop 报告：loop-008

- 结论：`ACCEPTED`
- 决定时间：`2026-09-20T17:59:15Z`
- 模式：`design`
- 冻结目标是否满足：`yes`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

Short TP0 torch-NPU profile separates warm and cold CPU scopes and device kernels, locating 0.436s nested aten::item within warm prepare input and 1.379s rank-local device inactivity; source and exposure remain unresolved, but the diagnostic design goal is satisfied.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_scope_profile/run1/scope_summary.json`
- `/data/wio/Inference_Foundry/evidence/20260920_scope_profile/run1/device_phase_summary.json`
- `/data/wio/Inference_Foundry/evidence/20260920_scope_profile/run1/raw_index.json`

## 下一步

Loop009: find repeated item synchronization source and causal exposed time, then test minimal semantically safe change with unprofiled matched A/B and numerical correctness
