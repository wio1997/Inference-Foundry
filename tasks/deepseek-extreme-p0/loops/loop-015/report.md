# Loop 报告：loop-015

- 结论：`PIVOTED`
- 决定时间：`2026-09-20T21:28:18Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

Captured one real 8096-row long-prefill scatter mapping per rank: all unique, but uniqueness not yet a general invariant. Existing V2 op gave bit-equal output and was 69% slower than SK in 25-call isolated NPU timing; no safe E2E patch or paired TTFT gain. Source SK deterministic sort+SyncAll suggests a direct unique-index path merits a separate build experiment.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260920_loop015_cold_kernels/probe`
- `/data/wio/Inference_Foundry/evidence/20260920_loop015_cold_kernels/scatter_v2_screen.json`

## 下一步

Loop016: inspect slot mapping construction and prototype a direct unique-index scatter path; require duplicate-safe fallback, per-call correctness, and paired cold TTFT before KEEP.
