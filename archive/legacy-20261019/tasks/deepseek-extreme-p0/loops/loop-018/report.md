# Loop 报告：loop-018

- 结论：`PIVOTED`
- 决定时间：`2026-09-21T01:13:02Z`
- 模式：`optimization`
- 冻结目标是否满足：`no`
- 可比性：`invalid`
- 因果结论：`not-identifiable`

## 决定依据

Saved cross-rank timestamps are confounded by stable ~579us rank6 clock offset plus residual ~206us end spread. Communication elapsed is Idle-only, so collective arrival skew cannot be separated from clock/reporting artifacts or translated to a safe high-value change using current evidence.

## 证据

- `/data/wio/Inference_Foundry/evidence/20260921_decode_comm_audit/collective_arrival_skew.json`
- `/data/wio/Inference_Foundry/evidence/20260921_decode_comm_audit/collective_clock_sensitivity.json`

## 下一步

Start Loop019: measure warm decode host metadata builder cost with paired no-profiler service traces and seek an active-callsite caching or vectorization candidate.
