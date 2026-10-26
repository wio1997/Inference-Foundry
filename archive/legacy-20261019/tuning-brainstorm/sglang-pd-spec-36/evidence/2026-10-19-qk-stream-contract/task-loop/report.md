# Loop 报告：L009-npu-qk-stream-applicability

- 结论：`INCONCLUSIVE`
- 决定时间：`2026-10-18T18:38:40Z`
- 模式：`spec`
- 冻结目标是否满足：`yes`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

Conditional layout gate excludes named neox Q/K subcandidate for unchanged observed interleaved config; actual object not introspected. Python wrapper contract suggests independent MLAPO argument mismatch, but native import unavailable in CPU isolation and class identity/event behavior unverified. One-line candidate retained unapplied. No NPU timing or PD result.

## 证据

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L009-R002-layout-cpu/result.json`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L009-R004-stream-type-cpu/result.json`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L009-R005-wrapper-contract/result.json`

## 下一步

Publish M019 conditional support evidence and unapplied candidate; prepare M001 tokenizer/template identity and frozen-input fixtures without NPU allocation.
