# Loop 报告：L008-npu-graph-shape-support

- 结论：`INCONCLUSIVE`
- 决定时间：`2026-10-18T18:20:42Z`
- 模式：`spec`
- 冻结目标是否满足：`yes`
- 可比性：`not-applicable`
- 因果结论：`partially-supported`

## 决定依据

Source and repaired CPU policy probes provide bounded support evidence; actual NPU graph table, replay correctness and performance remain untested. First R002 extraction error remains invalid and is not erased; R003 passes in a new Run. Do not accept a graph optimization or infer configured[16] equals actual captured table.

## 证据

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L008-R002-frozen-controller/stderr.log`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L008-R003-graph-cpu/result.json`
- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018/runs/L008-R004-graph-config/source-and-config.json`

## 下一步

Publish L008 with INCONCLUSIVE runtime/performance verdict, then continue M019 event/stream ownership audit while task-owned PD resources remain unavailable.
