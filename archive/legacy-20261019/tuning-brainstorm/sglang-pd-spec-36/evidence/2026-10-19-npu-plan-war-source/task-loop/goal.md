# Loop 目标：M017/M018 NPU plan/调度流与 WAR 生命周期边界

- Loop ID：`L026-npu-plan-war-lifetime`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T23:26:30Z`

## 目标/假设

L025 numerical CPU equivalence cannot prove item bypass preserves H2D/WAR/PD/free ordering. Trace actual producer call sites and conditional stream/event ownership before native acceptance.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Canonical real call chain and explicit dependency/CPU-buffer/PD ownership inventory, distinguishing source contracts and needed native traces; no hidden synchronization assumption accepted.

## 否定条件

Treat host enqueue order as device completion; infer native safety or speedup from CPU metadata/mock events; mutate unowned service or bypass resource lease.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
