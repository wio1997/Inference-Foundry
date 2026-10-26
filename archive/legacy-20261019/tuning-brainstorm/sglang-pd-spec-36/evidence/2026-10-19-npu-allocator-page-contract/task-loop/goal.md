# Loop 目标：M005/M011 actual allocator 页单位与释放所有权

- Loop ID：`L022-allocator-page-ownership`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T22:18:34Z`

## 目标/假设

L021 source tree ledger with allocator fixtures cannot prove actual allocation/free. Resolve actual allocator factory/page units and constructible CPU metadata path before native PD+MTP.

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- c2-short
- c3-long

## 接受条件

Trace canonical allocator classes/factory and verify actual supported CPU alloc/free/available/budget behavior with explicit native and payload limits.

## 否定条件

Call fixture page counter an actual allocator; infer NPU physical free/visibility or task runtime backend from source-only; mutate unowned service/KV.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
