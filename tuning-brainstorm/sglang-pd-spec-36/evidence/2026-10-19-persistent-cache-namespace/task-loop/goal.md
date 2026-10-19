# Loop 目标：M036 持久缓存namespace与模型生命周期

- Loop ID：`L037-persistent-cache-namespace`
- 模式：`correctness`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-19T06:55:10Z`

## 目标/假设

实际存储键及attach/clear/model-update生命周期应区分caller/adapter与缓存世代；同名模型或混版本状态不可在没有明确合同下复用旧数据。

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract
- lifecycle
- c2-short
- c3-long

## 接受条件

固定源码闭合实际key/config与生命周期，运行能区分同身份和旧namespace/模型版本的最低成本原组件实验；记录支持和否定前提。

## 否定条件

不同语义namespace/模型身份在原键路径得到同一持久键且可返回旧CPU数据，或clear/update只覆盖局部状态却被误当全局fence。

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
