# Loop 目标：M019 DSA Q/K副流分支与rope布局适用性

- Loop ID：`L009-npu-qk-stream-applicability`
- 模式：`spec`
- 所属 Task：`sglang-pd-spec-36`
- 执行 Skill：`NONE`
- 冻结时间：`2026-10-18T18:25:06Z`

## 目标/假设

被引用的Q/K副流只在neox rope且未走MLAPO且存在alt_stream条件下触发；观测GLM rope_interleave可使此子候选不适用，不能据源码存在推断运行重叠

## 允许修改的路径

- `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018`
- `/Users/wio/work/sglang-learning-lab/repo-export/tuning-brainstorm`

## 目标用例

- source-contract

## 接受条件

固定源代码与观测配置哈希，追踪构造传参、分派与分支条件；CPU精确语句/门控探针，明确没有NPU事件正确性或性能证明

## 否定条件

以单个alt_stream或变量名推断实际分支；把neox分支的否定推广为整个模型没有多流；改变rope布局冒充主线优化

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
