# Loop 目标：Trace prepare input item callsites without profiler

- Loop ID：`loop-010`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T18:18:35Z`

## 目标/假设

Repeated prepare input Tensor.item calls concentrate in a few host metadata callsites; device and elapsed-time attribution can identify a safe >=5 percent candidate

## 允许修改的路径

- `/data/wio/Inference_Foundry/scripts`
- `/data/wio/Inference_Foundry/evidence`
- `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Reversible tracing captures source line, tensor device, call count, and cumulative time across a warm decode window; result selects or falsifies one bounded candidate

## 否定条件

No repeated callsite explains material time, or tracing cannot run safely on the frozen service

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
