# Loop 目标：Resolve target draft host and cold prefill critical path

- Loop ID：`loop-008`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T17:28:41Z`

## 目标/假设

Torch-NPU profiler scopes and existing TP0 op timelines can distinguish target, draft, host preparation and cold prefill exposed time enough to rank a falsifiable >=5 percent candidate

## 允许修改的路径

- `/data/wio/Inference_Foundry/scripts`
- `/data/wio/Inference_Foundry/evidence`

## 目标用例

- mixed_32k_1024_c12
- cold_32k_128_c1

## 接受条件

Current baseline configuration runs, scope traces for warmed decode and cold prefill export successfully, and evidence ranks one bounded high-value candidate without using aggregate utilization as cause

## 否定条件

Profiler scopes unavailable or traces cannot distinguish target/draft/host and prefill critical path; then use minimal source timing instrumentation

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
