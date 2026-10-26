# Loop 目标：Use CPU QLI maxima only for prefill

- Loop ID：`loop-012`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T19:20:35Z`

## 目标/假设

Restricting QLI CPU local maxima to steps with num_prefills>0 preserves the measured 10.8 percent cold TTFT gain while restoring original pure-decode timing and eliminating mixed TTFT uncertainty

## 允许修改的路径

- `/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py`
- `/data/wio/Inference_Foundry/patches`
- `/data/wio/Inference_Foundry/evidence`
- `/data/wio/Inference_Foundry/scripts`

## 目标用例

- mixed_32k_1024_c12
- cold_32k_128_c1
- correctness_short

## 接受条件

Numerical metadata parity remains supported, 8 same-offset fresh cold prompts improve >5 percent, full mixed 3x48 passes show no material TPS/TTFT regression, functional gate passes

## 否定条件

Cold gain disappears, mixed regression persists, or correctness fails

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
