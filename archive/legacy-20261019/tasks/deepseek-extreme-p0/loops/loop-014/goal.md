# Loop 目标：Find exposed decode metadata host work

- Loop ID：`loop-014`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T20:28:19Z`

## 目标/假设

Warm DP1TP8 c12 input preparation repeats Python metadata work across 170 decode iterations; at least one deterministic repeated operation can be safely cached or vectorized and reduce E2E output latency beyond benchmark noise.

## 允许修改的路径

- `/data/wio/vllm_ascend_26/framework/vllm-ascend`
- `scripts`
- `patches`
- `evidence`
- `tasks`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Identify a single source-level repeated operation with measured host self-time and an invariant, implement a minimal gated patch, pass functional/numerical correctness, and improve paired unprofiled 48×1024 c12 output TPS by at least 5% without TTFT regression.

## 否定条件

No single correct change has a defensible >5% opportunity or A/B gain is within 4.3% baseline spread; reject or pivot to cold prefill/framework layer.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
