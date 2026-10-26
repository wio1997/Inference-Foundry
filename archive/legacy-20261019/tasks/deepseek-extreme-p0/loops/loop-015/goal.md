# Loop 目标：Isolate cold prefill device critical path

- Loop ID：`loop-015`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T21:05:37Z`

## 目标/假设

After the kept QLI prefill change, cold 32K TTFT has a removable device-side ScatterNdUpdateSk or Compressor cost that can be isolated from HCCL waiting and reduced without changing model numerics.

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

- cold_32k_128_c1
- mixed_32k_1024_c12
- correctness_short

## 接受条件

Identify a current critical-path kernel or host-to-device dependency with same-prompt evidence, implement a minimal correct change, and lower cold TTFT by at least 5% beyond Loop012 with no mixed 48x1024 c12 regression.

## 否定条件

Candidate kernel spans overlap or are required, no supported change can cut exposed TTFT, or paired A/B fails correctness/performance; pivot to another path.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
