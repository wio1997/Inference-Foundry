# Loop 目标：Direct unique-index DSA scatter feasibility

- Loop ID：`loop-016`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-20T21:29:38Z`

## 目标/假设

Long-prefill DSA compressor slot mapping is unique by construction, allowing a single-pass direct scatter to remove SK sort and cross-core sync on a measurable cold TTFT critical path, with a safe fallback for duplicate or invalid cases.

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

Prove path selection safe under scheduler behavior, implement minimal kernel or operator path, pass full numerical correctness and paired cold 32K TTFT improvement >=5% without mixed c12 regression.

## 否定条件

Uniqueness cannot be guaranteed cheaply, direct kernel cannot safely handle actual layouts, or paired correctness/performance does not improve.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
