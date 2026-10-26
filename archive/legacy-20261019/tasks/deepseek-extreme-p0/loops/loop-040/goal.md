# Loop 目标：Map and reduce target quantized projection redundancy

- Loop ID：`loop-040`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T00:48:01Z`

## 目标/假设

In frozen DeepSeek V4 c12 target execution, repeat quantized projections or quantization/input movement in the ~4.82ms quant-matmul family may expose a semantics-preserving fusion or scheduling improvement that reduces target stage by at least 5ms and survives eight-rank formal E2E.

## 允许修改的路径

- `runtime`
- `scripts`
- `diagnostics`
- `patches`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Exact per-call shape, source, dependency and quantization mapping; one valid same-state eight-rank candidate passes correctness and reduces unprofiled target event median by >=5ms, followed by formal 48x32K-to-1024 c12 E2E median >=15% above Stock 543.655 tok/s.

## 否定条件

No repeated/redundant projection group or no legal fusion can save >=5ms target stage, or gain fails correctness/E2E.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
