# Loop 目标：Reduce target MoE grouped matmul cost

- Loop ID：`loop-039`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-24T15:17:37Z`

## 目标/假设

The target graph contains 86 grouped-matmul kernels per c12 cycle summing about 9.97 ms in synchronized traces; a lower-cost backend or better fusion for the W4A8 MoE grouped-matmul path can remove at least 5 ms from the 46.56 ms unprofiled target stage without changing model state or outputs.

## 允许修改的路径

- `runtime`
- `scripts`
- `diagnostics`
- `patches`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Identify exact grouped-matmul call sites and shapes, pass same-state eight-rank target/KV/DSpark correctness, reduce unprofiled target NPU event median by at least 5 ms, then meet frozen formal 48x32K-to-1024 c12 E2E with median at least 15 percent above Stock 543.655 tok/s.

## 否定条件

No legal operator variant preserves W4A8 semantics, grouped-matmul cost cannot be reduced by 5 ms on real shapes, or E2E gain does not survive legal serving gates.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
