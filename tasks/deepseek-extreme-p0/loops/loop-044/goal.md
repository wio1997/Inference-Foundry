# Loop 目标：Find removable target device critical path

- Loop ID：`loop-044`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-25T02:59:00Z`

## 目标/假设

The frozen 96-token target stage (~46.6ms unprofiled) contains a larger exposed device compute path than the sub-ms DSpark Markov tail; per-layer and per-family interval attribution can identify a semantics-preserving product-specific change worth >=5ms target cadence and eventual formal E2E gain.

## 允许修改的路径

- `runtime`
- `bootstrap`
- `scripts`
- `patches`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Identify a source-backed candidate with a credible >=5ms target critical-path reduction, preserve same-state eight-rank target/KV/DSpark correctness, and demonstrate legal formal E2E gain over Run99.

## 否定条件

Valid trace and source show all >5ms kernel families are required or fully overlapped, or proposed change fails correctness or cadence/E2E comparison.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
