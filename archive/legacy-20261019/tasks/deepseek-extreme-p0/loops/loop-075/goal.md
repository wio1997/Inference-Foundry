# Loop 目标：Eight-rank attainable resource capacity and compulsory-work calibration

- Loop ID：`loop-075`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T23:49:55Z`

## 目标/假设

Same-shape concurrent 8x910B3 GMM/HBM service plus pinned work census will reduce Resource/Hardware Bound uncertainty without treating current traffic or isolated kernel time as compulsory Product latency

## 允许修改的路径

- `scripts`
- `evidence`
- `ACHIEVABLE_BOUND.md`
- `PERFORMANCE_MAP.md`
- `performance_knowledge`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

All8 concurrent exact-shape capacity samples with environment/exit/counter checks; explicit current-versus-necessary work ledger; no unsupported numeric Product ceiling

## 否定条件

Missing rank concurrency, changed shape/layout/dtype, failed output gate, profiler contamination, or result cannot tighten any bound input

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
