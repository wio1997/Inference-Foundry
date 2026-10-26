# Loop 目标：Non-GMM compulsory traffic and exposed target dependency

- Loop ID：`loop-062`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T01:18:00Z`

## 目标/假设

One non-GMM target shape has repeat/materialization traffic that can be reduced without changing KV/state/acceptance, and the reduction reaches the latest-rank target boundary

## 允许修改的路径

- `scripts/extreme_compressor_traffic_census.py`
- `evidence/20260926_loop062_nongmm`
- `ACHIEVABLE_BOUND.md`
- `PERFORMANCE_MAP.md`
- `RESULTS.md`
- `HANDOFF.md`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Exact shape and source read census, calibrated same-state Graph-path intervention with numerical/state gates, then repeated formal E2E if exposed gain exceeds noise

## 否定条件

Reads are necessary or localized speedup is hidden by overlap, dependency, acceptance or full E2E

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
