# Loop 目标：Graph communication payload and physical-bound calibration

- Loop ID：`loop-061`
- 模式：`design`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T01:02:12Z`

## 目标/假设

Existing graph traces expose stable rank-local collective payload sizes while physical link bytes and critical-path exposure remain unknown

## 允许修改的路径

- `scripts/extreme_hccl_payload_audit.py`
- `evidence/20260926_loop061_bound`
- `ACHIEVABLE_BOUND.md`
- `PERFORMANCE_MAP.md`
- `HANDOFF.md`
- `RESULTS.md`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Eight-rank eighty-window exact count and payload audit plus source/API semantics and limits

## 否定条件

Trace size or operation counts vary or cannot be matched to graph target

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
