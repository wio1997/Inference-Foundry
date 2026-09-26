# Loop 目标：Cross-cycle target metadata scheduling against DSpark

- Loop ID：`loop-063`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T02:54:51Z`

## 目标/假设

After acceptance, next target geometry and sparse metadata can be computed into private storage concurrently with current DSpark and committed to stable Graph addresses after all old-state consumers, reducing complete latest-rank cycle time without changing frozen semantics

## 允许修改的路径

- `runtime/extreme_decode.py`
- `runtime/fixed_decode.py`
- `runtime/target_metadata.py`
- `runtime/fixed_serving.py`
- `bootstrap/vllm_target_metadata_handoff.py`
- `scripts`
- `evidence/20260926_loop063_schedule`
- `ACHIEVABLE_BOUND.md`
- `PERFORMANCE_MAP.md`
- `RESULTS.md`
- `HANDOFF.md`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Static dependency and alias audit; A0 current versus A serial scratch/commit versus B side-stream overlap; eight-rank exact geometry/RoPE/SAS/QLI stable-header, KV/acceptance/parking gates; measured latest-rank full-cycle gain and repeated frozen formal E2E before KEEP

## 否定条件

Any state/metadata/KV or Graph alias mismatch, unresolved cross-stream race, no exposed latest-rank gain, or no repeatable formal E2E improvement

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
