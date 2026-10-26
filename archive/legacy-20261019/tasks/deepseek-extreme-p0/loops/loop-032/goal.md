# Loop 目标：Move DSpark host mirrors behind the proposer

- Loop ID：`loop-032`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`kernel-optimization`
- 冻结时间：`2026-09-22T09:27:52Z`

## 目标/假设

For fixed c12 DSpark7, bootstrap CPU metadata plus a side-stream copy of the 12 acceptance counts can advance exact DSA host mirrors for the next cycle, eliminating the three-tensor synchronous refresh from the current proposer critical path without changing tokens, KV state, or rank parity.

## 允许修改的路径

- `bootstrap/vllm_dspark_handoff.py`
- `runtime/extreme_decode.py`
- `scripts/analyze_extreme_runtime_profile.py`
- `patches/loop032_dspark_host_mirror_overlap.patch`
- `evidence/20260922_loop032_host_mirror`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Real-weight DP1xTP8 completes eight continuous cycles on all ranks with exact state/rank parity; no query_start_loc, target_seq_lens, or num_computed_tokens device-to-host copy remains in refresh_fixed_common; trace or controlled timing shows the refresh barrier removed or displaced behind proposer work.

## 否定条件

Any accepted token/state mismatch, DSA metadata error, rank divergence, or unavoidable wait equal to the original synchronous refresh falsifies the proposed handoff.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
