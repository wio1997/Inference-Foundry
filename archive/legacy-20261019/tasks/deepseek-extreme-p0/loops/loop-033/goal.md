# Loop 目标：Own and replay the fixed target verification region

- Loop ID：`loop-033`
- 模式：`optimization`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`kernel-optimization`
- 冻结时间：`2026-09-22T10:15:13Z`

## 目标/假设

The dominant 96-token target region can be captured or replayed from Extreme-owned fixed buffers and metadata without the generic ModelRunner, preserving mutable KV/DSA semantics while eliminating eager Python launch and the current CUDAGraphMode.NONE/skip_compiled restriction.

## 允许修改的路径

- `bootstrap/vllm_target_handoff.py`
- `runtime/target_adapter.py`
- `runtime/extreme_decode.py`
- `patches/loop033_extreme_target_replay.patch`
- `evidence/20260922_loop033_target_replay`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

Eight-rank real-weight continuous decode preserves accepted tokens, state advance, cache/mirror invariants and rank parity; target/cycle critical-path time materially decreases in a matched trace; the replay boundary retains no ModelRunner or Scheduler object.

## 否定条件

Graph capture cannot safely represent mutable KV/DSA/collective state, replay changes target or accepted outputs, or segmented replay adds synchronization/launch cost without reducing the target critical path.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
