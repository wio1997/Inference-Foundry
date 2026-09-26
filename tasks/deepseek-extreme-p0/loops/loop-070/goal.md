# Loop 目标：Dynamic owner metadata Graph same-prestate gate

- Loop ID：`loop-070`
- 模式：`correctness`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-26T14:58:20Z`

## 目标/假设

Run311 differs from Run307 by capturing native compressor_metadata inside the real Graph; dynamic metadata binding may be stale or coupled to the live forward cache. A private same-prestate Graph with fresh native A/B metadata and synthetic start shifts can isolate this mechanism without live KV mutation.

## 允许修改的路径

- `scripts/loop070_dynamic_metadata_fixture.py`
- `scripts/loop070_dynamic_metadata_patch.py`
- `scripts/run_loop070_dynamic_metadata.sh`
- `scripts/check_loop070_dynamic_metadata.py`

## 目标用例

- mixed_32k_1024_c12

## 接受条件

All8 real layer2 ranks: eager full-A and Graph A/B/A2 exact producer owner writes, QLI, Sparse, and Graph qsl/start/table/cos/sin/slots match fresh native references at start deltas0,-1,-4; carrier 12x1024 and Runtime pass; borrowed sources restored.

## 否定条件

Any rank exhibits first typed/output or metadata mismatch, stale response to a reference-changing shift, invalid graph capture, service/correctness failure, or source restore failure.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
