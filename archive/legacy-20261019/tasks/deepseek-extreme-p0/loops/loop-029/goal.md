# Loop 目标：Standalone multi-cycle decode runtime v0

- Loop ID：`loop-029`
- 模式：`implementation`
- 所属 Task：`deepseek-extreme-p0`
- 执行 Skill：`NONE`
- 冻结时间：`2026-09-21T13:23:51Z`

## 目标/假设

For the frozen c12 DeepSeek V4 Flash W4A8 DP1TP8 DSpark7 envelope, a dedicated fixed-buffer runtime can execute consecutive target verification, acceptance, state advance and three-layer proposal cycles with the existing weights/operators while bypassing per-cycle SchedulerOutput, request objects, dynamic metadata construction and generic model-runner dispatch.

## 允许修改的路径

- `runtime`
- `scripts`
- `patches`
- `evidence`
- `tasks/deepseek-extreme-p0`
- `SPECIALIZED_RUNTIME.md`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`

## 目标用例

- mixed_32k_1024_c12
- correctness_short

## 接受条件

Execute at least 8 causally connected decode cycles on all 8 ranks through a dedicated runtime entry; match oracle emitted/accepted tokens and every touched target/draft KV or recurrent state boundary at defined exact/tolerance checks; perform no per-cycle SchedulerOutput/request-object construction in the dedicated loop.

## 否定条件

The fixed model requires an unrepresentable host-side semantic dependency or the dedicated loop cannot reproduce oracle cycle-to-cycle outputs/state after correcting explicit buffer ownership and ordering.

## 固定条件与禁止变更

执行前在此记录固定接口、测量口径和实现边界。
