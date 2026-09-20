# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`DESIGNING`
- 活动 Loop：`loop-009`
- 已接受基线：`NONE`
- 证据成熟度：`E2_BENCHMARKED`
- 用例数：`3`
- 当前知识条目：`1`
- 下一步：Enable torch profiler with_modules stack for a short warm window; inspect item call stacks and exact source before any patch
- 更新时间：`2026-09-20T17:59:53Z`

## 最近 Loop

共 `9` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-001` | `PIVOTED` | `PIVOTED` | Initial SSE client omitted DeepSeek reasoning deltas, making first TTFT/TPOT invalid; corrected parser and preserved evidence |
| `loop-002` | `ACCEPTED` | `ACCEPTED` | Three corrected 48/48 warm-cache runs, golden 4/4; invalid parser run isolated in pivoted Loop 001 |
| `loop-003` | `PIVOTED` | `PIVOTED` | TP0 msprof distinguishes cold and warm compute/HCCL; warm HCCL union 1.165s of 3.401s with ~0.054s compute overlap, justifying a falsifiable FlashComm1 A/B; no same-condition optimization comparison yet |
| `loop-004` | `PIVOTED` | `PIVOTED` | FlashComm1-off alone violates vllm-ascend DSA CP requires SP constraint; config failed during worker init before performance measurement |
| `loop-005` | `PIVOTED` | `PIVOTED` | Strict golden4 exact-output gate is invalid: 4/4 mismatch even between repeat runs on unchanged candidate; cannot infer candidate correctness failure or performance |
| `loop-006` | `REJECTED` | `REJECTED` | Coupled SP/DSA-CP-off path has no robust E2E benefit: median output TPS -1.61% vs baseline and inside observed noise; TTFT median worsened ~7.8%, TPOT ~0.7%. Functional check passed but numerical equivalence remains unproven. |
| `loop-007` | `REJECTED` | `REJECTED` | DSA CP off alone regressed full warmed E2E performance: median output TPS -4.70%, TTFT +15.1%, TPOT +3.8%; no >5% gain. Functional gate passed; numerical equivalence not needed for rejected candidate. |
| `loop-008` | `ACCEPTED` | `ACCEPTED` | Short TP0 torch-NPU profile separates warm and cold CPU scopes and device kernels, locating 0.436s nested aten::item within warm prepare input and 1.379s rank-local device inactivity; source and exposure remain unresolved, but the diagnostic design goal is satisfied. |
| `loop-009` | `FROZEN` | `PENDING` | Enable torch profiler with_modules stack for a short warm window; inspect item call stacks and exact source before any patch |

## 阻塞项

- 暂无
