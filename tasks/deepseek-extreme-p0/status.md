# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-007`
- 已接受基线：`NONE`
- 证据成熟度：`E2_BENCHMARKED`
- 用例数：`3`
- 当前知识条目：`1`
- 下一步：Parameterize runner config assertions, stop rejected service, launch FlashComm1 true/DSA CP false, then full A/B
- 更新时间：`2026-09-20T17:06:14Z`

## 最近 Loop

共 `7` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-001` | `PIVOTED` | `PIVOTED` | Initial SSE client omitted DeepSeek reasoning deltas, making first TTFT/TPOT invalid; corrected parser and preserved evidence |
| `loop-002` | `ACCEPTED` | `ACCEPTED` | Three corrected 48/48 warm-cache runs, golden 4/4; invalid parser run isolated in pivoted Loop 001 |
| `loop-003` | `PIVOTED` | `PIVOTED` | TP0 msprof distinguishes cold and warm compute/HCCL; warm HCCL union 1.165s of 3.401s with ~0.054s compute overlap, justifying a falsifiable FlashComm1 A/B; no same-condition optimization comparison yet |
| `loop-004` | `PIVOTED` | `PIVOTED` | FlashComm1-off alone violates vllm-ascend DSA CP requires SP constraint; config failed during worker init before performance measurement |
| `loop-005` | `PIVOTED` | `PIVOTED` | Strict golden4 exact-output gate is invalid: 4/4 mismatch even between repeat runs on unchanged candidate; cannot infer candidate correctness failure or performance |
| `loop-006` | `REJECTED` | `REJECTED` | Coupled SP/DSA-CP-off path has no robust E2E benefit: median output TPS -1.61% vs baseline and inside observed noise; TTFT median worsened ~7.8%, TPOT ~0.7%. Functional check passed but numerical equivalence remains unproven. |
| `loop-007` | `RUNNING` | `PENDING` | 执行 Run dsa-cp-off-20260920：EXPECTED_FLASHCOMM1=true EXPECTED_DSA_CP=false ALLOW_NONDETERMINISTIC_GOLDEN=1 CANDIDATE_OUT=evidence/20260920_dsa_cp_off bash scripts/run_flashcomm_candidate.sh |

## 阻塞项

- 暂无
