# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-051`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`12`
- 下一步：Run197 read-only source and trace audit of c128 compressor output and compressed-KV scatter ownership, shape, bytes, dependency, and actual exclusive device time before design.
- 更新时间：`2026-09-25T09:39:57Z`

## 最近 Loop

共 `51` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-042` | `PIVOTED` | `PIVOTED` | Run137 host phase skew narrows during target and reappears after proposer, but latest-rank cadence ~54ms and no safe scheduler edit show skew alone is not a removable critical-path claim. Run140 identifies a concrete CPU-heavy eager DSpark path, directing next product-specific audit there. |
| `loop-043` | `PIVOTED` | `PIVOTED` | Run142 legal eight-rank segment timing shows the smallest safe replay boundary (Markov tail) has only ~0.695ms device interval; its 2.655ms Host work appears largely overlapped with target. Whole DSpark graph requires dynamic KV/metadata semantics and remains unproven. No high-value exposed DSpark replay candidate currently justifies implementation versus the ~46.6ms target stage. |
| `loop-044` | `PIVOTED` | `PIVOTED` | Target-stage families were decomposed through Run143-152: GMM reads active W4 weights at about 1TB/s in one-card product-shape counters, peer wait dominates apparent HCCL variation, DSA compressor calls are distinct, and small mixed-dtype allGather pairs have no large direct saving. No semantics-safe >=5ms target edit is established. Run153 identifies a larger unlocalized first-token E2E range, so pivot to E2E boundary attribution. |
| `loop-045` | `PIVOTED` | `PIVOTED` | Legal diagnostics localize pre-handoff wall to prefill _model_forward, with one profiled call showing Host submission pacing; no safe removable prefill edit or formal E2E gain yet. Tail parked-slot fraction is 18.525%, but source-safe compaction remains unproven. Move to exact-state prefill execution feasibility. |
| `loop-046` | `PIVOTED` | `PIVOTED` | Exact-state prefill graph has no within-cohort shape reuse; existing DSA CP graph rejects prefill. Legal250/500ms Core admission holds aggregate requests but yield no material net diagnostic envelope gain and worsen or preserve four prefill calls. No correctness-gated formal E2E-worthy candidate. Return to dominant ~46.56ms target and inactive-slot tail. |
| `loop-047` | `PIVOTED` | `PIVOTED` | Run175 tail has 122 partial cycles but Run180 matched same-service Stock target-size event screen gives only 0.471-0.602s/cohort before substantial c12 ABI packing and state costs; direct Extreme benefit unproven and broad implementation has low expected product value relative to prefill Host work |
| `loop-048` | `PIVOTED` | `PIVOTED` | Run186 shows prefill forward thread CPU near wall but no concrete removable inner-layer fraction; Run188 proves full-cohort consolidation from7/8 to1 prefill call after1.089s wait yet only0.163s diagnostic envelope improvement vs no-wait A/A2 mean, with32 extra decode cycles. No justified formal E2E or KEEP candidate. |
| `loop-049` | `PIVOTED` | `PIVOTED` | Run190 finds static expert_map execution/weight-loader mismatch needing substantial correctness integration. Run191 fixed-map pair-swap simulation over two captured cycles gains57/78 active reads in sample but only+3/-1 on the other cycle; no robust transferable reduction. Run189 ideal2.0-2.2ms/cycle is unattainable arithmetic, and separate-service GMM rank spread<0.45ms. Deprioritize placement; no code or formal E2E. |
| `loop-050` | `PIVOTED` | `PIVOTED` | Run192/193 profiled first-collective wait traces target arrival skew, but Run194 proposer scope is8.62x Run98 low-overhead event, so profile skew cannot be promoted to product savings. Run195 low-overhead8-rank proposer duration spread median0.063ms and target0.090ms across260 steady cycles; no persistent large imbalance. Remaining HCCL kernels sum only about5.2ms in profile and have no concrete removable mechanism. Pivot from arrival skew toward required target compute/traffic and independent bound review. |
| `loop-051` | `EVALUATING` | `PENDING` | 审查 Run run196 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
