# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`DESIGNING`
- 活动 Loop：`NONE`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`12`
- 下一步：Open Loop054: low-overhead target GMM versus exposed communication benefit discriminator using saved Run116/Run107/Run98 evidence, then bounded intervention.
- 更新时间：`2026-09-25T11:04:22Z`

## 最近 Loop

共 `53` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-044` | `PIVOTED` | `PIVOTED` | Target-stage families were decomposed through Run143-152: GMM reads active W4 weights at about 1TB/s in one-card product-shape counters, peer wait dominates apparent HCCL variation, DSA compressor calls are distinct, and small mixed-dtype allGather pairs have no large direct saving. No semantics-safe >=5ms target edit is established. Run153 identifies a larger unlocalized first-token E2E range, so pivot to E2E boundary attribution. |
| `loop-045` | `PIVOTED` | `PIVOTED` | Legal diagnostics localize pre-handoff wall to prefill _model_forward, with one profiled call showing Host submission pacing; no safe removable prefill edit or formal E2E gain yet. Tail parked-slot fraction is 18.525%, but source-safe compaction remains unproven. Move to exact-state prefill execution feasibility. |
| `loop-046` | `PIVOTED` | `PIVOTED` | Exact-state prefill graph has no within-cohort shape reuse; existing DSA CP graph rejects prefill. Legal250/500ms Core admission holds aggregate requests but yield no material net diagnostic envelope gain and worsen or preserve four prefill calls. No correctness-gated formal E2E-worthy candidate. Return to dominant ~46.56ms target and inactive-slot tail. |
| `loop-047` | `PIVOTED` | `PIVOTED` | Run175 tail has 122 partial cycles but Run180 matched same-service Stock target-size event screen gives only 0.471-0.602s/cohort before substantial c12 ABI packing and state costs; direct Extreme benefit unproven and broad implementation has low expected product value relative to prefill Host work |
| `loop-048` | `PIVOTED` | `PIVOTED` | Run186 shows prefill forward thread CPU near wall but no concrete removable inner-layer fraction; Run188 proves full-cohort consolidation from7/8 to1 prefill call after1.089s wait yet only0.163s diagnostic envelope improvement vs no-wait A/A2 mean, with32 extra decode cycles. No justified formal E2E or KEEP candidate. |
| `loop-049` | `PIVOTED` | `PIVOTED` | Run190 finds static expert_map execution/weight-loader mismatch needing substantial correctness integration. Run191 fixed-map pair-swap simulation over two captured cycles gains57/78 active reads in sample but only+3/-1 on the other cycle; no robust transferable reduction. Run189 ideal2.0-2.2ms/cycle is unattainable arithmetic, and separate-service GMM rank spread<0.45ms. Deprioritize placement; no code or formal E2E. |
| `loop-050` | `PIVOTED` | `PIVOTED` | Run192/193 profiled first-collective wait traces target arrival skew, but Run194 proposer scope is8.62x Run98 low-overhead event, so profile skew cannot be promoted to product savings. Run195 low-overhead8-rank proposer duration spread median0.063ms and target0.090ms across260 steady cycles; no persistent large imbalance. Remaining HCCL kernels sum only about5.2ms in profile and have no concrete removable mechanism. Pivot from arrival skew toward required target compute/traffic and independent bound review. |
| `loop-051` | `PIVOTED` | `PIVOTED` | Run197 maps exact source/trace chain and finds c128 20 scatter kernels only0.337ms/cycle gross profiled, while all c128+c4 matched scatter is1.123ms/cycle. Compressor ABI has cmp_kv/state_cache outputs but no final cache/slot inputs; direct-write needs intrusive kernel/tiling integration and state parity. This is too small a screened return for next implementation compared with prefill Host exposure, not proof of hardware bound. No product gain or E2E candidate. |
| `loop-052` | `PIVOTED` | `PIVOTED` | Run200 legal same-service prefill-only same-stream candidate activated8/8 but B shape-matched forward wall is18.4ms slower than A2; client changes confounded by decode cycles and response hashes unstable even between controls. Run202 one-card same-stream event pair Host cost15.65us, making even2 redundant pairs per43 layers ~1.35ms/forward gross, ~12ms/9-call cohort. Event/stream simplification has no material observed product value. Remaining prefill active CPU is broader operator submission; pursue bounded fixed-model native path only if its removable fraction can be demonstrated. |
| `loop-053` | `PIVOTED` | `PIVOTED` | Run208 verifies fixed shape/address but changing slot/block/SAS/QLI values; Run209 maps fresh metadata producers and SWA/compressor/indexer writes. No isolated >=0.5s/cohort removable Host segment or bounded safe prefill graph capture; defer invasive cross-module replay. |

## 阻塞项

- 暂无
