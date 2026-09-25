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
- 下一步：Find a concrete executable graph-compatible target or DSpark mechanism with source-backed >=1ms/cycle exposed saving and same-state correctness plan before starting another service; current Run107/98 profiles and MRV2 audit are reusable, avoid repeat census or Run93 E2E
- 更新时间：`2026-09-25T15:46:30Z`

## 最近 Loop

共 `57` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-048` | `PIVOTED` | `PIVOTED` | Run186 shows prefill forward thread CPU near wall but no concrete removable inner-layer fraction; Run188 proves full-cohort consolidation from7/8 to1 prefill call after1.089s wait yet only0.163s diagnostic envelope improvement vs no-wait A/A2 mean, with32 extra decode cycles. No justified formal E2E or KEEP candidate. |
| `loop-049` | `PIVOTED` | `PIVOTED` | Run190 finds static expert_map execution/weight-loader mismatch needing substantial correctness integration. Run191 fixed-map pair-swap simulation over two captured cycles gains57/78 active reads in sample but only+3/-1 on the other cycle; no robust transferable reduction. Run189 ideal2.0-2.2ms/cycle is unattainable arithmetic, and separate-service GMM rank spread<0.45ms. Deprioritize placement; no code or formal E2E. |
| `loop-050` | `PIVOTED` | `PIVOTED` | Run192/193 profiled first-collective wait traces target arrival skew, but Run194 proposer scope is8.62x Run98 low-overhead event, so profile skew cannot be promoted to product savings. Run195 low-overhead8-rank proposer duration spread median0.063ms and target0.090ms across260 steady cycles; no persistent large imbalance. Remaining HCCL kernels sum only about5.2ms in profile and have no concrete removable mechanism. Pivot from arrival skew toward required target compute/traffic and independent bound review. |
| `loop-051` | `PIVOTED` | `PIVOTED` | Run197 maps exact source/trace chain and finds c128 20 scatter kernels only0.337ms/cycle gross profiled, while all c128+c4 matched scatter is1.123ms/cycle. Compressor ABI has cmp_kv/state_cache outputs but no final cache/slot inputs; direct-write needs intrusive kernel/tiling integration and state parity. This is too small a screened return for next implementation compared with prefill Host exposure, not proof of hardware bound. No product gain or E2E candidate. |
| `loop-052` | `PIVOTED` | `PIVOTED` | Run200 legal same-service prefill-only same-stream candidate activated8/8 but B shape-matched forward wall is18.4ms slower than A2; client changes confounded by decode cycles and response hashes unstable even between controls. Run202 one-card same-stream event pair Host cost15.65us, making even2 redundant pairs per43 layers ~1.35ms/forward gross, ~12ms/9-call cohort. Event/stream simplification has no material observed product value. Remaining prefill active CPU is broader operator submission; pursue bounded fixed-model native path only if its removable fraction can be demonstrated. |
| `loop-053` | `PIVOTED` | `PIVOTED` | Run208 verifies fixed shape/address but changing slot/block/SAS/QLI values; Run209 maps fresh metadata producers and SWA/compressor/indexer writes. No isolated >=0.5s/cohort removable Host segment or bounded safe prefill graph capture; defer invasive cross-module replay. |
| `loop-054` | `PIVOTED` | `PIVOTED` | Generic GMM-versus-communication timing would repeat prior evidence without a changeable mechanism. Existing GMM traffic is near packed active bytes in one-card screens; first HCCL span is arrival wait in perturbed profile. Independent review and Sol select a bounded prefill MoE-only graph feasibility test. |
| `loop-055` | `PIVOTED` | `PIVOTED` | One-graph-per-layer all43 first88 bank failed synchronized OOM at layer17 on all8; sparse first88 call frequency and four-layer diagnostic imply smaller bank lacks material product E2E headroom. Community MRV2 source confirms reusable target/draft graph machinery but not this fixed DSA-CP whole-prefill contract. |
| `loop-056` | `PIVOTED` | `PIVOTED` | Community MRV2 DSpark graph management exists but current frozen V1 handoff lacks recursive dynamic metadata/KV/slot/output ownership; Run142/98 do not prove >=1ms exposed gain. Model-only shadow remains deferred option; target46.56ms dominates. |
| `loop-057` | `PIVOTED` | `PIVOTED` | Run233 captured surrogate at synthetic96 rows saved0.45us/call; Run235 real-local12-row surrogate fails FP32 router-output numerical gate before timing; official MRV2 fused op is unregistered in installed runtime. Read-only Run234 source/census found no defensible >=1ms/cycle target replacement; task sums and shared graph pool are not product savings. |

## 阻塞项

- 暂无
