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
- 当前知识条目：`17`
- 下一步：Map exact-shape non-GMM source read/workspace/state traffic and compulsory arithmetic, then select a same-state Graph-path dependency or kernel intervention with numerical/KV/state gates and repeated formal E2E; revisit architecture if exposed local gain is small
- 更新时间：`2026-09-26T01:17:13Z`

## 最近 Loop

共 `61` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-052` | `PIVOTED` | `PIVOTED` | Run200 legal same-service prefill-only same-stream candidate activated8/8 but B shape-matched forward wall is18.4ms slower than A2; client changes confounded by decode cycles and response hashes unstable even between controls. Run202 one-card same-stream event pair Host cost15.65us, making even2 redundant pairs per43 layers ~1.35ms/forward gross, ~12ms/9-call cohort. Event/stream simplification has no material observed product value. Remaining prefill active CPU is broader operator submission; pursue bounded fixed-model native path only if its removable fraction can be demonstrated. |
| `loop-053` | `PIVOTED` | `PIVOTED` | Run208 verifies fixed shape/address but changing slot/block/SAS/QLI values; Run209 maps fresh metadata producers and SWA/compressor/indexer writes. No isolated >=0.5s/cohort removable Host segment or bounded safe prefill graph capture; defer invasive cross-module replay. |
| `loop-054` | `PIVOTED` | `PIVOTED` | Generic GMM-versus-communication timing would repeat prior evidence without a changeable mechanism. Existing GMM traffic is near packed active bytes in one-card screens; first HCCL span is arrival wait in perturbed profile. Independent review and Sol select a bounded prefill MoE-only graph feasibility test. |
| `loop-055` | `PIVOTED` | `PIVOTED` | One-graph-per-layer all43 first88 bank failed synchronized OOM at layer17 on all8; sparse first88 call frequency and four-layer diagnostic imply smaller bank lacks material product E2E headroom. Community MRV2 source confirms reusable target/draft graph machinery but not this fixed DSA-CP whole-prefill contract. |
| `loop-056` | `PIVOTED` | `PIVOTED` | Community MRV2 DSpark graph management exists but current frozen V1 handoff lacks recursive dynamic metadata/KV/slot/output ownership; Run142/98 do not prove >=1ms exposed gain. Model-only shadow remains deferred option; target46.56ms dominates. |
| `loop-057` | `PIVOTED` | `PIVOTED` | Run233 captured surrogate at synthetic96 rows saved0.45us/call; Run235 real-local12-row surrogate fails FP32 router-output numerical gate before timing; official MRV2 fused op is unregistered in installed runtime. Read-only Run234 source/census found no defensible >=1ms/cycle target replacement; task sums and shared graph pool are not product savings. |
| `loop-058` | `PIVOTED` | `PIVOTED` | Executable V0 replay closes Run99 formal samples exactly and preserves trajectory; TP8 chain and wave audit constrain communication capacity and locate E2E variability, but scenario savings and true hardware bound remain unvalidated |
| `loop-059` | `PIVOTED` | `PIVOTED` | Legal all-rank boundary pass localizes Run99 residual to variable prefill and decode work rather than small publication/admission edges; Run188 shows phase reductions couple to decode cycles, and no true hardware floor or E2E candidate has been established |
| `loop-060` | `PIVOTED` | `PIVOTED` | Real eight-rank graph HBM task counters and exact shapes are measured, but unique compulsory traffic, HCCL link bytes, attainable graph bandwidth and dependency-critical exposure remain unproved; no correctness-preserving same-shape candidate has passed a formal E2E intervention. The frozen-product hardware-attainable throughput bound is not identifiable from these measurements. |
| `loop-061` | `PIVOTED` | `PIVOTED` | Eight-rank graph HCCL operation payload and same-host official HCCL Test capability are measured, but trace Size(Byte) is not physical link bytes, standalone test is not product Graph path, and critical-path overlap remains unknown; Astra High confirms V0 future ranges are sensitivities rather than attainable bounds. No E2E candidate or true hardware ceiling established. |

## 阻塞项

- 暂无
