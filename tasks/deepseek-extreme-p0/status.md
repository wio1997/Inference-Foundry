# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-046`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`12`
- 下一步：Run168 read-only source/write-set and metadata feasibility audit for first warmed measured genuine prefill forward.
- 更新时间：`2026-09-25T06:02:29Z`

## 最近 Loop

共 `46` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-037` | `PIVOTED` | `PIVOTED` | Runs100-104 passed legal serving but HTTP profiler RPC activation/stop latency varied from immediate to 23 seconds, missing decode or generating 10+ GiB traces; no bounded eight-rank target attribution. The profiler control boundary must move into the Runtime cycle. |
| `loop-038` | `PIVOTED` | `PIVOTED` | Cycle-scheduled profiling met the technical attribution goal in Run107: legal 8-rank cohort and 15 canonical synchronized target windows isolate 2836 kernels/cycle and quantify compute, communication, overlap, and a 9.966 ms grouped-matmul family. Run105 invalid launcher remains preserved, so the mixed-history Loop is pivoted; forced synchronization makes performance diagnostic only. |
| `loop-039` | `PIVOTED` | `PIVOTED` | Current fused W4A8 GMM1 already skips empty experts; one-card synthetic route test gives no graph-level >=5ms candidate. Run107 GMM sums are diagnostic, not removable benefit. Highest next discriminating work is non-GMM quant matmul call attribution; GMM and communication remain open. |
| `loop-040` | `PIVOTED` | `PIVOTED` | Run131 mapped 236 quant kernels/target and alternating c4/c128 pattern. Run132 8-rank legal call probe covered wkv but no new shared-input pair. Run133 source audit shows the apparent split/indexer repeats already share quantization; no semantics-valid projection fusion candidate currently warrants another 8-rank service run. Shift to HC/clone/cache source and trace attribution; quant family remains later candidate if a concrete legal replacement appears. |
| `loop-041` | `PIVOTED` | `PIVOTED` | Run134 HC/copy census and Run135 cache scatter map reveal no source-backed semantics-safe >=2ms edit. 126 cache scatters match SWA/compressed/indexer distinct writes by layer; clone device attribution is inconclusive. The larger remaining exposed HCCL/phase skew needs causal analysis before implementation. |
| `loop-042` | `PIVOTED` | `PIVOTED` | Run137 host phase skew narrows during target and reappears after proposer, but latest-rank cadence ~54ms and no safe scheduler edit show skew alone is not a removable critical-path claim. Run140 identifies a concrete CPU-heavy eager DSpark path, directing next product-specific audit there. |
| `loop-043` | `PIVOTED` | `PIVOTED` | Run142 legal eight-rank segment timing shows the smallest safe replay boundary (Markov tail) has only ~0.695ms device interval; its 2.655ms Host work appears largely overlapped with target. Whole DSpark graph requires dynamic KV/metadata semantics and remains unproven. No high-value exposed DSpark replay candidate currently justifies implementation versus the ~46.6ms target stage. |
| `loop-044` | `PIVOTED` | `PIVOTED` | Target-stage families were decomposed through Run143-152: GMM reads active W4 weights at about 1TB/s in one-card product-shape counters, peer wait dominates apparent HCCL variation, DSA compressor calls are distinct, and small mixed-dtype allGather pairs have no large direct saving. No semantics-safe >=5ms target edit is established. Run153 identifies a larger unlocalized first-token E2E range, so pivot to E2E boundary attribution. |
| `loop-045` | `PIVOTED` | `PIVOTED` | Legal diagnostics localize pre-handoff wall to prefill _model_forward, with one profiled call showing Host submission pacing; no safe removable prefill edit or formal E2E gain yet. Tail parked-slot fraction is 18.525%, but source-safe compaction remains unproven. Move to exact-state prefill execution feasibility. |
| `loop-046` | `EVALUATING` | `PENDING` | 审查 Run run169 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
