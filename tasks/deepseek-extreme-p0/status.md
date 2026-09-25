# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-044`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`12`
- 下一步：Run143: reconstruct ordered target compute/communication intervals in all 15 valid Run107 windows, attribute layer-family exposure and compare unprofiled Run98 stage; estimate candidate bounds without summing overlapping kernels.
- 更新时间：`2026-09-25T02:59:00Z`

## 最近 Loop

共 `44` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-035` | `PIVOTED` | `PIVOTED` | Frozen Loop035 technical goal is met: causal DSpark gid2 slot refresh fixes sustained acceptance; Run85/72/84/90/92 bounded correctness and Run87 DAG profile pass; formal 48x32K-to-1024 c12 Extreme median improves 217.342 to 525.417 tok/s (+141.747%), 128/128 rank/cohort rows pass. TaskCtl cannot mark accepted because preserved invalid diagnostic attempts (including Run89 OOM and Run91 unreachable fixed Stock shape) remain in this Loop; those attempts do not invalidate the official benchmark. P0 above-Stock target remains open. |
| `loop-036` | `PIVOTED` | `PIVOTED` | The frozen technical goal passes: legal 8-rank 299-cycle stable metadata header shadow and full serving gates, same-contract metadata stage 8.6730 to 0.6674 ms, and formal 48-request median 525.417 to 571.681 tok/s (+8.805%) above Stock by 5.155%. Preserved Run94/95 invalid setup and Run96 full-buffer failure prevent treating every mixed-history Run as pass; full AICPU tails have dynamic self-replay noise. Retain static implementation and pivot to target stage attribution. |
| `loop-037` | `PIVOTED` | `PIVOTED` | Runs100-104 passed legal serving but HTTP profiler RPC activation/stop latency varied from immediate to 23 seconds, missing decode or generating 10+ GiB traces; no bounded eight-rank target attribution. The profiler control boundary must move into the Runtime cycle. |
| `loop-038` | `PIVOTED` | `PIVOTED` | Cycle-scheduled profiling met the technical attribution goal in Run107: legal 8-rank cohort and 15 canonical synchronized target windows isolate 2836 kernels/cycle and quantify compute, communication, overlap, and a 9.966 ms grouped-matmul family. Run105 invalid launcher remains preserved, so the mixed-history Loop is pivoted; forced synchronization makes performance diagnostic only. |
| `loop-039` | `PIVOTED` | `PIVOTED` | Current fused W4A8 GMM1 already skips empty experts; one-card synthetic route test gives no graph-level >=5ms candidate. Run107 GMM sums are diagnostic, not removable benefit. Highest next discriminating work is non-GMM quant matmul call attribution; GMM and communication remain open. |
| `loop-040` | `PIVOTED` | `PIVOTED` | Run131 mapped 236 quant kernels/target and alternating c4/c128 pattern. Run132 8-rank legal call probe covered wkv but no new shared-input pair. Run133 source audit shows the apparent split/indexer repeats already share quantization; no semantics-valid projection fusion candidate currently warrants another 8-rank service run. Shift to HC/clone/cache source and trace attribution; quant family remains later candidate if a concrete legal replacement appears. |
| `loop-041` | `PIVOTED` | `PIVOTED` | Run134 HC/copy census and Run135 cache scatter map reveal no source-backed semantics-safe >=2ms edit. 126 cache scatters match SWA/compressed/indexer distinct writes by layer; clone device attribution is inconclusive. The larger remaining exposed HCCL/phase skew needs causal analysis before implementation. |
| `loop-042` | `PIVOTED` | `PIVOTED` | Run137 host phase skew narrows during target and reappears after proposer, but latest-rank cadence ~54ms and no safe scheduler edit show skew alone is not a removable critical-path claim. Run140 identifies a concrete CPU-heavy eager DSpark path, directing next product-specific audit there. |
| `loop-043` | `PIVOTED` | `PIVOTED` | Run142 legal eight-rank segment timing shows the smallest safe replay boundary (Markov tail) has only ~0.695ms device interval; its 2.655ms Host work appears largely overlapped with target. Whole DSpark graph requires dynamic KV/metadata semantics and remains unproven. No high-value exposed DSpark replay candidate currently justifies implementation versus the ~46.6ms target stage. |
| `loop-044` | `EVALUATING` | `PENDING` | 审查 Run run148 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
