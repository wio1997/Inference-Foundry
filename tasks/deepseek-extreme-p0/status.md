# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-042`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`12`
- 下一步：Run136: reconstruct per-rank absolute prepare/target/proposer event timing and first-collective intervals from Run106/107 and Run98; distinguish fixed offset from cycle growth and identify the latest rank.
- 更新时间：`2026-09-25T01:28:26Z`

## 最近 Loop

共 `42` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-033` | `ACCEPTED` | `ACCEPTED` | Matched 64-cycle real-weight runs establish both correctness and causality. Reusing the fixed target graph from Extreme-owned buffers preserves exact state/mirror/rank invariants and cuts cycle wall 87.71%, raising internal decode-window throughput 669.2%, with no ModelRunner or Scheduler retained in the cycle. |
| `loop-034` | `PIVOTED` | `PIVOTED` | The frozen milestone is satisfied: CPU gates pass, every official run completes 48/48 requests at exactly 1024 tokens, and 128 rank/cohort records pass with no per-cycle ModelRunner/Scheduler re-entry. TaskCtl cannot mark the mixed-history loop accepted because the preserved collision attempt is invalid; that attempt is environmental, while the formal run is valid and shows Extreme at 217.342 tok/s, 60.022% below Stock. |
| `loop-035` | `PIVOTED` | `PIVOTED` | Frozen Loop035 technical goal is met: causal DSpark gid2 slot refresh fixes sustained acceptance; Run85/72/84/90/92 bounded correctness and Run87 DAG profile pass; formal 48x32K-to-1024 c12 Extreme median improves 217.342 to 525.417 tok/s (+141.747%), 128/128 rank/cohort rows pass. TaskCtl cannot mark accepted because preserved invalid diagnostic attempts (including Run89 OOM and Run91 unreachable fixed Stock shape) remain in this Loop; those attempts do not invalidate the official benchmark. P0 above-Stock target remains open. |
| `loop-036` | `PIVOTED` | `PIVOTED` | The frozen technical goal passes: legal 8-rank 299-cycle stable metadata header shadow and full serving gates, same-contract metadata stage 8.6730 to 0.6674 ms, and formal 48-request median 525.417 to 571.681 tok/s (+8.805%) above Stock by 5.155%. Preserved Run94/95 invalid setup and Run96 full-buffer failure prevent treating every mixed-history Run as pass; full AICPU tails have dynamic self-replay noise. Retain static implementation and pivot to target stage attribution. |
| `loop-037` | `PIVOTED` | `PIVOTED` | Runs100-104 passed legal serving but HTTP profiler RPC activation/stop latency varied from immediate to 23 seconds, missing decode or generating 10+ GiB traces; no bounded eight-rank target attribution. The profiler control boundary must move into the Runtime cycle. |
| `loop-038` | `PIVOTED` | `PIVOTED` | Cycle-scheduled profiling met the technical attribution goal in Run107: legal 8-rank cohort and 15 canonical synchronized target windows isolate 2836 kernels/cycle and quantify compute, communication, overlap, and a 9.966 ms grouped-matmul family. Run105 invalid launcher remains preserved, so the mixed-history Loop is pivoted; forced synchronization makes performance diagnostic only. |
| `loop-039` | `PIVOTED` | `PIVOTED` | Current fused W4A8 GMM1 already skips empty experts; one-card synthetic route test gives no graph-level >=5ms candidate. Run107 GMM sums are diagnostic, not removable benefit. Highest next discriminating work is non-GMM quant matmul call attribution; GMM and communication remain open. |
| `loop-040` | `PIVOTED` | `PIVOTED` | Run131 mapped 236 quant kernels/target and alternating c4/c128 pattern. Run132 8-rank legal call probe covered wkv but no new shared-input pair. Run133 source audit shows the apparent split/indexer repeats already share quantization; no semantics-valid projection fusion candidate currently warrants another 8-rank service run. Shift to HC/clone/cache source and trace attribution; quant family remains later candidate if a concrete legal replacement appears. |
| `loop-041` | `PIVOTED` | `PIVOTED` | Run134 HC/copy census and Run135 cache scatter map reveal no source-backed semantics-safe >=2ms edit. 126 cache scatters match SWA/compressed/indexer distinct writes by layer; clone device attribution is inconclusive. The larger remaining exposed HCCL/phase skew needs causal analysis before implementation. |
| `loop-042` | `EVALUATING` | `PENDING` | Run139: bounded legal 8-rank proposer subphase capture with paired host wall/thread CPU timestamps around mirror commit, prepare, pack and borrowed _propose set_inputs/metadata/runnable; no device barrier; restore patches and stop service. Compare latest-rank steady cadence and acceptance. |

## 阻塞项

- 暂无
