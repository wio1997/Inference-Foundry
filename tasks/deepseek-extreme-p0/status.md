# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-038`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`11`
- 下一步：Implement opt-in cycle-scheduled profiling in ExtremeDecodeRuntime and launch one legal cohort.
- 更新时间：`2026-09-24T14:23:03Z`

## 最近 Loop

共 `38` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-029` | `PIVOTED` | `PIVOTED` | The product milestone is satisfied by run17, but this implementation loop intentionally preserves earlier failed diagnostic runs for cache layout and replay hypotheses; TaskCtl therefore cannot label the mixed-history loop accepted. Run17 completed eight real-weight Extreme-owned cycles on all eight ranks before generic ModelRunner target forward, with exact state advance and identical rank state. |
| `loop-030` | `ACCEPTED` | `ACCEPTED` | Read-only certification confirms all eight run17 ranks passed eight real-weight runtime-owned cycles with pre-ModelRunner handoff, zero post-handoff oracle target calls, no retained ModelRunner, 67 caches, exact state advance and identical rank state. |
| `loop-031` | `ACCEPTED` | `ACCEPTED` | All eight ranks produced parsed runtime-only traces with correctness preserved. Scope and device-union analysis identifies the target/proposer critical path and isolates DSpark CPU mirror refresh as a removable synchronization residue, satisfying the frozen attribution and candidate-selection goal. |
| `loop-032` | `PIVOTED` | `PIVOTED` | The structural hypothesis is supported: real-weight correctness holds for 64 cycles, all runtime-owned host mirrors equal device state, and the matched profile shows the DSpark refresh barrier falling by 12.117 ms median with a 7.495 ms (-1.40%) full-cycle median reduction. TaskCtl cannot register an accepted optimization verdict because the already-recorded Loop031 profile Run omitted a metric field; the evidence comparison remains preserved explicitly and the implementation is retained. |
| `loop-033` | `ACCEPTED` | `ACCEPTED` | Matched 64-cycle real-weight runs establish both correctness and causality. Reusing the fixed target graph from Extreme-owned buffers preserves exact state/mirror/rank invariants and cuts cycle wall 87.71%, raising internal decode-window throughput 669.2%, with no ModelRunner or Scheduler retained in the cycle. |
| `loop-034` | `PIVOTED` | `PIVOTED` | The frozen milestone is satisfied: CPU gates pass, every official run completes 48/48 requests at exactly 1024 tokens, and 128 rank/cohort records pass with no per-cycle ModelRunner/Scheduler re-entry. TaskCtl cannot mark the mixed-history loop accepted because the preserved collision attempt is invalid; that attempt is environmental, while the formal run is valid and shows Extreme at 217.342 tok/s, 60.022% below Stock. |
| `loop-035` | `PIVOTED` | `PIVOTED` | Frozen Loop035 technical goal is met: causal DSpark gid2 slot refresh fixes sustained acceptance; Run85/72/84/90/92 bounded correctness and Run87 DAG profile pass; formal 48x32K-to-1024 c12 Extreme median improves 217.342 to 525.417 tok/s (+141.747%), 128/128 rank/cohort rows pass. TaskCtl cannot mark accepted because preserved invalid diagnostic attempts (including Run89 OOM and Run91 unreachable fixed Stock shape) remain in this Loop; those attempts do not invalidate the official benchmark. P0 above-Stock target remains open. |
| `loop-036` | `PIVOTED` | `PIVOTED` | The frozen technical goal passes: legal 8-rank 299-cycle stable metadata header shadow and full serving gates, same-contract metadata stage 8.6730 to 0.6674 ms, and formal 48-request median 525.417 to 571.681 tok/s (+8.805%) above Stock by 5.155%. Preserved Run94/95 invalid setup and Run96 full-buffer failure prevent treating every mixed-history Run as pass; full AICPU tails have dynamic self-replay noise. Retain static implementation and pivot to target stage attribution. |
| `loop-037` | `PIVOTED` | `PIVOTED` | Runs100-104 passed legal serving but HTTP profiler RPC activation/stop latency varied from immediate to 23 seconds, missing decode or generating 10+ GiB traces; no bounded eight-rank target attribution. The profiler control boundary must move into the Runtime cycle. |
| `loop-038` | `RUNNING` | `PENDING` | 执行 Run run106：docker exec vllm-ascend26-dsv4f-w4a8 bash -lc /data/wio/Inference_Foundry/scripts/run_loop038_cycle_profile.sh |

## 阻塞项

- 暂无
