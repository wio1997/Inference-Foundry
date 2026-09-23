# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-035`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`10`
- 下一步：Compute per-slot and per-cohort acceptance from Loop034 rank records, compare to frozen Stock counters, then obtain bounded runtime-only stage profile.
- 更新时间：`2026-09-23T03:44:19Z`

## 最近 Loop

共 `35` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-026` | `REJECTED` | `REJECTED` | Removing one of three trained draft layers saves23.21% proposer model time but destroys proposal quality: advanced tokens fall to1.363/cycle, far below3.15 break-even; output TPS drops55.85% to217.092, close to target-only208.047. Full three-layer semantics are necessary. |
| `loop-027` | `PIVOTED` | `PIVOTED` | Legacy DSpark is hard-disabled from graph; the available v2 DSpark graph path fails before capture because generic KV-group discovery finds no draft attention group for this DeepSeek V4 checkpoint. Stock v2 integration is therefore not an immediately runnable graph solution. Use the working legacy path as semantic/operator oracle and extract the fixed proposer-target execution contract for a specialized runtime. |
| `loop-028` | `PIVOTED` | `PIVOTED` | Loop028 established a fixed-shape c12 proposer contract and an executable exact in-process replay boundary on 8/8 ranks, but the frozen success gate required target verification, accepted-token parity and complete mutated-state comparison. Those boundaries were not captured, so the full standalone fixed-cycle claim is not yet supported. |
| `loop-029` | `PIVOTED` | `PIVOTED` | The product milestone is satisfied by run17, but this implementation loop intentionally preserves earlier failed diagnostic runs for cache layout and replay hypotheses; TaskCtl therefore cannot label the mixed-history loop accepted. Run17 completed eight real-weight Extreme-owned cycles on all eight ranks before generic ModelRunner target forward, with exact state advance and identical rank state. |
| `loop-030` | `ACCEPTED` | `ACCEPTED` | Read-only certification confirms all eight run17 ranks passed eight real-weight runtime-owned cycles with pre-ModelRunner handoff, zero post-handoff oracle target calls, no retained ModelRunner, 67 caches, exact state advance and identical rank state. |
| `loop-031` | `ACCEPTED` | `ACCEPTED` | All eight ranks produced parsed runtime-only traces with correctness preserved. Scope and device-union analysis identifies the target/proposer critical path and isolates DSpark CPU mirror refresh as a removable synchronization residue, satisfying the frozen attribution and candidate-selection goal. |
| `loop-032` | `PIVOTED` | `PIVOTED` | The structural hypothesis is supported: real-weight correctness holds for 64 cycles, all runtime-owned host mirrors equal device state, and the matched profile shows the DSpark refresh barrier falling by 12.117 ms median with a 7.495 ms (-1.40%) full-cycle median reduction. TaskCtl cannot register an accepted optimization verdict because the already-recorded Loop031 profile Run omitted a metric field; the evidence comparison remains preserved explicitly and the implementation is retained. |
| `loop-033` | `ACCEPTED` | `ACCEPTED` | Matched 64-cycle real-weight runs establish both correctness and causality. Reusing the fixed target graph from Extreme-owned buffers preserves exact state/mirror/rank invariants and cuts cycle wall 87.71%, raising internal decode-window throughput 669.2%, with no ModelRunner or Scheduler retained in the cycle. |
| `loop-034` | `PIVOTED` | `PIVOTED` | The frozen milestone is satisfied: CPU gates pass, every official run completes 48/48 requests at exactly 1024 tokens, and 128 rank/cohort records pass with no per-cycle ModelRunner/Scheduler re-entry. TaskCtl cannot mark the mixed-history loop accepted because the preserved collision attempt is invalid; that attempt is environmental, while the formal run is valid and shows Extreme at 217.342 tok/s, 60.022% below Stock. |
| `loop-035` | `EVALUATING` | `PENDING` | 审查 Run run-20260923T180026Z 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
