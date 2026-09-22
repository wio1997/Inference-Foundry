# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`INTEGRATING`
- 活动 Loop：`loop-031`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`4`
- 下一步：Add low-overhead stage markers around the Extreme-owned chain and capture an eight-rank runtime-only trace; attribute target, TP/EP communication, acceptance/state advance, DSpark common refresh and proposer regions before selecting an implementation change.
- 更新时间：`2026-09-22T08:23:32Z`

## 最近 Loop

共 `31` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-022` | `REJECTED` | `REJECTED` | Exact stream neighbors falsify a material main-path MoE wait. The two 53-54 ms outer waits gate MEMCPY_ASYNC on auxiliary streams 42/43. The apparent 25.734 ms MoE wait is shared stream36 waiting at the next layer before dynamic quant; default stream47 waits for shared output are about 0.00002 ms. Intra-call synchronization medians are 0.191, 0.051 and 0.011 ms, below baseline noise. |
| `loop-023` | `REJECTED` | `REJECTED` | k5 is not a runnable TP8 configuration in the current vLLM 0.26 path: graph shapes must be divisible by both 6 and 8. The service failed during KV/backend initialization, so there is no correctness or performance comparison. With model dspark_block_size>=5, the next smaller valid k below7 does not exist under this invariant. |
| `loop-024` | `REJECTED` | `REJECTED` | The capacity change is functional and removes the8096 warning, but the bounded c12 screen is within prior k7 variability:472.871 tok/s, +2.77% vs diagnostic median and -3.83% vs closest same-runner Loop020 control. It does not exceed the4.3% promotion threshold, so avoid an expensive full benchmark. |
| `loop-025` | `ACCEPTED` | `ACCEPTED` | Matched no-spec control proves k7 DSpark is architecturally valuable:491.698 versus208.047 tok/s (2.363x) and17.554 versus54.333ms TPOT (-67.69%). The difference dwarfs4.3% noise. Retain DSpark; optimize proposer necessary compute/acceptance rather than exit speculative decoding. |
| `loop-026` | `REJECTED` | `REJECTED` | Removing one of three trained draft layers saves23.21% proposer model time but destroys proposal quality: advanced tokens fall to1.363/cycle, far below3.15 break-even; output TPS drops55.85% to217.092, close to target-only208.047. Full three-layer semantics are necessary. |
| `loop-027` | `PIVOTED` | `PIVOTED` | Legacy DSpark is hard-disabled from graph; the available v2 DSpark graph path fails before capture because generic KV-group discovery finds no draft attention group for this DeepSeek V4 checkpoint. Stock v2 integration is therefore not an immediately runnable graph solution. Use the working legacy path as semantic/operator oracle and extract the fixed proposer-target execution contract for a specialized runtime. |
| `loop-028` | `PIVOTED` | `PIVOTED` | Loop028 established a fixed-shape c12 proposer contract and an executable exact in-process replay boundary on 8/8 ranks, but the frozen success gate required target verification, accepted-token parity and complete mutated-state comparison. Those boundaries were not captured, so the full standalone fixed-cycle claim is not yet supported. |
| `loop-029` | `PIVOTED` | `PIVOTED` | The product milestone is satisfied by run17, but this implementation loop intentionally preserves earlier failed diagnostic runs for cache layout and replay hypotheses; TaskCtl therefore cannot label the mixed-history loop accepted. Run17 completed eight real-weight Extreme-owned cycles on all eight ranks before generic ModelRunner target forward, with exact state advance and identical rank state. |
| `loop-030` | `ACCEPTED` | `ACCEPTED` | Read-only certification confirms all eight run17 ranks passed eight real-weight runtime-owned cycles with pre-ModelRunner handoff, zero post-handoff oracle target calls, no retained ModelRunner, 67 caches, exact state advance and identical rank state. |
| `loop-031` | `FROZEN` | `PENDING` | Add low-overhead stage markers around the Extreme-owned chain and capture an eight-rank runtime-only trace; attribute target, TP/EP communication, acceptance/state advance, DSpark common refresh and proposer regions before selecting an implementation change. |

## 阻塞项

- 暂无
