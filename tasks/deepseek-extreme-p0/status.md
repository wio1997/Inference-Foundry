# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`DESIGNING`
- 活动 Loop：`NONE`
- 已接受基线：`NONE`
- 证据成熟度：`E2_BENCHMARKED`
- 用例数：`3`
- 当前知识条目：`3`
- 下一步：PAUSED by user after Loop028 commit. On resume, extend the proven proposer boundary through target verification, device-side acceptance and explicit state/KV mutation parity before claiming a standalone full cycle.
- 更新时间：`2026-09-21T12:51:38Z`

## 最近 Loop

共 `28` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-019` | `PIVOTED` | `PIVOTED` | QLI first NPU scalar read accounts for most measured first-builder host time, but the already-tested all-step CPU-max removal passed parity and gave no robust mixed TPS gain (+0.52% within noise, TTFT worse). Metadata op itself is only ~0.4ms/call. No new semantics-safe, high-value decode patch justified; host span is not a removable E2E bound. |
| `loop-020` | `PIVOTED` | `PIVOTED` | TP0 trace flows causally join all 121038 async launches whose CPU origin is inside a draft_token scope to device X tasks at exact timestamps. Across 170 scopes, device-task union clipped to the host scope is median 51.661 ms versus 52.894 ms host scope, so proposer time is predominantly device-active rather than a removable host-only bubble. Draft acceptance advances only 3.54-3.64 tokens for 7 drafted tokens, but no safe material scheduling intervention is identified; DSpark ACLGraph is explicitly unsupported/eager. |
| `loop-021` | `REJECTED` | `REJECTED` | Exact flow attribution falsifies the proposed Index/Pad/Copy chain as a material standalone target: aclnnIndex clipped device union is 0.545 ms median when present and ConstantPadNd 0.337 ms, both below the 4.3 percent frozen TPS spread. The dominant apparent span is EVENT_WAIT, chiefly 51.169 ms at outer draft scope and 26.595 ms under moe_forward_shared; event waits are stream dependencies and cannot be counted as compute or removable time. |
| `loop-022` | `REJECTED` | `REJECTED` | Exact stream neighbors falsify a material main-path MoE wait. The two 53-54 ms outer waits gate MEMCPY_ASYNC on auxiliary streams 42/43. The apparent 25.734 ms MoE wait is shared stream36 waiting at the next layer before dynamic quant; default stream47 waits for shared output are about 0.00002 ms. Intra-call synchronization medians are 0.191, 0.051 and 0.011 ms, below baseline noise. |
| `loop-023` | `REJECTED` | `REJECTED` | k5 is not a runnable TP8 configuration in the current vLLM 0.26 path: graph shapes must be divisible by both 6 and 8. The service failed during KV/backend initialization, so there is no correctness or performance comparison. With model dspark_block_size>=5, the next smaller valid k below7 does not exist under this invariant. |
| `loop-024` | `REJECTED` | `REJECTED` | The capacity change is functional and removes the8096 warning, but the bounded c12 screen is within prior k7 variability:472.871 tok/s, +2.77% vs diagnostic median and -3.83% vs closest same-runner Loop020 control. It does not exceed the4.3% promotion threshold, so avoid an expensive full benchmark. |
| `loop-025` | `ACCEPTED` | `ACCEPTED` | Matched no-spec control proves k7 DSpark is architecturally valuable:491.698 versus208.047 tok/s (2.363x) and17.554 versus54.333ms TPOT (-67.69%). The difference dwarfs4.3% noise. Retain DSpark; optimize proposer necessary compute/acceptance rather than exit speculative decoding. |
| `loop-026` | `REJECTED` | `REJECTED` | Removing one of three trained draft layers saves23.21% proposer model time but destroys proposal quality: advanced tokens fall to1.363/cycle, far below3.15 break-even; output TPS drops55.85% to217.092, close to target-only208.047. Full three-layer semantics are necessary. |
| `loop-027` | `PIVOTED` | `PIVOTED` | Legacy DSpark is hard-disabled from graph; the available v2 DSpark graph path fails before capture because generic KV-group discovery finds no draft attention group for this DeepSeek V4 checkpoint. Stock v2 integration is therefore not an immediately runnable graph solution. Use the working legacy path as semantic/operator oracle and extract the fixed proposer-target execution contract for a specialized runtime. |
| `loop-028` | `PIVOTED` | `PIVOTED` | Loop028 established a fixed-shape c12 proposer contract and an executable exact in-process replay boundary on 8/8 ranks, but the frozen success gate required target verification, accepted-token parity and complete mutated-state comparison. Those boundaries were not captured, so the full standalone fixed-cycle claim is not yet supported. |

## 阻塞项

- 暂无
