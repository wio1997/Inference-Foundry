# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-023`
- 已接受基线：`NONE`
- 证据成熟度：`E2_BENCHMARKED`
- 用例数：`3`
- 当前知识条目：`1`
- 下一步：Audit the frozen launch command and DSpark configuration path, then run a bounded no-profiler k5 screen against k7 with identical warmup and 12x32K-to-512 c12 requests before a full repeated benchmark.
- 更新时间：`2026-09-21T06:05:32Z`

## 最近 Loop

共 `23` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-014` | `PIVOTED` | `PIVOTED` | No supported >=5% mixed TPS candidate emerged. The 8 metadata builders are group-specific and shared local/ratio state is already cached. First builder timing may include required device synchronization; Loop011 all-step QLI substitution did not improve mixed throughput. Avoid changing semantics merely to reduce inclusive host spans. |
| `loop-015` | `PIVOTED` | `PIVOTED` | Captured one real 8096-row long-prefill scatter mapping per rank: all unique, but uniqueness not yet a general invariant. Existing V2 op gave bit-equal output and was 69% slower than SK in 25-call isolated NPU timing; no safe E2E patch or paired TTFT gain. Source SK deterministic sort+SyncAll suggests a direct unique-index path merits a separate build experiment. |
| `loop-016` | `REJECTED` | `REJECTED` | Captured-shape isolated kernel was faster, but actual DSA-CP fast path activated on eight ranks and same-prompt cold TTFT worsened by 70.19ms mean (+2.9642 percent), 0/8 pairs improved, zero prefix hits. No E2E gain. |
| `loop-017` | `REJECTED` | `REJECTED` | mBase256 isolated candidate built, but first comparable exact-shape screen never completed after >8 minutes versus stock ~1.514ms/call; runtime/integration/tiling cause cannot be distinguished. No E2E or correctness evidence; operational gate failed. |
| `loop-018` | `PIVOTED` | `PIVOTED` | Saved cross-rank timestamps are confounded by stable ~579us rank6 clock offset plus residual ~206us end spread. Communication elapsed is Idle-only, so collective arrival skew cannot be separated from clock/reporting artifacts or translated to a safe high-value change using current evidence. |
| `loop-019` | `PIVOTED` | `PIVOTED` | QLI first NPU scalar read accounts for most measured first-builder host time, but the already-tested all-step CPU-max removal passed parity and gave no robust mixed TPS gain (+0.52% within noise, TTFT worse). Metadata op itself is only ~0.4ms/call. No new semantics-safe, high-value decode patch justified; host span is not a removable E2E bound. |
| `loop-020` | `PIVOTED` | `PIVOTED` | TP0 trace flows causally join all 121038 async launches whose CPU origin is inside a draft_token scope to device X tasks at exact timestamps. Across 170 scopes, device-task union clipped to the host scope is median 51.661 ms versus 52.894 ms host scope, so proposer time is predominantly device-active rather than a removable host-only bubble. Draft acceptance advances only 3.54-3.64 tokens for 7 drafted tokens, but no safe material scheduling intervention is identified; DSpark ACLGraph is explicitly unsupported/eager. |
| `loop-021` | `REJECTED` | `REJECTED` | Exact flow attribution falsifies the proposed Index/Pad/Copy chain as a material standalone target: aclnnIndex clipped device union is 0.545 ms median when present and ConstantPadNd 0.337 ms, both below the 4.3 percent frozen TPS spread. The dominant apparent span is EVENT_WAIT, chiefly 51.169 ms at outer draft scope and 26.595 ms under moe_forward_shared; event waits are stream dependencies and cannot be counted as compute or removable time. |
| `loop-022` | `REJECTED` | `REJECTED` | Exact stream neighbors falsify a material main-path MoE wait. The two 53-54 ms outer waits gate MEMCPY_ASYNC on auxiliary streams 42/43. The apparent 25.734 ms MoE wait is shared stream36 waiting at the next layer before dynamic quant; default stream47 waits for shared output are about 0.00002 ms. Intra-call synchronization medians are 0.191, 0.051 and 0.011 ms, below baseline noise. |
| `loop-023` | `RUNNING` | `PENDING` | 执行 Run k5-screen-20260921：MAX_MODEL_LEN=1048576 RUN_TS=LOOP023-K5-20260921 bash scripts/serve_loop023_k5.sh; python3 scripts/run_loop014_prepare_trace.py --out evidence/20260921_loop023_k5_screen/run1 |

## 阻塞项

- 暂无
