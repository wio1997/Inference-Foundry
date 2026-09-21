# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-019`
- 已接受基线：`NONE`
- 证据成熟度：`E2_BENCHMARKED`
- 用例数：`3`
- 当前知识条目：`1`
- 下一步：Inspect Loop014 no-profiler builder traces and active DSA-CP builder source; quantify repeated allocations/work by shape before editing.
- 更新时间：`2026-09-21T01:13:57Z`

## 最近 Loop

共 `19` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-010` | `ACCEPTED` | `ACCEPTED` | Targeted wrapper found exact DSA CP QLI NPU item callsite and matching CPU local maxima already available; warm/cold rank timing supports a falsifiable candidate, without claiming E2E gain |
| `loop-011` | `PIVOTED` | `PIVOTED` | All-step QLI CPU-max candidate passed >=192 per-rank parity checks and improved exact same cold prompts 8/8 by mean284.83ms (-10.81%), but paired mixed TPS +0.52% is noise and median mean TTFT worsened11.9%; refine to prefill-only rather than KEEP |
| `loop-012` | `ACCEPTED` | `ACCEPTED` | Runtime CPU/NPU QLI maxima parity passed on all eight ranks in Loop011. Prefill-only patch passed functional gate. Same 16 cold prompts all improved: 2626.03 to 2362.89 ms mean TTFT (-10.02%) with no prefix cache hits. Full mixed median output TPS 547.55 versus paired original 537.60 is within baseline noise; median TTFT 1143.58 versus 1262.74 ms shows no observed regression. |
| `loop-013` | `PIVOTED` | `PIVOTED` | Valid c12 trace isolates large draft host and TP8 HCCL/idle envelopes but does not prove either removable. The candidate DSpark draft graph path is hard-disabled in source; prior upstream graph attempt was reverted. Further source-level cause and numerical constraints are needed before an E2E optimization. |
| `loop-014` | `PIVOTED` | `PIVOTED` | No supported >=5% mixed TPS candidate emerged. The 8 metadata builders are group-specific and shared local/ratio state is already cached. First builder timing may include required device synchronization; Loop011 all-step QLI substitution did not improve mixed throughput. Avoid changing semantics merely to reduce inclusive host spans. |
| `loop-015` | `PIVOTED` | `PIVOTED` | Captured one real 8096-row long-prefill scatter mapping per rank: all unique, but uniqueness not yet a general invariant. Existing V2 op gave bit-equal output and was 69% slower than SK in 25-call isolated NPU timing; no safe E2E patch or paired TTFT gain. Source SK deterministic sort+SyncAll suggests a direct unique-index path merits a separate build experiment. |
| `loop-016` | `REJECTED` | `REJECTED` | Captured-shape isolated kernel was faster, but actual DSA-CP fast path activated on eight ranks and same-prompt cold TTFT worsened by 70.19ms mean (+2.9642 percent), 0/8 pairs improved, zero prefix hits. No E2E gain. |
| `loop-017` | `REJECTED` | `REJECTED` | mBase256 isolated candidate built, but first comparable exact-shape screen never completed after >8 minutes versus stock ~1.514ms/call; runtime/integration/tiling cause cannot be distinguished. No E2E or correctness evidence; operational gate failed. |
| `loop-018` | `PIVOTED` | `PIVOTED` | Saved cross-rank timestamps are confounded by stable ~579us rank6 clock offset plus residual ~206us end spread. Communication elapsed is Idle-only, so collective arrival skew cannot be separated from clock/reporting artifacts or translated to a safe high-value change using current evidence. |
| `loop-019` | `EVALUATING` | `PENDING` | 审查 Run active-dsacp-builder-stage-20260921 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
