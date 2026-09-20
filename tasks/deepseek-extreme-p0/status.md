# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-014`
- 已接受基线：`NONE`
- 证据成熟度：`E2_BENCHMARKED`
- 用例数：`3`
- 当前知识条目：`1`
- 下一步：Inspect model_runner_v1.py prepare-input call chain and trace operator self-times; locate one repeated metadata operation with a measurable upper bound before editing.
- 更新时间：`2026-09-20T20:28:19Z`

## 最近 Loop

共 `14` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-005` | `PIVOTED` | `PIVOTED` | Strict golden4 exact-output gate is invalid: 4/4 mismatch even between repeat runs on unchanged candidate; cannot infer candidate correctness failure or performance |
| `loop-006` | `REJECTED` | `REJECTED` | Coupled SP/DSA-CP-off path has no robust E2E benefit: median output TPS -1.61% vs baseline and inside observed noise; TTFT median worsened ~7.8%, TPOT ~0.7%. Functional check passed but numerical equivalence remains unproven. |
| `loop-007` | `REJECTED` | `REJECTED` | DSA CP off alone regressed full warmed E2E performance: median output TPS -4.70%, TTFT +15.1%, TPOT +3.8%; no >5% gain. Functional gate passed; numerical equivalence not needed for rejected candidate. |
| `loop-008` | `ACCEPTED` | `ACCEPTED` | Short TP0 torch-NPU profile separates warm and cold CPU scopes and device kernels, locating 0.436s nested aten::item within warm prepare input and 1.379s rank-local device inactivity; source and exposure remain unresolved, but the diagnostic design goal is satisfied. |
| `loop-009` | `PIVOTED` | `PIVOTED` | with_modules stack profiler crashed all workers during stop_profile; offline parser warned of lost data and exported no FRAMEWORK operator events, so item callsites remain unidentified |
| `loop-010` | `ACCEPTED` | `ACCEPTED` | Targeted wrapper found exact DSA CP QLI NPU item callsite and matching CPU local maxima already available; warm/cold rank timing supports a falsifiable candidate, without claiming E2E gain |
| `loop-011` | `PIVOTED` | `PIVOTED` | All-step QLI CPU-max candidate passed >=192 per-rank parity checks and improved exact same cold prompts 8/8 by mean284.83ms (-10.81%), but paired mixed TPS +0.52% is noise and median mean TTFT worsened11.9%; refine to prefill-only rather than KEEP |
| `loop-012` | `ACCEPTED` | `ACCEPTED` | Runtime CPU/NPU QLI maxima parity passed on all eight ranks in Loop011. Prefill-only patch passed functional gate. Same 16 cold prompts all improved: 2626.03 to 2362.89 ms mean TTFT (-10.02%) with no prefix cache hits. Full mixed median output TPS 547.55 versus paired original 537.60 is within baseline noise; median TTFT 1143.58 versus 1262.74 ms shows no observed regression. |
| `loop-013` | `PIVOTED` | `PIVOTED` | Valid c12 trace isolates large draft host and TP8 HCCL/idle envelopes but does not prove either removable. The candidate DSpark draft graph path is hard-disabled in source; prior upstream graph attempt was reverted. Further source-level cause and numerical constraints are needed before an E2E optimization. |
| `loop-014` | `RUNNING` | `PENDING` | 执行 Run attention-builder-trace-20260920：Same no-profiler functional+48x128 warmup+12x512 c12 sample, with per-builder stage timers |

## 阻塞项

- 暂无
