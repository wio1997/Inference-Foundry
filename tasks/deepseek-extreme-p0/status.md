# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`NONE`
- 已接受基线：`NONE`
- 证据成熟度：`E2_BENCHMARKED`
- 用例数：`3`
- 当前知识条目：`1`
- 下一步：Investigate the warm mixed decode critical path and find an intervention with measurable output TPS gain; retain prefill-only QLI patch.
- 更新时间：`2026-09-20T19:47:44Z`

## 最近 Loop

共 `12` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-003` | `PIVOTED` | `PIVOTED` | TP0 msprof distinguishes cold and warm compute/HCCL; warm HCCL union 1.165s of 3.401s with ~0.054s compute overlap, justifying a falsifiable FlashComm1 A/B; no same-condition optimization comparison yet |
| `loop-004` | `PIVOTED` | `PIVOTED` | FlashComm1-off alone violates vllm-ascend DSA CP requires SP constraint; config failed during worker init before performance measurement |
| `loop-005` | `PIVOTED` | `PIVOTED` | Strict golden4 exact-output gate is invalid: 4/4 mismatch even between repeat runs on unchanged candidate; cannot infer candidate correctness failure or performance |
| `loop-006` | `REJECTED` | `REJECTED` | Coupled SP/DSA-CP-off path has no robust E2E benefit: median output TPS -1.61% vs baseline and inside observed noise; TTFT median worsened ~7.8%, TPOT ~0.7%. Functional check passed but numerical equivalence remains unproven. |
| `loop-007` | `REJECTED` | `REJECTED` | DSA CP off alone regressed full warmed E2E performance: median output TPS -4.70%, TTFT +15.1%, TPOT +3.8%; no >5% gain. Functional gate passed; numerical equivalence not needed for rejected candidate. |
| `loop-008` | `ACCEPTED` | `ACCEPTED` | Short TP0 torch-NPU profile separates warm and cold CPU scopes and device kernels, locating 0.436s nested aten::item within warm prepare input and 1.379s rank-local device inactivity; source and exposure remain unresolved, but the diagnostic design goal is satisfied. |
| `loop-009` | `PIVOTED` | `PIVOTED` | with_modules stack profiler crashed all workers during stop_profile; offline parser warned of lost data and exported no FRAMEWORK operator events, so item callsites remain unidentified |
| `loop-010` | `ACCEPTED` | `ACCEPTED` | Targeted wrapper found exact DSA CP QLI NPU item callsite and matching CPU local maxima already available; warm/cold rank timing supports a falsifiable candidate, without claiming E2E gain |
| `loop-011` | `PIVOTED` | `PIVOTED` | All-step QLI CPU-max candidate passed >=192 per-rank parity checks and improved exact same cold prompts 8/8 by mean284.83ms (-10.81%), but paired mixed TPS +0.52% is noise and median mean TTFT worsened11.9%; refine to prefill-only rather than KEEP |
| `loop-012` | `ACCEPTED` | `ACCEPTED` | Runtime CPU/NPU QLI maxima parity passed on all eight ranks in Loop011. Prefill-only patch passed functional gate. Same 16 cold prompts all improved: 2626.03 to 2362.89 ms mean TTFT (-10.02%) with no prefix cache hits. Full mixed median output TPS 547.55 versus paired original 537.60 is within baseline noise; median TTFT 1143.58 versus 1262.74 ms shows no observed regression. |

## 阻塞项

- 暂无
