# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-015`
- 已接受基线：`NONE`
- 证据成熟度：`E2_BENCHMARKED`
- 用例数：`3`
- 当前知识条目：`1`
- 下一步：Revisit retained cold TP0 trace and per-request TTFT alignment, inspect ScatterNdUpdateSk and Compressor call sites, then select one falsifiable prefill intervention.
- 更新时间：`2026-09-20T21:05:37Z`

## 最近 Loop

共 `15` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-006` | `REJECTED` | `REJECTED` | Coupled SP/DSA-CP-off path has no robust E2E benefit: median output TPS -1.61% vs baseline and inside observed noise; TTFT median worsened ~7.8%, TPOT ~0.7%. Functional check passed but numerical equivalence remains unproven. |
| `loop-007` | `REJECTED` | `REJECTED` | DSA CP off alone regressed full warmed E2E performance: median output TPS -4.70%, TTFT +15.1%, TPOT +3.8%; no >5% gain. Functional gate passed; numerical equivalence not needed for rejected candidate. |
| `loop-008` | `ACCEPTED` | `ACCEPTED` | Short TP0 torch-NPU profile separates warm and cold CPU scopes and device kernels, locating 0.436s nested aten::item within warm prepare input and 1.379s rank-local device inactivity; source and exposure remain unresolved, but the diagnostic design goal is satisfied. |
| `loop-009` | `PIVOTED` | `PIVOTED` | with_modules stack profiler crashed all workers during stop_profile; offline parser warned of lost data and exported no FRAMEWORK operator events, so item callsites remain unidentified |
| `loop-010` | `ACCEPTED` | `ACCEPTED` | Targeted wrapper found exact DSA CP QLI NPU item callsite and matching CPU local maxima already available; warm/cold rank timing supports a falsifiable candidate, without claiming E2E gain |
| `loop-011` | `PIVOTED` | `PIVOTED` | All-step QLI CPU-max candidate passed >=192 per-rank parity checks and improved exact same cold prompts 8/8 by mean284.83ms (-10.81%), but paired mixed TPS +0.52% is noise and median mean TTFT worsened11.9%; refine to prefill-only rather than KEEP |
| `loop-012` | `ACCEPTED` | `ACCEPTED` | Runtime CPU/NPU QLI maxima parity passed on all eight ranks in Loop011. Prefill-only patch passed functional gate. Same 16 cold prompts all improved: 2626.03 to 2362.89 ms mean TTFT (-10.02%) with no prefix cache hits. Full mixed median output TPS 547.55 versus paired original 537.60 is within baseline noise; median TTFT 1143.58 versus 1262.74 ms shows no observed regression. |
| `loop-013` | `PIVOTED` | `PIVOTED` | Valid c12 trace isolates large draft host and TP8 HCCL/idle envelopes but does not prove either removable. The candidate DSpark draft graph path is hard-disabled in source; prior upstream graph attempt was reverted. Further source-level cause and numerical constraints are needed before an E2E optimization. |
| `loop-014` | `PIVOTED` | `PIVOTED` | No supported >=5% mixed TPS candidate emerged. The 8 metadata builders are group-specific and shared local/ratio state is already cached. First builder timing may include required device synchronization; Loop011 all-step QLI substitution did not improve mixed throughput. Avoid changing semantics merely to reduce inclusive host spans. |
| `loop-015` | `RUNNING` | `PENDING` | 执行 Run scatter-slot-uniqueness-20260920：Run gated one-shot capture of actual long-prefill dsa_kv_compress_scatter slot_mapping on all ranks; functional gate and cold offset24 32Kx128 c1 |

## 阻塞项

- 暂无
