# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`OPTIMIZING`
- 活动 Loop：`loop-066`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`17`
- 下一步：Inspect actual native CompressorMetadata/Compressor state and scatter writes at layer2; build private same-entry A/A and full96 versus owner16 fixture
- 更新时间：`2026-09-26T09:23:56Z`

## 最近 Loop

共 `66` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-057` | `PIVOTED` | `PIVOTED` | Run233 captured surrogate at synthetic96 rows saved0.45us/call; Run235 real-local12-row surrogate fails FP32 router-output numerical gate before timing; official MRV2 fused op is unregistered in installed runtime. Read-only Run234 source/census found no defensible >=1ms/cycle target replacement; task sums and shared graph pool are not product savings. |
| `loop-058` | `PIVOTED` | `PIVOTED` | Executable V0 replay closes Run99 formal samples exactly and preserves trajectory; TP8 chain and wave audit constrain communication capacity and locate E2E variability, but scenario savings and true hardware bound remain unvalidated |
| `loop-059` | `PIVOTED` | `PIVOTED` | Legal all-rank boundary pass localizes Run99 residual to variable prefill and decode work rather than small publication/admission edges; Run188 shows phase reductions couple to decode cycles, and no true hardware floor or E2E candidate has been established |
| `loop-060` | `PIVOTED` | `PIVOTED` | Real eight-rank graph HBM task counters and exact shapes are measured, but unique compulsory traffic, HCCL link bytes, attainable graph bandwidth and dependency-critical exposure remain unproved; no correctness-preserving same-shape candidate has passed a formal E2E intervention. The frozen-product hardware-attainable throughput bound is not identifiable from these measurements. |
| `loop-061` | `PIVOTED` | `PIVOTED` | Eight-rank graph HCCL operation payload and same-host official HCCL Test capability are measured, but trace Size(Byte) is not physical link bytes, standalone test is not product Graph path, and critical-path overlap remains unknown; Astra High confirms V0 future ranges are sensitivities rather than attainable bounds. No E2E candidate or true hardware ceiling established. |
| `loop-062` | `PIVOTED` | `PIVOTED` | Causal DSpark-unused MTP stash traffic/collective removal passed correctness, but same-host repeated formal median advantage is confounded; normalized runtime per cycle has mixed signs. Numeric resource and scheduling bounds remain unknown. Pivot to complete execution dependency scheduling rather than promoting an unproved TPS gain. |
| `loop-063` | `PIVOTED` | `PIVOTED` | Run275 candidate-consumed continuous correctness/alias passed, but no-verifier B versus A0 latest-rank runtime was +1.472ms/cycle slower across all four cohorts; client TPS difference tracked -6.233s residual, not exposed decode saving. Run278 launcher exit127 after active-script edit makes comparison diagnostic-only. No formal baseline promotion. |
| `loop-064` | `PIVOTED` | `PIVOTED` | Run284 local CP overlap advanced matched join but Run285 numerical gate is inconclusive; Run287 original FULL Graph confirms 16 owner full update rows per rank over 11968 rank-cycles yet native/lifetime consumers remain open. Astra High independently selects H003 hidden AllGather and local Q overlap as shorter-closure next scheduling test. No E2E gain or numeric ceiling. |
| `loop-065` | `PIVOTED` | `PIVOTED` | H003 delayed hidden gather overlaps local Q on all8 and advances one-layer next AllToAll by paired median10.125us versus async-immediate; full Target/cycle and Product gain unproven, same-state typed parity pending. Run287 owner and parking evidence suggests larger architectural gap |
| `loop-066` | `EVALUATING` | `PENDING` | 审查 Run run298 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
