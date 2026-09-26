# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`DESIGNING`
- 活动 Loop：`loop-072`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`20`
- 下一步：Run319 source-only gate for layer4 MoE row independence, selected EP dispatcher/Graph shape and all8 zero-row hazards; then implement reversible private A/A/B/A fixture if viable
- 更新时间：`2026-09-26T15:58:09Z`

## 最近 Loop

共 `72` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-063` | `PIVOTED` | `PIVOTED` | Run275 candidate-consumed continuous correctness/alias passed, but no-verifier B versus A0 latest-rank runtime was +1.472ms/cycle slower across all four cohorts; client TPS difference tracked -6.233s residual, not exposed decode saving. Run278 launcher exit127 after active-script edit makes comparison diagnostic-only. No formal baseline promotion. |
| `loop-064` | `PIVOTED` | `PIVOTED` | Run284 local CP overlap advanced matched join but Run285 numerical gate is inconclusive; Run287 original FULL Graph confirms 16 owner full update rows per rank over 11968 rank-cycles yet native/lifetime consumers remain open. Astra High independently selects H003 hidden AllGather and local Q overlap as shorter-closure next scheduling test. No E2E gain or numeric ceiling. |
| `loop-065` | `PIVOTED` | `PIVOTED` | H003 delayed hidden gather overlaps local Q on all8 and advances one-layer next AllToAll by paired median10.125us versus async-immediate; full Target/cycle and Product gain unproven, same-state typed parity pending. Run287 owner and parking evidence suggests larger architectural gap |
| `loop-066` | `PIVOTED` | `PIVOTED` | Run296 typed consumer semantics pass, Run297 eager no stable gain, Run298 private Graph small gain below control drift; owner16 immediate method remains valid locally but complete producer/lifetime and Product gain unproved |
| `loop-067` | `PIVOTED` | `PIVOTED` | Sampled storage liveness census plus source-bound state window removes concrete cross-layer RAW obstacle; persistent full-producer correctness and wall benefit still unproved |
| `loop-068` | `PIVOTED` | `PIVOTED` | Private value and Graph cost gates pass; persistent live Target semantics and exposed multi-rank cycle effect require reversible one-layer integration |
| `loop-069` | `PIVOTED` | `PIVOTED` | Run311 integrates owner16 full producer in all8 FULL Graph Runtime ranks and completes 12x1024, but content differs cross-service; Run310 vs unchanged original Run312 also differs in all12 reasoning hashes and cycles315 vs290. Candidate correctness and wall benefit are not identifiable. Run311 wall/cycle 56.810ms exceeds original 56.545/56.643ms; no Product gain promoted. Next isolate dynamic metadata inside Graph on same prestate. |
| `loop-070` | `PIVOTED` | `PIVOTED` | Real-entry private Graph parity 8/8; synthetic shifts create nonfinite Sparse in original A, so no further synthetic repeats or Product claim |
| `loop-071` | `PIVOTED` | `PIVOTED` | Run315-317 show complete two-cycle restore spans Target metadata, Draft/Host and distinct Graphs; owner16 local 24-32us/layer is a small conditional screen, while Run318 original FULL Graph has 2989 parked slot-cycles across 1496 cycles. Astra High independently prioritizes complete MoE parked-row experiment. Owner remains unresolved, not rejected. |
| `loop-072` | `EVALUATING` | `PENDING` | 审查 Run run323 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
