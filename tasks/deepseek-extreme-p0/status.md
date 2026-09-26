# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`DESIGNING`
- 活动 Loop：`loop-074`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`20`
- 下一步：Validate Run287 original all8 emitted/mask and client/runtime cohort mapping, compute no-cost FIFO and incremental refill-cost sensitivity, record historical R06/R36/R37 priors with changed TP8 conditions
- 更新时间：`2026-09-26T18:44:08Z`

## 最近 Loop

共 `74` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-065` | `PIVOTED` | `PIVOTED` | H003 delayed hidden gather overlaps local Q on all8 and advances one-layer next AllToAll by paired median10.125us versus async-immediate; full Target/cycle and Product gain unproven, same-state typed parity pending. Run287 owner and parking evidence suggests larger architectural gap |
| `loop-066` | `PIVOTED` | `PIVOTED` | Run296 typed consumer semantics pass, Run297 eager no stable gain, Run298 private Graph small gain below control drift; owner16 immediate method remains valid locally but complete producer/lifetime and Product gain unproved |
| `loop-067` | `PIVOTED` | `PIVOTED` | Sampled storage liveness census plus source-bound state window removes concrete cross-layer RAW obstacle; persistent full-producer correctness and wall benefit still unproved |
| `loop-068` | `PIVOTED` | `PIVOTED` | Private value and Graph cost gates pass; persistent live Target semantics and exposed multi-rank cycle effect require reversible one-layer integration |
| `loop-069` | `PIVOTED` | `PIVOTED` | Run311 integrates owner16 full producer in all8 FULL Graph Runtime ranks and completes 12x1024, but content differs cross-service; Run310 vs unchanged original Run312 also differs in all12 reasoning hashes and cycles315 vs290. Candidate correctness and wall benefit are not identifiable. Run311 wall/cycle 56.810ms exceeds original 56.545/56.643ms; no Product gain promoted. Next isolate dynamic metadata inside Graph on same prestate. |
| `loop-070` | `PIVOTED` | `PIVOTED` | Real-entry private Graph parity 8/8; synthetic shifts create nonfinite Sparse in original A, so no further synthetic repeats or Product claim |
| `loop-071` | `PIVOTED` | `PIVOTED` | Run315-317 show complete two-cycle restore spans Target metadata, Draft/Host and distinct Graphs; owner16 local 24-32us/layer is a small conditional screen, while Run318 original FULL Graph has 2989 parked slot-cycles across 1496 cycles. Astra High independently prioritizes complete MoE parked-row experiment. Owner remains unresolved, not rejected. |
| `loop-072` | `PIVOTED` | `PIVOTED` | Run326 all8 same-prestate complete MoE Graph A/A_repeat/B/A2 showed routed48 B slower than independent A controls in 10/10 paired max-rank endpoints, median +49.585us; numerical control envelope failed on 6/7 nonempty ranks. Reject this implementation, not all parked schedules or a hardware bound. |
| `loop-073` | `PIVOTED` | `PIVOTED` | Run328 all8 same-state full-active layer4 Graph proves replicated gate weights and exact routing IDs, but B gate placement shows 0/10 strict complete-MoE wins and variable paired B-minus-A time; one rank narrowly misses output control envelope. Do not integrate this B schedule; exchange law and alternate overlap remain open. |
| `loop-074` | `EVALUATING` | `PENDING` | 审查 Run run344 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
