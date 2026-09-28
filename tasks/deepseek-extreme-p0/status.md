# Task 状态

- Task：`deepseek-extreme-p0`
- 标题：DeepSeek Extreme P0
- 算子：`deepseek_v4_flash_w4a8`
- 开发框架：`vllm-ascend`
- 状态：`ACTIVE`
- 阶段：`DESIGNING`
- 活动 Loop：`loop-081`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`3`
- 当前知识条目：`42`
- 下一步：Build reversible source-pinned branch and typed tensor descriptor capture for one selected FULL96 layer0 cut; preflight CPU negatives before live acquisition
- 更新时间：`2026-09-27T22:43:58Z`

## 最近 Loop

共 `81` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-072` | `PIVOTED` | `PIVOTED` | Run326 all8 same-prestate complete MoE Graph A/A_repeat/B/A2 showed routed48 B slower than independent A controls in 10/10 paired max-rank endpoints, median +49.585us; numerical control envelope failed on 6/7 nonempty ranks. Reject this implementation, not all parked schedules or a hardware bound. |
| `loop-073` | `PIVOTED` | `PIVOTED` | Run328 all8 same-state full-active layer4 Graph proves replicated gate weights and exact routing IDs, but B gate placement shows 0/10 strict complete-MoE wins and variable paired B-minus-A time; one rank narrowly misses output control envelope. Do not integrate this B schedule; exchange law and alternate overlap remain open. |
| `loop-074` | `PIVOTED` | `PIVOTED` | Run346/347 show 1118 FIFO and 1011-1015 capacity relaxations do not close necessary prefill/seed/resource and real-arrival costs; user prioritizes credible Bound interval. Run345 was never installed; keep legal-refill hypothesis open, pivot next measurement to all8 concurrent resource capacity and full work ledger. |
| `loop-075` | `PIVOTED` | `PIVOTED` | Conditional GMM2 bank sensitivity and native replay/counter/binding gates calibrated; compulsory work and same-path FULL Graph cache/resource join still missing, so no numerical Product ceiling |
| `loop-076` | `ACCEPTED` | `ACCEPTED` | Same-run Target route/native GMM counter attribution and corrected Draft device-tail work census are validated; timing and Product ceiling remain uncalibrated due hot-path snapshot sync and incomplete work/dependency inventory |
| `loop-077` | `INCONCLUSIVE` | `INCONCLUSIVE` | Clean Current route, HCCL ABI and sparse phase evidence narrows conditional numerators and observed cadence, but complete compulsory work, mixed attainable capacity and all8 DAG remain missing; no finite Product ceiling or performance KEEP |
| `loop-078` | `INCONCLUSIVE` | `INCONCLUSIVE` | Exact clean route and local count-copy observations narrow Current numerator and normal-path ordering, but all-layer row identity, downstream Draft/DSA consumption, A/A acceptance parity and all8 typed joins are incomplete. Run411 startup OOM and Run417 container reset invalidate combined timing calibration. V3.11 has no matched necessary-work/capacity certificate or finite Product ceiling. |
| `loop-079` | `INCONCLUSIVE` | `INCONCLUSIVE` | Run576-578 narrow current route, resident GMM service and counter traffic, but formal W-minus/C-plus and Product DAG remain incomplete; Run578 original paired timing invalid and strict endpoints null |
| `loop-080` | `INCONCLUSIVE` | `INCONCLUSIVE` | Run579-580 reproducibly reject independent-ready whole-chain GMM-HCCL ideal overlap; Run581 source gate selects one legal production DAG cut but loaded branch, typed endpoints and resource lower bounds remain open |
| `loop-081` | `EVALUATING` | `PENDING` | 审查 Run run620 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
