# Inference Foundry / DeepSeek Extreme — HANDOFF

> Purpose：新对话、新 Agent、新机器或长时间中断后的恢复入口。  
> Rule：本文件保存“当前现场”和恢复规则，不写成长历史。历史细节放 `PROJECT_STATE.md`、TaskCtl、Performance Knowledge 和 `evidence/`。  
> 文档用于帮助 Agent 恢复与判断，不是固定 SOP。

---

# 1. Project identity

项目：**Inference Foundry / DeepSeek Extreme P0**

当前唯一中心目标：

> 在 DeepSeek V4 Flash W4A8 + 8×Ascend 910B3 + DP1×TP8 + DSpark7 的冻结 workload 下，先把 **Framework / Scheduling 层的可实现性能极限找出来，并把 Runtime 真正做到尽可能接近这个极限**。

当前 Calibration Workload / Primary Performance Anchor：

- Model：DeepSeek V4 Flash
- Quant：W4A8
- Hardware：8×Ascend 910B3
- Parallelism：DP1×TP8
- Speculation：DSpark7
- Workload：48×32K→1024
- Concurrency：12

当前阶段暂不把 primitive/operator 优化作为主要变量。

当前 32K/c12 只是第一条校准案例，可以为它做激进专项化，但不要把其固定 shape、固定 cohort 或当前 Runtime 结构自动提升为跨模型通用原则。

---

# 2. New-session recovery order

任何新对话 / 新 Agent 开始工作时，建议按以下顺序恢复：

1. `HANDOFF.md`
2. `AGENTS.md`
3. `MISSION.md`
4. `FOUNDRY_METHOD.md`
5. `PROJECT_STATE.md`
6. `PERFORMANCE_MAP.md`
7. `ACHIEVABLE_BOUND.md`
8. `RESULTS.md`
9. 当前 TaskCtl / recovery pack
10. `git log -5 --oneline`
11. 当前问题真正需要的 evidence / Performance Knowledge

不要从旧聊天、本地旧副本或记忆猜当前状态。

---

# 3. Source-of-truth priority

出现冲突时，优先级：

1. 当前真实机器 / 进程 / 文件状态
2. 当前远端 Git HEAD + source status
3. TaskCtl current loop/run
4. committed evidence
5. `HANDOFF.md` 当前 checkpoint
6. `PROJECT_STATE.md` / `PERFORMANCE_MAP.md` / `ACHIEVABLE_BOUND.md` / `RESULTS.md`
7. 旧聊天 / 旧记忆

`HANDOFF.md` 是恢复入口，但不能覆盖真实机器、Git、TaskCtl 和 evidence。

---

# 4. Current checkpoint

## Active continuation — Run677 complete / Run678 preparation

Run669 exact-ID reuse excludes prior-cohort tails from Run602/606. In Run606 the corrected Target Host work is 1.785–2.533 s/cohort, proposal .297–.455 s, and post-proposal→next execute gaps only31–103 ms. Main opportunity is repeated mixed-prefill caused by incremental request readiness. Source finds default renderer_num_workers=1; CPU text-path screen workers1/4/12 reduces median12-request batch from1352.753 to446.931/297.508ms (corrected effective Ascend thinking template) with exact token IDs. This screen applies the serving thinking patch; it excludes complete API validation and Core prefix hashing, and is not Product evidence.

Current configuration: input-token cache + initial prefix-hash memo + renderer workers4 + metadata Graph. Latest formal Current measurement: **660.3094257763639 output tok/s**. Historical best achieved remains **676.8246305810562 (Run674, Graph OFF)**; do not assign that old number to the changed configuration.

Run675 KEEP; read evidence/20260929_loop081_bound/run675/verdict.md. Run677 completed and independently REJECTED: OFF_A681.915150 / ON668.881455 / OFF_B671.066709 tok/s. All576requests/all384all8records/cleanup/source SHA pass; controller3836844 ended, 8080closed, all8idle. Read run677/verdict.md and summary.json. Current accepted configuration unchanged. Run678 combines first8 startup slot certificate with deferred audit against eager original baseline (AUDIT_BASELINE0); source High review passed after cross-group alias/dtype hardening; Zcode preparation/preflight completed:54CPU same-state scenarios and all fallback/alias/mutation/overflow gates passed; all9cleanup0 and current source/script SHA independently verified. Run678 ready for one live OFF_A/ON/OFF_B. No new live service yet; verify actual processes before launch. Run676 narrow ordinary-boundary observer remains unexecuted fallback; no duplicate experiment. Historical sections below do not override this current paragraph.


As of 2026-09-29，恢复后仍需用实时 Git HEAD / TaskCtl / 机器状态重新确认。

## Formal Current

- 正式 achieved Current：**660.309426 output tok/s (Run675 latest configuration)**
- 来源：Run675 input-token + initial prefix-hash cache + metadata Graph
- 冻结 workload：48×32K→1024, c12
- Run99 571.681 为历史 Current；Run675 为最新已验证整合结果；Run674 676.824631 是历史最佳 achieved

## Current main line

当前主线：

> **Framework / Scheduling-only Bound + 最大 causally removable Gap**

当前仍未可信得到：

- Framework-only numerical ceiling
- Achievable Bound
- Current→Bound 最大可消除 Gap

不要为了给数字而编造 Bound。

---

# 5. Latest evidence — Run667 / Run668

## Run667 — metadata Graph formal E2E

metadata Graph 的局部收益已经被正式验证：

- OFF median：578.117 tok/s
- ON median：567.938 tok/s
- ON 正式 E2E 比同期 OFF 低约 1.761%
- ON 每 serving cycle 仍快约 1.2–1.6 ms
- 但 client−Runtime residual 三轮均更高
- repeat3 还存在明显 trajectory 差异：
  - OFF 1166 cycles
  - ON 1256 cycles
  - ON 多 90 cycles
  - useful tokens/cycle 从 42.15 降至 39.13

结论：

> metadata Graph 作为独立候选不晋升；Run99 仍是正式 Current。

这 90 cycles 的差异属于算法/acceptance trajectory 变化，不能当成 Framework overhead。

---

## Run668 — Product wall stage alignment

Run668 做了最小 OFF/ON/OFF 阶段对齐。

主要结果：

| Arm   | client wall | cycles | Runtime ms/cycle | request start→handoff | handoff build | metadata capture | Runtime→client end |
| ----- | ----------: | -----: | ---------------: | --------------------: | ------------: | ---------------: | -----------------: |
| OFF_A |    82.197 s |   1225 |        56.495 ms |              12.141 s |      ~15.4 ms |                0 |           ~0.833 s |
| ON    |    78.314 s |   1195 |        55.349 ms |              11.189 s |      ~15.4 ms |    ~168 ms total |           ~0.799 s |
| OFF_B |    86.680 s |   1196 |        56.621 ms |              18.167 s |      ~15.4 ms |                0 |           ~0.779 s |

metadata Graph capture 每 cohort 约 40–47 ms，四个 measured cohort 合计约 168 ms。

关键结论：

> Run667 的 2–10 秒 Product residual 波动，不是 metadata capture 直接造成的。

真正的大波动发生在：

> **请求开始 → Runtime handoff 之前**

因此不要继续单独调 metadata Graph。

---

# 6. Highest-value unresolved region

复用 Run332 后，当前最高价值的未知区间收窄到：

> **residual-prefill / DSpark seed → Runtime handoff 的多次 execute 链**

Run332 历史证据：

- client start → first worker execute：约 0.20–0.23 s
- first worker execute → Runtime handoff：约 **2.09–2.43 s / cohort**
- 每 cohort 约 8 次 execute
- 调度 token 约 1324–1366

目前还没有把这 2.09–2.43 s 分清：

- 哪些是必要 residual-prefill 计算
- 哪些是 DSpark seed 必要计算
- 哪些是 Host / scheduler 等待
- 哪些是 rank rendezvous
- 哪些是 HCCL / Device 等待
- 哪些是真正可消除的 Framework / Scheduling Gap

这就是当前最重要的问题。

---

# 7. Current next direction

恢复后不要停在 Run668 的 checkpoint。

优先继续：

> 复用 Run332、Run337、Run602、Run606，以及相关源码和 Performance Knowledge，把 residual-prefill / seed → Runtime handoff 的完整执行链拆开。

优先做：

- 源码分析
- 旧 evidence 复用
- DAG 重建
- 必要 work 与等待分类
- rank rendezvous / Host issue / HCCL / Device completion 对齐

在已有证据能回答问题时，不要重复跑 NPU。

如果仍缺一条决定性依赖或 cost：

> 只设计补这一条 unknown 的最小实验。

一旦定位出可信的 causally removable Gap：

> 直接进入 Runtime intervention → correctness → frozen repeated formal E2E → Re-bound。

不要因为“还不能安全改 Runtime”就停止；这意味着继续找因果，不是结束工作回合。

---

# 8. Current machine / source state

Run668 收尾状态：

- 服务已停
- 8080 无监听
- 8 张 NPU idle
- source / script SHA 已恢复
- 内层实验脚本和 cleanup 均成功
- 外层 Zcode CLI 在任务完成后滞留，最终由 timeout 结束
- 该 timeout 不影响已经独立确认完成的 Run668
- 不要因此重跑 Run668

恢复后仍应重新检查：

- `git rev-parse HEAD`
- `git status --short`
- `npu-smi info`
- `docker ps`
- port 8080
- TaskCtl current state

---

# 9. Agent architecture

## Astra Light — Main Agent

Astra Light 是默认主 Agent。

负责：

- 恢复当前真实状态
- 判断当前最高价值问题
- 决定下一步实验 / 分析 / Runtime 改动
- 维护 DAG / Resource Model / Bound / Performance Map
- 设计 correctness / benchmark
- KEEP / REJECT / PIVOT
- 整合其他 Agent 结果
- 持续推进主线

Astra Light 不需要机械执行历史 `next_action`。

上一阶段的 next_action 只是候选；恢复后应结合最新证据独立判断它是否仍是最高价值方向。

## Astra Medium — Important Problem Escalation

适合：

- profiling / benchmark 归因复杂
- 多个解释难以区分
- KEEP / REJECT / PIVOT 判断不稳
- Bound 中的重要假设需要独立复核
- 当前路线反复
- 多轮实验不收敛
- 需要第二视角 challenge

Medium 用于提高判断质量，不默认接管主线。

## Astra High — Critical Architecture / Bound Review

适合少量高价值问题：

- Optimistic / Achievable Bound 是否可信
- Framework/Scheduling 极限是否接近
- 重大 Runtime architecture
- 多组 evidence 冲突
- Persistent Execution
- Whole-cycle Replay
- DSpark Graph
- 跨模块调度重构
- 一个错误判断会浪费大量实验

最终仍以真实源码、实验、correctness 和正式 E2E 为证据。

## Zcode / DeepSeek — Mechanical Executor

主要负责：

- 启停服务
- benchmark
- profiling / trace / log
- 环境检查
- 数据提取
- 重复测试
- 已明确方案的简单修改
- 脚本运行
- 限定范围源码定位

不要让其独立裁决：

- Bound
- 性能极限
- 根因
- KEEP / REJECT / PIVOT
- 重大架构
- 高风险 correctness

其结果需要根据任务风险验收。

详细调用和验收以 `ZCODE_OPERATIONS.md` 为准。

---

# 10. Model selection principle

不规定固定调用比例，也不按关键词机械路由。

默认：

**Astra Light 持续推进  
→ 重要复杂问题升级 Astra Medium  
→ 极关键架构 / Bound 问题升级 Astra High  
→ Zcode / DeepSeek 承担机械执行**

模型分工是为了提高判断质量和推进效率，不是新的 SOP。

---

# 11. Historical evidence reuse

历史 Run 的目标是减少重复实验。

在启动成本较高的新实验前，优先检查：

- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `performance_knowledge/entries.jsonl`
- `scripts/performance_knowledge.py`
- TaskCtl
- `evidence/`

历史结论只能作为：

- evidence
- hypothesis prior
- counterexample

不能自动作为当前 KEEP / REJECT。

如果已有证据足够，直接复用。

如果条件变化，只验证变化部分。

如果一次低成本实验比历史检索更便宜，Astra Light 可以直接验证。

---

# 12. Hard constraints

真正的硬门槛只有少数几类：

## Correctness

不能用错误语义换性能。

## Formal E2E

正式性能结论必须来自同口径、可重复、合同一致的真实 E2E。

## Evidence integrity

不要把 profile/microbench/diagnostic/simulation 直接升级成 Product gain。

## Comparability

改变 workload / semantics / acceptance / parallelism / protocol 后，不能继续当成同口径正式比较。

## Zcode verification

高价值结果要核验实际命令、模型、exit code、timeout、产物、服务/NPU/source/cleanup。

## No fake background execution

没有真实 PID / runner / automation / 持续日志时，不要声称后台仍在运行。

---

# 13. Continue / stop rule

不要因为以下任何一个事件就默认结束工作回合：

- 一个 Run 完成
- 一个 commit 完成
- 一次 REJECT
- 一次 evidence 提交
- 服务正常停机
- 超过 Stock
- 暂时不能安全修改 Runtime

每个 checkpoint 后，Astra Light 重新判断：

- 当前最大 Gap 是什么
- 最大 unknown 是什么
- 原路线是否仍值得
- 是否出现更高价值候选
- 是否应该换问题或换阶段

如果没有真实 blocker，且仍存在明确高价值下一步，就继续推进。

只有以下情况适合真正停下：

- 需要用户提供外部信息 / 权限 / 资源
- 存在无法靠现有证据自行裁决的重大分叉
- correctness 风险无法继续
- 当前阶段没有高价值下一步
- 机器/环境强制结束

---

# 14. Document update policy

## 高频更新

- `HANDOFF.md`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`
- TaskCtl

## 低频更新

- `AGENTS.md`
- `MISSION.md`
- `FOUNDRY_METHOD.md`

不要把状态变化写进规则层文档。

---

# 15. New-chat bootstrap prompt

新对话可直接发送：

> 这是 Inference Foundry / DeepSeek Extreme P0。请先读取仓库 `HANDOFF.md`、`AGENTS.md`、`MISSION.md`、`FOUNDRY_METHOD.md`，再读取 `PROJECT_STATE.md`、`PERFORMANCE_MAP.md`、`ACHIEVABLE_BOUND.md`、`RESULTS.md`、当前 TaskCtl / recovery pack 和最近 Git commit。不要依赖旧聊天或本地旧副本猜状态，以真实机器、远端 Git HEAD、TaskCtl 和 committed evidence 为事实源。
>
> 当前中心目标只有一个：在冻结 DeepSeek V4 Flash W4A8 + 8×910B3 + DP1×TP8 + DSpark7 + 48×32K→1024 c12 workload 下，先把 Framework/Scheduling 层的可实现性能极限找出来，并持续把 Runtime 做到更接近这个极限；primitive/operator 暂不作为主要变量。
>
> Astra Light 是主 Agent，负责独立判断和持续推进；复杂归因可升级 Astra Medium；关键 Bound / 架构问题可升级 Astra High；Zcode/DeepSeek 负责机械执行并按风险验收。
>
> 恢复后把历史 next_action 当候选而不是命令。先简要告诉我 Current、Bound 状态、最大 Gap/unknown、最近有效证据、机器/source 状态，以及你重新判断后的最高价值下一步，然后直接继续工作。不要因为一个 Run、commit、REJECT、服务停机或 evidence 提交就结束工作回合。

---

# 16. One-line recovery rule

> **先恢复事实，再独立判断；优先复用历史证据；围绕最大 Gap 持续推进，直到真正遇到 blocker。**
