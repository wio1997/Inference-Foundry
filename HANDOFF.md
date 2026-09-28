# Inference Foundry / DeepSeek Extreme — HANDOFF

> Purpose: 新对话、新 Agent、新机器或长时间中断后的唯一恢复入口。  
> Rule: 每次重要 checkpoint 后更新本文件，并与代码/evidence 一起 commit + push。  
> 本文件应保持“短、准、可执行”，不要变成历史日志；历史细节放 `PROJECT_STATE.md`、TaskCtl 和 `evidence/`。

---

# 1. Project identity

项目：**Inference Foundry / DeepSeek Extreme P0**

当前目标有两层：

## A. DeepSeek Extreme P0

在当前冻结 calibration contract 下，把 DeepSeek V4 Flash 的 Framework / Scheduling / Runtime 执行尽可能逼近当前可实现性能边界。

当前 P0：

- Model: DeepSeek V4 Flash
- Quant: W4A8
- Hardware: 8×Ascend 910B3
- Parallelism: DP1×TP8
- Speculation: DSpark7
- Primary Performance Anchor: 48×32K→1024, c12

## B. Inference Foundry Method

用 DeepSeek P0 第一次完整跑通：

**Contract → Current → Execution DAG → Necessary Work → Resource Model → Bound → Gap → Runtime Restructuring → Correctness → Formal E2E → Re-bound**

P0 的具体 Runtime 可以高度专项化。

以后跨模型复用的是方法、工具、证据规范和知识，不要求复用 DeepSeek 的具体代码/shape。

---

# 2. New-session read order

任何新对话 / 新 Agent 开始工作时：

1. `HANDOFF.md`
2. `AGENTS.md`
3. `MISSION.md`
4. `FOUNDRY_METHOD.md`
5. `PROJECT_STATE.md`
6. `PERFORMANCE_MAP.md`
7. `ACHIEVABLE_BOUND.md`
8. `RESULTS.md`
9. `git log -5 --oneline`
10. `python3 scripts/taskctl.py resume --task-dir tasks/deepseek-extreme-p0`

不要先通读整个仓库和全部 evidence。

只有当前问题需要时再打开对应 raw evidence。

---

# 3. Current checkpoint — MUST REFRESH AT EACH HANDOFF

> 本节是唯一允许频繁修改的“当前现场”。  
> 交班前必须更新，不要让新 Agent 从旧聊天猜状态。

As of 2026-09-28:

- Formal achieved Current: **571.681 output tok/s**（Run99，当前正式 achieved point）
- Frozen formal workload: `48×32K→1024, c12`
- 当前主线：**Framework/Scheduling-only Bound**
- 当前阶段保持 primitive/operator 实现及其 shape-conditioned cost 不作为主要优化变量
- 当前 Framework-only numerical ceiling: **尚未可信识别 / 不得编造**
- Loop081 正在恢复真实 scheduling DAG、Host issue、async queue、Graph boundary、rank arrival skew、collective peer wait 和资源约束
- 已确认：长 HCCL duration 可能包含 peer wait；不能直接当 intrinsic communication cost
- 已确认：Host marker gap 不等于同样大小的 exposed device idle / removable wall
- 最新证据必须以 `git log`、TaskCtl、`FRAMEWORK_SCHEDULING_BOUND.md` 为准；不要仅依赖本节历史数字

### Immediate objective

继续收紧 Framework/Scheduling-only model，直到能够：

1. 给出可审计的 Optimistic Bound；
2. 通过最小实验校准高价值假设；
3. 收紧为 Achievable Bound / interval；
4. 指出最大的 causally removable gap；
5. 做至少一个 Bound-guided structural Runtime change；
6. formal E2E；
7. Re-bound。

### Do not do yet

- 不要因为未来要支持其他 workload 就先扩成通用 scheduler；
- 不要同时启动第二个模型；
- 不要把 32K/c12 的偶然属性写成 Foundry 方法论；
- 不要跳过 Bound，回到“看到哪里慢就优化哪里”的模式；
- 不要把 primitive/kernel 优化混进当前 Framework-only 归因，除非主线明确 pivot。

---

# 4. Source-of-truth priority

出现冲突时，按以下优先级判断：

1. 当前真实机器/进程/文件状态
2. 当前 Git HEAD + source status
3. TaskCtl current loop/run
4. committed evidence
5. `PROJECT_STATE.md` / `PERFORMANCE_MAP.md` / `ACHIEVABLE_BOUND.md`
6. `HANDOFF.md`
7. 旧聊天 / 旧记忆

旧聊天只能作为线索，不能覆盖最新仓库和机器事实。

---

# 5. Multi-agent architecture

## Sol — main owner

负责：

- 主线；
- 架构；
- DAG / Bound / Resource Model；
- profiling 归因；
- correctness；
- benchmark 设计；
- KEEP / REJECT / PIVOT；
- 更新状态文件；
- 最终采信其他 Agent 结果。

**只有 Sol 可以把子 Agent 产物升级为项目结论。**

---

## Astra Medium — independent reviewer

适合：

- benchmark/profile 复核；
- 多个候选之间的判断；
- 实验设计 challenge；
- 重新检查 Performance Map；
- 发现主 Agent 的归因偏差。

不接管主线。

---

## Astra High — high-cost architectural reviewer

适合：

- Bound 是否成立；
- 执行架构是否走偏；
- 复杂、矛盾 evidence；
- Graph / Persistent / Whole-cycle replay；
- 重大架构分叉；
- 一个判断错误会浪费大量后续实验的场景。

Astra 结论是 independent evidence，不自动覆盖 Sol。

---

## DeepSeek / Zcode — mechanical executor

只做：

- 启停服务；
- 环境检查；
- benchmark；
- 重复测试；
- 运行已有脚本；
- profiling/log/trace 采集；
- 机械数据整理；
- 限定范围源码定位；
- 简单、易验收修改。

不要交给 Zcode：

- Bound 裁决；
- 根因判断；
- KEEP/REJECT/PIVOT；
- 重大 Runtime 架构；
- 跨模块优化方向选择。

---

# 6. Zcode / DeepSeek reliability gate

这是强制规则。

**委派成功 ≠ 任务成功。**

每次 Zcode/DeepSeek 返回后，Sol 必须逐项核验：

## Execution identity

- [ ] 实际命令是什么？
- [ ] working directory 是什么？
- [ ] 配置模型是什么？
- [ ] 日志中实际观察到的模型/runner 是什么？

## Process result

- [ ] exit code 是多少？
- [ ] 是否 timeout？
- [ ] 是否被 kill / OOM / signal？
- [ ] stdout/stderr 是否完整？

### Rule

任何 timeout 默认 `INVALID/INCONCLUSIVE`，除非产物和进程状态能独立证明目标动作已完整完成。

## Artifacts

- [ ] 预期文件是否存在？
- [ ] 文件大小是否合理？
- [ ] timestamp 是否属于本次 Run？
- [ ] JSON/CSV 是否可解析？
- [ ] 必要时是否记录 SHA256？

## Benchmark validity

- [ ] 请求数正确？
- [ ] 成功请求数正确？
- [ ] input/output token 合同正确？
- [ ] concurrency 正确？
- [ ] warm/cold/prefix 协议正确？
- [ ] correctness gate 是否真实执行？
- [ ] 是否误把 diagnostic TPS 当 formal TPS？

## Environment

- [ ] `npu-smi info`
- [ ] `docker ps`
- [ ] 服务 PID
- [ ] port 占用
- [ ] 残留进程
- [ ] HBM 是否恢复
- [ ] 是否存在第二个服务冲突

## Source state

- [ ] Git HEAD
- [ ] `git status --short`
- [ ] 临时 patch 是否记录
- [ ] cleanup / restore 是否成功
- [ ] shared bind-mount source 是否被意外修改

只要关键项不通过，不得写成 PASS / KEEP。

---

# 7. No-background rule

不要声称：

- “实验还在后台继续”
- “Agent 会自动跑完”
- “我会之后回来汇报”

除非存在可验证的：

- PID；
- process；
- automation；
- runner；
- 日志持续写入；
- 明确的远端任务机制。

如果没有，就明确写：

> 当前没有后台实验在运行。

这是项目交接的重要事实。

---

# 8. Run lifecycle

每个真实 Run 推荐遵循：

1. Preflight
2. Freeze inputs/config
3. Environment check
4. Start service / target process
5. Health check
6. Correctness gate
7. Measurement
8. Artifact validation
9. Cleanup / restore
10. Independent review（高价值 Run）
11. TaskCtl record
12. Update state / Bound / Performance Map
13. Commit + push

### Never skip cleanup verification

尤其是：

- source patch；
- service PID；
- profiling flag；
- env vars；
- port；
- NPU process；
- temporary observer。

---

# 9. Evidence classes

任何数字必须带类别：

- `formal_e2e`
- `matched_internal_ab`
- `diagnostic`
- `profile`
- `microbenchmark`
- `simulation`
- `theoretical_bound`
- `independent_review`

禁止类别漂移：

`profile 10ms` ≠ `formal removable 10ms`

`microbench +20%` ≠ `product +20%`

`simulation ceiling` ≠ `achievable performance`

---

# 10. Bound discipline

当前方法同时维护：

## Current

真实、已验证实现现在达到多少。

## Optimistic Bound

在必要依赖和乐观资源假设下的数学最好情况。

## Achievable Bound

通过真实实验校准 overlap、contention、Host/Device、Graph 等假设后收紧的工程边界。

任何 Bound 都必须写：

- workload；
- numerator；
- makespan scope；
- dependency assumptions；
- resource assumptions；
- unknown edges；
- observer effects；
- evidence links；
- confidence。

如果主要未知仍未闭合，允许写：

> numerical bound unidentified

禁止为了“必须给一个数字”而编造 ceiling。

---

# 11. Handoff update checklist

每次准备结束一个长会话/阶段时：

- [ ] 当前 Git HEAD
- [ ] 当前 active loop/run
- [ ] formal Current
- [ ] Bound 状态
- [ ] 最大已知 Gap / 最大未知
- [ ] 最新 KEEP / REJECT / PIVOT
- [ ] 当前服务/进程状态
- [ ] source clean/dirty
- [ ] raw evidence 在哪里
- [ ] next_action
- [ ] blocker
- [ ] 是否存在后台任务
- [ ] 更新本文件第 3 节
- [ ] 更新 TaskCtl / PROJECT_STATE
- [ ] commit + push

---

# 12. New-chat bootstrap prompt

新对话可以直接发送：

> 这是 Inference Foundry / DeepSeek Extreme P0。请先读取仓库根目录 `HANDOFF.md`、`AGENTS.md`、`MISSION.md`、`FOUNDRY_METHOD.md`，再读取 `PROJECT_STATE.md`、`PERFORMANCE_MAP.md`、`ACHIEVABLE_BOUND.md`、`RESULTS.md` 和当前 TaskCtl recovery pack。不要从旧对话猜当前状态，以 Git HEAD、TaskCtl、evidence 和真实机器状态为准。
>
> 当前目标是用 DeepSeek V4 Flash W4A8 + 8×910B3 + DP1×TP8 + DSpark7 + 32K→1K c12 calibration workload，完整跑通 Current→DAG→Resource Model→Bound→Gap→Runtime restructuring→Correctness→Formal E2E→Re-bound 的 Foundry 闭环。当前先聚焦 Framework/Scheduling，避免无意混入 primitive/operator 优化。
>
> Sol 负责主线和裁决；Astra 用于独立复核；Zcode/DeepSeek 只用于机械执行。任何 Zcode 结果必须核验实际模型、命令、退出码、timeout、产物、服务/NPU/source 状态后才能采信。没有真实 PID/runner 时，不要声称后台实验仍在运行。
>
> 恢复完成后先给出：Current、当前 Bound 状态、最大已知 Gap/未知、最新有效 Run、当前机器/服务状态、next_action，然后继续推进。

---

# 13. What to upload to GitHub

每次稳定 checkpoint 至少提交：

- `AGENTS.md`
- `MISSION.md`
- `FOUNDRY_METHOD.md`
- `HANDOFF.md`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`
- `tasks/deepseek-extreme-p0/**`
- 小型关键 evidence / manifest / finding

大 raw trace/log 可以不进 Git，但必须：

- 有路径；
- 有 SHA/manifest；
- 有 scope；
- 有 limitation；
- 新 Agent 能找到。

---

# 14. One-line recovery rule

> **先恢复事实，再恢复判断；先验证子 Agent 产物，再继续实验；先沿 Bound 找 Gap，再改代码。**
