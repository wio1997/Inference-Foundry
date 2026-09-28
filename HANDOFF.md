# Inference Foundry / DeepSeek Extreme — HANDOFF

> Purpose：新对话、新 Agent、新机器或长时间中断后的唯一恢复入口。  
> Rule：每次重要 checkpoint 后更新本文件，并与代码/evidence 一起 commit + push。  
> 本文件只保存“当前现场”和恢复规则，不写成长历史；历史细节放 `PROJECT_STATE.md`、TaskCtl 和 `evidence/`。

---

# 1. Project identity

项目：**Inference Foundry / DeepSeek Extreme P0**

当前目标有两层：

## A. DeepSeek Extreme P0

在冻结 calibration contract 下，把 DeepSeek V4 Flash 的 Framework / Scheduling / Runtime 执行尽可能逼近当前可实现性能边界。

当前 P0：

- Model：DeepSeek V4 Flash
- Quant：W4A8
- Hardware：8×Ascend 910B3
- Parallelism：DP1×TP8
- Speculation：DSpark7
- Primary Performance Anchor：48×32K→1024，c12

## B. Inference Foundry Method

用 DeepSeek P0 第一次完整跑通：

**Contract → Current → Execution DAG → Necessary Work → Resource Model → Bound → Gap → Runtime Restructuring → Correctness → Formal E2E → Re-bound**

P0 的 Runtime 可以高度专项化。

未来跨模型真正复用的是：

- 方法；
- schema；
- 工具；
- evidence 规则；
- Bound / DAG / Resource Model 逻辑；
- performance knowledge；
- agent workflow。

不要求复用 DeepSeek 的具体 Runtime 代码、固定 shape 或调度策略。

---

# 2. New-session read order

任何新对话 / 新 Agent 开始工作时，按顺序读取：

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

只有当前问题需要时，再打开对应 raw evidence。

---

# 3. Current checkpoint — MUST REFRESH AT EACH HANDOFF

> 本节是“当前现场快照”。  
> 每次准备结束一个长会话、Loop 或阶段时必须刷新。

As of 2026-09-28：

- Formal achieved Current：**571.681 output tok/s**（Run99，当前正式 achieved point）
- Frozen formal workload：`48×32K→1024, c12`
- 当前主线：**Framework/Scheduling-only Bound**
- 当前阶段：保持正确 primitive/operator 及其 shape-conditioned cost 不作为主要优化变量
- 当前 Framework-only numerical ceiling：**尚未可信识别，禁止为了给数字而编造**
- 当前核心工作：恢复/收紧真实 scheduling DAG、Host issue、async queue、Graph boundary、rank arrival skew、collective peer wait、resource contention
- 已确认：长 HCCL duration 可能包含 peer wait，不能直接当 intrinsic communication cost
- 已确认：Host marker gap 不等于同样大小的 exposed device idle / removable wall
- 当前目标不是继续“找优化点”，而是建立可审计的 `Current → Optimistic Bound → Achievable Bound → Remaining Gap`

### Immediate objective

继续 Framework/Scheduling-only 主线，直到能够：

1. 给出可审计的 Optimistic Bound；
2. 明确主要 unknown / sensitivity；
3. 用最小实验校准高价值假设；
4. 收紧为 Achievable Bound 或可信 interval；
5. 指出最大的 causally removable Gap；
6. 做至少一个 Bound-guided structural Runtime change；
7. correctness；
8. repeated formal E2E；
9. Re-bound。

### Do not do yet

- 不要为了未来通用性先扩成通用 scheduler；
- 不要同时启动第二个模型；
- 不要把 32K/c12 的偶然属性写成 Foundry 方法论；
- 不要跳过 Bound，回到“看到哪里慢就优化哪里”；
- 不要把 primitive/kernel 优化混进当前 Framework-only 归因，除非主线明确 PIVOT；
- 不要因为超过 Stock 就停止。

### Latest-state rule

本节只是恢复入口，不替代最新事实。

新 Agent 恢复后必须用：

- Git HEAD；
- TaskCtl；
- committed evidence；
- 当前机器状态；

重新确认本节内容仍然成立。

---

# 4. Source-of-truth priority

出现冲突时，按以下优先级判断：

1. **当前真实机器 / 进程 / 文件状态**
2. **当前 Git HEAD + source status**
3. **TaskCtl current loop/run**
4. **committed evidence**
5. **`HANDOFF.md` 当前 checkpoint**
6. **`PROJECT_STATE.md` / `PERFORMANCE_MAP.md` / `ACHIEVABLE_BOUND.md` / `RESULTS.md`**
7. **旧聊天 / 旧记忆**

说明：

- `HANDOFF.md` 是新会话的当前现场摘要，所以优先于长期累积文档中的旧摘要；
- 但任何 `HANDOFF.md` 内容都不能覆盖真实机器、Git、TaskCtl 和 evidence；
- 旧聊天只能作为线索，不能作为最新项目事实。

---

# 5. Multi-agent architecture

## Sol — main owner

负责：

- 主线推进；
- Runtime / execution architecture；
- Execution DAG；
- Bound；
- Resource Model；
- profiling 归因；
- correctness；
- benchmark 设计；
- KEEP / REJECT / PIVOT；
- Performance Map 更新；
- 审阅和整合其他 Agent 结果。

**只有 Sol 可以把子 Agent 产物升级为项目结论。**

---

## Astra Medium — independent reviewer

适合：

- benchmark/profile 结果复核；
- 多个候选之间的判断；
- 重要实验设计 challenge；
- 重新检查 Performance Map；
- 发现 Sol 归因偏差；
- 当前路线反复或收敛变慢时独立复盘。

不接管主线。

---

## Astra High — high-value architectural reviewer

适合：

- Bound 是否成立；
- 执行架构是否走偏；
- 多组 evidence 冲突；
- Graph / Persistent / Whole-cycle Replay；
- 重大架构分叉；
- Performance Map 与 Achievable Bound 存在巨大但难解释的 Gap；
- 一个错误判断会浪费大量后续实验的关键决策。

Astra High 的结论是 independent evidence，不自动覆盖 Sol。

---

## DeepSeek / Zcode — mechanical executor

只做边界明确、低风险、容易验收的机械执行，例如：

- 启停服务；
- 环境检查；
- benchmark；
- 重复测试；
- 运行已有脚本；
- profiling / log / trace 采集；
- 检查进程 / NPU；
- 提取数据；
- 整理结果；
- 限定范围源码定位；
- 简单、可机械验收的修改。

不要把下面任务交给 Zcode 直接裁决：

- Bound；
- 根因；
- KEEP / REJECT / PIVOT；
- 重大 Runtime 架构；
- 跨模块优化方向；
- 性能极限判断。

---

# 6. Zcode / DeepSeek reliability gate

这是强制规则。

**委派成功 ≠ 任务成功。**

每次 Zcode/DeepSeek 返回后，Sol 必须核验：

## Execution identity

- [ ] 实际命令
- [ ] working directory
- [ ] 配置模型
- [ ] 日志中实际观察到的模型 / runner

## Process result

- [ ] exit code
- [ ] timeout
- [ ] kill / OOM / signal
- [ ] stdout / stderr 是否完整

### Rule

任何 timeout 默认 `INVALID / INCONCLUSIVE`，除非产物和进程状态能独立证明目标动作已完整完成。

## Artifacts

- [ ] 预期文件存在
- [ ] 文件大小合理
- [ ] timestamp 属于本次 Run
- [ ] JSON / CSV 可解析
- [ ] 必要时记录 SHA256

## Benchmark validity

- [ ] 请求数正确
- [ ] 成功请求数正确
- [ ] input/output token 合同正确
- [ ] concurrency 正确
- [ ] warm/cold/prefix 协议正确
- [ ] correctness gate 真实执行
- [ ] 没有把 diagnostic TPS 当 formal TPS

## Environment

- [ ] `npu-smi info`
- [ ] `docker ps`
- [ ] 服务 PID
- [ ] port 占用
- [ ] 残留进程
- [ ] HBM 是否恢复
- [ ] 是否存在第二服务冲突

## Source state

- [ ] Git HEAD
- [ ] `git status --short`
- [ ] 临时 patch 是否记录
- [ ] cleanup / restore 是否成功
- [ ] shared bind-mount source 是否被意外修改

关键项不通过，不得写成 PASS / KEEP。

---

# 7. No-background rule

不要声称：

- “实验还在后台继续”
- “Agent 会自动跑完”
- “之后会自动回来汇报”

除非存在可验证的：

- PID；
- process；
- automation；
- runner；
- 持续写入日志；
- 明确远端任务机制。

如果没有，应明确写：

> 当前没有后台实验在运行。

---

# 8. Run lifecycle

每个真实 Run 推荐：

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
13. Update HANDOFF current checkpoint when needed
14. Commit + push

### Never skip cleanup verification

尤其检查：

- source patch；
- service PID；
- profiling flag；
- env vars；
- port；
- NPU process；
- temporary observer。

---

# 9. Evidence classes

任何数字必须标明类别：

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

任何 Bound 都必须说明：

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

# 11. Document update policy

这些文件不是同一种更新频率。

## 高频更新

### `HANDOFF.md`

每次长会话 / 关键 Loop / 重大 checkpoint 结束时更新。

主要更新：

- Current；
- active loop/run；
- Bound 状态；
- 最大 Gap / 最大 unknown；
- latest KEEP / REJECT / PIVOT；
- machine/service/source 状态；
- next_action；
- blocker；
- background task 状态。

### `PROJECT_STATE.md`

每个有意义的 Loop / Run 后更新当前项目状态和历史记录。

### `PERFORMANCE_MAP.md`

当 bottleneck、critical path、Gap 或 causal attribution 改变时更新。

### `ACHIEVABLE_BOUND.md`

当：

- 新增/删除必要依赖；
- primitive/resource cost 证据变化；
- overlap/contention 假设被验证或否定；
- Bound interval / sensitivity 改变；
  时更新。

### `RESULTS.md`

当产生正式可对外保留的性能/正确性结论时更新。

---

## 低频更新

### `AGENTS.md`

只有以下情况才改：

- Agent 分工改变；
- 实验纪律改变；
- 结果验收规则改变；
- 发现长期会导致 Agent 跑偏的新规则。

不要每个 Loop 改。

### `MISSION.md`

只有产品目标、P0 calibration contract 或阶段目标真正变化时才改。

不要拿它记录日常进度。

### `FOUNDRY_METHOD.md`

只有 Foundry 方法本身经过实践证明需要修订时才改。

例如：

- Bound 流程需要增加新步骤；
- DAG / Resource Model schema 发现缺陷；
- 第二 workload / 第二模型证明某方法不具备可迁移性。

不要拿它记录 DeepSeek 的日常实验。

---

# 12. Handoff update checklist

每次准备结束一个长会话 / 阶段时：

- [ ] 当前 Git HEAD
- [ ] 当前 active loop/run
- [ ] formal Current
- [ ] Bound 状态
- [ ] 最大已知 Gap / 最大 unknown
- [ ] 最新 KEEP / REJECT / PIVOT
- [ ] 当前服务 / 进程状态
- [ ] source clean / dirty
- [ ] raw evidence 在哪里
- [ ] next_action
- [ ] blocker
- [ ] 是否存在后台任务
- [ ] 更新本文件第 3 节
- [ ] 更新 TaskCtl / PROJECT_STATE
- [ ] 必要时更新 PERFORMANCE_MAP / ACHIEVABLE_BOUND / RESULTS
- [ ] commit + push

---

# 13. New-chat bootstrap prompt

新对话可以直接发送：

> 这是 Inference Foundry / DeepSeek Extreme P0。请先读取仓库根目录 `HANDOFF.md`、`AGENTS.md`、`MISSION.md`、`FOUNDRY_METHOD.md`，再读取 `PROJECT_STATE.md`、`PERFORMANCE_MAP.md`、`ACHIEVABLE_BOUND.md`、`RESULTS.md` 和当前 TaskCtl recovery pack。不要从旧对话猜当前状态，以当前机器、Git HEAD、TaskCtl 和 committed evidence 为准。
>
> 当前目标是用 DeepSeek V4 Flash W4A8 + 8×910B3 + DP1×TP8 + DSpark7 + 32K→1K c12 calibration workload，完整跑通 Current→DAG→Resource Model→Bound→Gap→Runtime restructuring→Correctness→Formal E2E→Re-bound 的 Foundry 闭环。当前先聚焦 Framework/Scheduling，避免无意混入 primitive/operator 优化。
>
> Sol 负责主线和裁决；Astra 用于独立复核；Zcode/DeepSeek 只用于机械执行。任何 Zcode 结果必须核验实际模型、命令、退出码、timeout、产物、服务/NPU/source 状态后才能采信。没有真实 PID/runner 时，不要声称后台实验仍在运行。
>
> 恢复完成后，先给出：
>
> 1. Current；
> 2. 当前 Bound 状态；
> 3. 最大已知 Gap / 最大 unknown；
> 4. 最新有效 Run；
> 5. 当前机器 / 服务 / source 状态；
> 6. next_action；
>    然后继续推进。

---

# 14. What to upload to GitHub

每次稳定 checkpoint 至少提交：

- `HANDOFF.md`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`
- `tasks/deepseek-extreme-p0/**`
- 小型关键 evidence / manifest / finding

以下文件只在规则/目标/方法变化时提交新版本：

- `AGENTS.md`
- `MISSION.md`
- `FOUNDRY_METHOD.md`

大 raw trace / log 可以不进 Git，但必须：

- 有路径；
- 有 SHA / manifest；
- 有 scope；
- 有 limitation；
- 新 Agent 能找到。

---

# 15. One-line recovery rule

> **先恢复事实，再恢复判断；先验证子 Agent 产物，再继续实验；先沿 Bound 找 Gap，再改代码。**
