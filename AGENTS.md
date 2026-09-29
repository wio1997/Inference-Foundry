# Inference Foundry / DeepSeek Extreme — AGENTS

> 作用：给 Agent 提供目标、上下文、证据纪律和协作方式。  
> 目的不是把 Agent 变成固定 SOP 执行器，而是让高能力 Agent 在可靠边界内保持最大技术自主性。

---

# 1. Project Goal

当前项目目标：

> 构建 DeepSeek Extreme P0，并用它跑通 Inference Foundry 第一条“寻找推理极限并逼近它”的完整闭环。

当前 P0 固定：

- DeepSeek V4 Flash
- W4A8
- 8×Ascend 910B3
- DP1×TP8
- DSpark7

当前 `48×32K→1024, c12` 是：

> **Calibration Workload / Primary Performance Anchor**

它用于控制变量、建立 Current、恢复 DAG、构建 Bound、验证方法。

它不是 Inference Foundry 的永久架构假设。

允许针对当前 calibration workload 做非常激进的专项化，但不要把固定 32K、固定 c12、固定 cohort、当前 shape 或当前 Runtime 结构自动提升为跨模型通用原则。

跨模型真正希望复用的是：

**Contract → Current → Execution DAG → Necessary Work → Resource Model → Bound → Gap → Runtime Restructuring → Correctness → Formal E2E → Re-bound**

而不是 DeepSeek P0 的具体 Runtime 代码。

---

# 2. Agent Autonomy

本项目强调：

> **目标和裁决标准相对固定，解决路径自由。**

除硬约束外，本文件、`FOUNDRY_METHOD.md`、`MISSION.md`、`HANDOFF.md` 中的流程、顺序、候选方向和工具选择，都应视为：

- 默认工作框架；
- 已有经验；
- 高价值先验；
- 推荐做法；

而不是必须机械执行的 SOP。

Sol 可以根据新的源码、profiling、实验结果、架构判断或更优证据：

- 跳过低价值步骤；
- 合并或拆分阶段；
- 调整 Foundry Method 的执行顺序；
- 推翻已有假设；
- 放弃既定计划；
- 选择文档没有列出的优化方向；
- 设计新的测量方法；
- 修改 Runtime 边界；
- 重新定义当前最大 Gap；
- 重新解释已有 Performance Map；
- 挑战已有 Bound；
- 直接进入更高价值的实验或实现。

如果偏离默认路线，只需说明：

1. 为什么原路线当前价值更低；
2. 新路线预期获得什么新信息或收益；
3. 如何保持 correctness、证据质量和可比性。

不要为了“遵守流程”而浪费实验。

---

# 3. 真正的 Hard Constraints

只有少数规则是硬门槛。

## 3.1 Correctness

不能用错误语义换性能。

必须根据候选改动的风险，验证必要的：

- state；
- KV；
- acceptance；
- rank / shape；
- model semantics；
- request completion；
- API / output contract。

具体校验方式可由 Sol 根据问题自主设计。

## 3.2 Formal performance evidence

正式性能结论必须来自：

> 同口径、可重复、合同一致的真实 E2E。

以下可以作为分析证据，但不能单独升级为最终产品收益：

- profile；
- microbenchmark；
- diagnostic TPS；
- internal accepted-token TPS；
- simulation；
- theoretical saving。

## 3.3 Evidence integrity

不要把：

- profiler duration 当 removable wall；
- historical KEEP 当当前 KEEP；
- microbench speedup 当 Product speedup；
- simulation ceiling 当 achieved performance；
- 未完成/超时/缺产物的 Run 当成功。

不确定时允许明确写：

- unknown；
- unresolved；
- inconclusive；
- numerical bound unidentified。

宁可保留未知，也不要制造确定性。

## 3.4 Frozen-contract comparability

如果要和正式 Baseline / Current 做性能比较，应保持相关合同一致。

如果修改了：

- workload；
- concurrency；
- model semantics；
- acceptance；
- sampling；
- parallelism；
- benchmark protocol；

需要明确说明结果已经不再是直接同口径对比。

## 3.5 Zcode / DeepSeek result verification

Zcode / DeepSeek 可以做大量机械工作，但其报告不能自动成为项目事实。

采信前至少应确认：

- 实际命令；
- 工作目录；
- 实际模型 / runner；
- exit code；
- timeout / kill / OOM；
- 关键日志；
- 预期产物；
- benchmark 请求/token 合同；
- 必要的服务/NPU状态；
- source Git 状态；
- cleanup / restore。

验证深度应与任务风险匹配。

高价值性能结论和 correctness 结论需要严格验证。

## 3.6 No fake background execution

不要声称实验、Agent 或任务仍在后台继续，除非有可验证的：

- PID；
- process；
- runner；
- automation；
- 持续日志；
- 远端任务机制。

没有就明确说明当前没有后台任务。

---

# 4. 默认工作哲学

下面是优先原则，不是强制顺序。

## 4.1 优先研究 Gap，而不是模块

默认优先问：

> 当前真实执行与当前可实现性能边界之间，最大的可消除 Gap 是什么？

而不是：

> 下一个应该优化哪个模块？

模块只是实现边界，不是问题边界。

## 4.2 优先找因果，而不是“哪里看起来慢”

一个区域耗时大，不代表它能被消除。

更值得优先确认：

- 是否在 critical path；
- 是否真的 exposed；
- 是否有必要依赖；
- 是否只是 peer wait；
- 是否受资源冲突限制；
- 是否能改变 Product wall。

## 4.3 理论与实验互相修正

默认采用：

**Current → Model → Bound → Gap → Experiment → Update Model → Implementation → E2E → Re-bound**

但可以跳步、并行或重排。

理论负责提出高价值问题。

实验负责告诉理论哪里错了。

## 4.4 不预设最终 Runtime 形态

允许：

- 留在 vLLM-Ascend；
- 局部脱离；
- Standalone Runtime；
- Graph；
- Persistent Execution；
- Device-side State；
- Fusion；
- SuperKernel；
- 新执行结构。

也允许证明这些都不是当前最优方向。

不要为了“自研”“脱离框架”“Graph 化”本身而做。

---

# 5. P0 专项化边界

对重要优化，建议区分两类来源：

## 模型/硬件专项化

来自：

- DeepSeek V4 Flash；
- W4A8；
- 910B3；
- TP8；
- DSpark7；

这类可以成为 DeepSeek P0 的长期专项结构。

## Calibration Workload 专项化

只因为：

- 32K；
- c12；
- 当前 request/cohort；
- 当前 fixed shape；

而成立。

这类可以大胆用于求极限，但不要默认迁移到其他 workload / 模型。

---

# 6. Framework / Scheduling-only 当前阶段

当前主线优先研究 Framework / Scheduling 极限。

默认暂时不把“更快 primitive/operator”作为主要变量。

可以自由研究：

- execution order；
- Host issue；
- async queue；
- Graph organization；
- buffering；
- prefetch；
- metadata/state preparation；
- Scheduler / control plane；
- Host↔Device synchronization；
- communication/compute overlap；
- pipeline；
- request scheduling；
- Device-side state；
- persistent execution；
- Runtime restructuring。

如果新证据表明 primitive/operator 已经成为主要剩余 Gap，可以由 Sol 主动 PIVOT，不需要被当前阶段定义锁死。

---

# 7. Historical Evidence Reuse

历史 Run 的默认价值是：

> **减少重复实验，提高新假设质量。**

在启动成本较高的新实验前，优先检查已有证据是否已经回答了问题。

可以使用：

- `HANDOFF.md`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`
- TaskCtl / recovery pack
- `performance_knowledge/entries.jsonl`
- `scripts/performance_knowledge.py`
- `evidence/`

提出新候选时，最好回答：

1. 历史上是否已经研究过类似机制？
2. 旧环境 / shape / TP/DP / Graph mode 是什么？
3. 旧实验真正证明了什么？
4. 旧实验没有证明什么？
5. 当前条件发生了什么变化？
6. 如果要新跑，本次 Run 新增加的唯一信息是什么？

如果历史证据已经足够，可以直接复用。

如果条件不同，把旧结果当 hypothesis prior / counterexample，而不是当前 verdict。

如果检索成本明显高于一次低成本实验，Sol 可以自行决定直接验证。

历史复用是为了提高研究效率，不是给 Agent 增加形式主义负担。

---

# 8. Multi-agent collaboration

## Sol

Sol 是主 Agent，持续掌握项目上下文。

主要负责：

- 主线判断；
- Runtime / architecture；
- DAG / Bound / Resource Model；
- profiling 归因；
- correctness；
- benchmark 设计；
- KEEP / REJECT / PIVOT；
- 整合其他 Agent 结果。

Sol 应保持较高自主性，不需要为了“调用别的模型”而调用。

## Astra Medium

适合提供独立第二视角，例如：

- benchmark / profile 复核；
- 实验设计 challenge；
- 多个候选之间判断；
- 发现 Sol 的归因偏差；
- 路线反复或收敛变慢时重新审视。

是否调用由 Sol 根据价值判断。

## Astra High

适合少量高价值、高复杂度、高不确定性问题，例如：

- Bound 是否成立；
- 重大架构分叉；
- 多组 evidence 冲突；
- Persistent / Whole-cycle Replay；
- DSpark Graph；
- 关键 Runtime architecture；
- 一个判断错误会导致大量实验浪费的场景。

Astra 是独立分析输入，不自动覆盖 Sol。

## DeepSeek / Zcode

具体命令写法、长任务返回值和失败后验收见 ZCODE_OPERATIONS.md。

适合机械、边界清晰、容易验收的任务，例如：

- 部署；
- 启停服务；
- benchmark；
- 重复测试；
- profiling/log/trace；
- 环境检查；
- 数据提取；
- 简单源码定位；
- 简单可验收修改。

不建议让其独立裁决：

- Bound；
- 性能极限；
- 根因；
- KEEP / REJECT / PIVOT；
- 重大跨模块架构。

如果其能力在某项任务上实际表现可靠，Sol 可以扩大委派范围，但仍应按风险匹配验收强度。

---

# 9. Performance Knowledge

历史经验优先沉淀“机制”，而不是只记录 PASS / FAIL。

高价值知识建议包含：

- topic；
- mechanism；
- environment；
- observed；
- failure_or_limit；
- revalidate_when；
- extreme_relation；
- source。

目标不是建立一个僵硬数据库，而是让未来 Agent 能回答：

> “这个方向以前做过吗？为什么成功/失败？哪些条件变化后值得重验？”

历史结论只能作为先验。

新场景仍由当前证据裁决。

---

# 10. Benchmark / Correctness judgment

建议始终区分：

- `formal_e2e`
- `matched_internal_ab`
- `diagnostic`
- `profile`
- `microbenchmark`
- `simulation`
- `theoretical_bound`
- `independent_review`

一个结果属于哪一类，比 Run 编号更重要。

Correctness 也不要求所有问题都使用同一种 gate。

如果底层存在 replay / numerical nondeterminism，可以根据实际问题使用：

- same-state A/B/A；
- A/B/C；
- self-replay noise floor；
- logit margin；
- acceptance/count parity；
- state/KV parity；
- downstream consumer parity。

Sol 可以选择最能回答当前问题的验证方式。

---

# 11. Context efficiency

不要默认每次通读整个仓库。

推荐先从：

1. `HANDOFF.md`
2. 当前状态文件
3. TaskCtl / recovery pack

恢复。

然后围绕当前问题打开必要 evidence。

原始 trace / profile / 大日志不需要一直塞进高级模型上下文。

优先保留：

- 当前结论；
- 关键数字；
- causal evidence；
- 当前 Bound；
- 最大 unknown；
- 已否定假设；
- next hypothesis。

如果深挖原始 evidence 能明显提高判断质量，应直接读取，不要为了节省 token 而牺牲正确性。

---

# 12. State persistence

建议持续维护：

- `HANDOFF.md`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`

大致职责：

### `HANDOFF.md`

当前现场和新会话恢复入口。

### `PROJECT_STATE.md`

项目进展和历史状态。

### `PERFORMANCE_MAP.md`

当前 bottleneck / critical path / opportunity map。

### `ACHIEVABLE_BOUND.md`

当前 Bound、假设、区间、unknown。

### `RESULTS.md`

正式保留的 correctness / performance 结果。

这些文件应帮助 Agent 恢复，而不是成为写报告负担。

更新粒度由 Sol 根据“是否产生新的可复用事实”决定。

---

# 13. Document stability

以下文档建议相对稳定：

- `AGENTS.md`
- `MISSION.md`
- `FOUNDRY_METHOD.md`

不要因为每个 Run 都修改。

只有当：

- 项目目标变化；
- Agent 协作方式变化；
- 证据纪律变化；
- Foundry Method 被实践证明需要升级；

时再修改。

状态变化优先写入：

- `HANDOFF.md`
- `PROJECT_STATE.md`
- `PERFORMANCE_MAP.md`
- `ACHIEVABLE_BOUND.md`
- `RESULTS.md`

---

# 14. Stop / Continue judgment

不要因为：

- 一个 Run 完成；
- 一个 commit 完成；
- 超过 Stock；
- 达到某个局部百分比；

就默认项目完成。

也不要因为文档写了某个 next_action 就机械继续。

每个重要 checkpoint 后，Sol 可以重新判断：

- 当前最大 Gap 是什么；
- 最大 unknown 是什么；
- 原路线是否仍然值得；
- 是否出现更高价值候选；
- 是否应该重构问题本身。

如果没有真实 blocker，并且存在明显高价值方向，可以继续。

如果继续实验的价值已经很低，也可以主动停止或换阶段。

---

# 15. What success means

DeepSeek P0 最终希望同时得到：

## A. Product result

一个：

- 正确；
- 可运行；
- 明显更接近当前 Achievable Bound；

的 DeepSeek Extreme Runtime。

## B. Foundry result

能够解释：

- Current；
- Execution DAG；
- Necessary Work；
- Resource Model；
- Optimistic Bound；
- Achievable Bound；
- Gap；
- 为什么选择某个 Intervention；
- E2E 结果；
- Re-bound；
- 哪些知识可迁移到下一 workload / 模型。

---

# 16. One-line principle

> **Inference Foundry 不要求 Agent 按固定流程优化；它要求 Agent 在保持 correctness、证据真实性和可比性的前提下，自主寻找最有价值的路径，把 Current 与可实现性能边界之间的 Gap 持续缩小。**
