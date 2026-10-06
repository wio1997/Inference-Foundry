# GLM Agent研究分工与执行策略

2026-10-06，从checkpoint158（父commit `e378d459e08e9f12300118e158bba16db6c4d04f`）之后生效，替代旧日常分工与研究优先级。历史Run、raw evidence、模型调用记录及裁决保持原样。[局部AGENTS](../../AGENTS.md)是研究规则入口；本文不切换当前会话模型、不启动服务/实验、不实现自动模型路由。

启动按[最小恢复顺序](../START_NEW_CHAT.md#恢复顺序)；本文仅在分工、Astra Review或执行委派时读取。

## 默认分工

| 角色 | 产物与边界 |
|---|---|
| GPT-6.1 Sol high：主研究Agent / 性能架构师 | 长期负责目标对齐、最大性能Gap、因果分析、架构、关键源码、Runtime / Scheduler复杂重构、实验设计、matched A/B及KEEP / REJECT / Current最终裁决 |
| Astra：独立Challenger | 只在下列触发条件下做独立路线/架构Review；可以挑战或否定Sol，不参与日常Run、不常驻、不做每轮审批 |
| Zcode / DeepSeek与现场脚本/controller：执行层 | 服务器操作、服务启动/停止/恢复、benchmark、monitor、profile采集、大日志/trace归约、artifact身份核验及范围明确的机械修改；不决定长期性能主线 |

Sol high可以长期持有任务。HANDOFF、历史Run、Zcode Summary及上一轮next_action是证据，不是必须继承的路线。允许推翻历史假设、停止无Product Gain的高投入方向、质疑代码顺序与框架边界、删除不必要通用Runtime层、改变调度/状态组织/执行顺序、大幅重构控制层；不得产生sunk-cost bias。medium/ultra的旧分工不再构成固定研究角色或接管主线的规则。

## Astra Challenger Review

以下任一情况触发独立Review，主Agent在当前优化点/checkpoint记录具体信号；不要求每Run复核或以多模型一致裁决性能：

1. 重大架构分叉，例如固定PD / Full Replica / Elastic PD。
2. 连续较多有效实验没有新增代码级KEEP。
3. MTP / PD / KV / Graph / Scheduler / Communication跨模块根因长期无法闭环。
4. 准备高成本、大范围Runtime重构。
5. Current长时间没有明显提升。
6. Sol怀疑当前路线已经偏离最大性能Gap。

按问题给Astra关键源码及消费者、关键diff、profile/trace、matched Run与决定性raw evidence的可访问位置和身份，尽量让其直接读取；HANDOFF只作索引，不能只读HANDOFF后复述现有结论。缺证据或缺源码访问写unknown，由执行层提取必要原始片段；摘要不能补成已验证根因。给事实与反例，避免把Sol路线结论预设成Review答案。

Astra独立回答：

1. 当前最大的可消除性能Gap是什么？
2. 当前主线是否值得继续？
3. 有没有更高价值的执行架构优化？
4. 哪些历史方向应该停止？
5. 如果今天根据现有证据重新开始，会选择什么路线？

结果区分事实、推断、unknown，指向代码路径/证据。Sol亲查相关真实代码与完整E2E，最终裁决PERF_KEEP / REJECT / Current；Review意见不是性能KEEP。Astra暂不可用如实记录，先完成Sol的Goal Review与证据研究，不无限重复低价值Run，不假称已审查。

### 固定Review Package与输出

Sol不复制整个历史聊天，只给紧凑输入包：

1. 最终产品。
2. 当前可信Current（产品代码、active stack/合同/workload/E2E引用，缺失写unknown）。
3. 当前critical path。
4. 最大Gap假说。
5. 关键源码入口。
6. 当前patch / diff（尚无patch明确写无）。
7. 最重要profile / trace。
8. 最重要matched Run。
9. 2～3个已否定解释（不足如实写unknown，不发明反例）。
10. 当前可选架构/方向。

Astra必须输出：**最大Gap排名；当前主线KEEP / STOP / PIVOT建议；Top 3最高价值代码问题；明确Stop List；最小区分性实验；证据中最可能被误读之处**。这里路线KEEP不是代码PERF_KEEP。Sol记录**接受哪些 / 拒绝哪些 / 为什么**，随后在唯一主假说与Reset中体现决定。Astra不成为第二个长期主Agent，Top 3不是同时激活的候选，也不能生成无限探索队列。

## Zcode执行层与Sol直接研究责任

Sol定义代码问题、假说、服务/Run合同和验收；Zcode在边界内执行、采集和归约，给有证据的条件结论。涉及**最大Gap、根因、架构、关键代码优化或KEEP**，Sol必须亲自阅读必要源码、关键diff、profile和决定性原始证据，不能只依赖Zcode Summary。大raw留现场，主Agent定向读关键片段；不需要接收或重复归约全部日志。

Zcode接到新真实性能Run前须有Sol的Reset/唯一假说与Hypothesis、Distinguishing evidence、Decision table引用；缺少时返回needs_decision，不自行补成研究路线或批量扫描。大型正式E2E等待诊断支持、correctness通过、patch值得产品裁决；共享controller不把未来候选批量入队。规则维护本身不调用现场服务/启动性能实验。

按[Job/Result协议](../ZCODE_PROTOCOL.md)交接；已有subagent如使用同样协议，不能替代Sol研究责任。共享服务/源码/设备由唯一controller排程，离线只读归约可独立。Zcode不可用或委派成本超过工作本身时直接用现成脚本，保留同样的真实性与归约要求。

- `zcode --prompt`执行现成shell任务时，prompt是一条完整命令；复杂执行复用现场controller。机械修改给输入、产物、允许范围和验收，不能把架构/长期路线委派成执行决定。CLI路径、有效model id和现场身份按实际环境核验，不用旧PID盲操作。
- 服务启动/停止/恢复由执行层处理；记录必要实例/进程、就绪/退出及活动资源证据。外层返回0不证明内层实验成功或服务就绪。
- 长测试/monitor由真实驻留脚本运行。完成、异常、阈值越界或需要决策才返回，健康采样落盘；不靠模型持续对话或不断调用Zcode轮询。
- 重试保留原因与已有Run，不覆盖raw，不把监控轮询算新性能Run；退出/交接记录真实运行和待处理状态。

## 返回与工程成本

返回紧凑Result：实际优化点/Run、执行状态与code/config身份；相关指标/异常、窗口/过滤/rank覆盖；artifact稳定位置、必要hash/样本locator及归约方式；需要Sol决策的具体问题。事实、推断和unknown分开，缺失不填0，执行真实性不等于性能KEEP。

PID/hash/epoch/controller/artifact/ownership只保留支持真实性、资源安全和可复现的必要信息；不会影响性能归因、correctness或KEEP的记录不扩张为主线。初始化、模型调用、返工与恢复成本用于选择实验方式，不能包装成E2E Product Gain；不另建模型评测或复杂自动路由系统。

## 研究规则引用

选择代码问题与架构见[PLAN](../../PLAN.md#工程候选与成本)；PERF_KEEP、matched A/B、完整E2E、噪声与stack判据见[RECORDING](../../RECORDING.md#4-提交与裁决)。[Goal Review阈值与五问](../../PLAN.md#goal-review与checkpoint)、[双轨/模块约束](../../RECORDING.md#7-双轨backlog模块约束与完成)、[阶段完成](../../PLAN.md#阶段性完成条件)各自维护，本文不复述。Review结果必须回到唯一主假说与Reset，不自动激活多个方向。
