# GLM Extreme — AGENTS

**Rule version：`GLM-RESEARCH-RULES-v2`**。checkpoint158之后生效；本次仅精简入口，研究门槛不变。rule commit按[启动核验](docs/START_NEW_CHAT.md#0-先核验branch--head--rule-version)从Git定位，不固定旧SHA。本文件是规则入口，链接细则同样有效；与旧方案、HANDOFF或next_action冲突时，以这里的当前规则为准。

目标：按用户2026-10-06明确约束，在现有两台服务器、现有正确算子下，构建功能完整的**标准P/D分离**GLM推理框架并逼近性能极限。P池与D池分别组织DP/TP/EP；例如各角色DP1/TP16或DP2/TP8，实际fit与正确性须核验。Full Replica / Complete-request Placement不满足本产品目标；历史现役事实保留，不作为后续产品路线。不得通过D本地完整请求fallback减少PD覆盖来验收功能或性能。不改算子计算实现、不开发新kernel、不做5.2/5.3精度或量化质量比较。完整合同见[MISSION](MISSION.md#性能与功能合同)与[目标纠正裁决](records/points/GLM-OPT-0002/research/engine_commit_20261006/PD_SCOPE_CORRECTION.md)。

- **GPT-6.1 Sol high是性能架构师**，负责最大Gap、因果、架构、关键代码、实验与最终裁决；必须亲读必要源码/diff/profile/决定性raw。Astra按需独立Challenger，Zcode/DeepSeek负责执行；[分工与触发条件](docs/research/AGENT_MODEL_STRATEGY.md)，不实现自动模型路由。
- 历史Run、raw、裁决和源码保留；HANDOFF/Summary/next_action只作证据。允许推翻假设、停止无Product Gain路线、删除不必要Runtime层和大幅重构控制层，不因历史投入继承路线。
- **代码优化优先**；配置/部署只用于baseline、验证假说与资源辨因，禁止无限扫描。标准PD含P Prefill→KV Transfer→D Decode；各副本完整Prefill+Decode须标Full Replica / Complete-request Placement。
- 默认**1个active性能假说 + 1个区分diagnostic**，切换先关闭/park旧方向；每个真实性能Run前有Hypothesis / Distinguishing evidence / Decision table。A/B不改变决定则不执行；大型E2E须先有诊断支持、correctness通过和产品裁决价值。
- **correctness + matched A/B + 可重复真实完整E2E Product Gain**才可PERF_KEEP；无完整E2E不提升Current。配置/部署/代码分账，累计stack收益重测不相加；只有代码PERF_KEEP算主要性能进展，具体比较条件见[RECORDING](RECORDING.md#4-提交与裁决)。
- 恢复现场只读，保护现役服务；共享资源由唯一controller排程。PID/hash/epoch等只记录真实性、安全、复现和裁决所需；不扩张为研究主线。

## 0. 启动身份与Performance Research Reset

按[最小恢复顺序](docs/START_NEW_CHAT.md#恢复顺序)定向读取，不递归通读链接。身份/现场未知如实列缺项；定位当前产品与一个最大Gap后即可进入相关源码和证据分析。新会话或强制Goal Review后，任何GPU/NPU性能Run前须输出[Reset](docs/START_NEW_CHAT.md#首次实际研究输出performance-research-reset)；Gap证据不足时只做一个最小补证据diagnostic/profile，再更新Reset。已有证据足够无需重跑，规则维护不启动实验。

## 8. 强制Goal Review

连续5个有效性能Run无代码KEEP、3个candidate无Product Gain、3个INVALID/driver failure，或模块/方向扩散、参数扫描主导、非目标问题长期占主线、最大Gap不清时，暂停惯性Run。完整阈值和五问仅维护在[PLAN](PLAN.md#goal-review与checkpoint)；可提前触发，不值得继续立即停止，必要时Astra Review，之后重新Reset。

## 按动作读取细则

| 当前动作 | 只读相应细则 |
|---|---|
| 选择代码问题 / 架构 | [PLAN：工程候选与成本](PLAN.md#工程候选与成本)，相关源码与证据由[RECOVERY_INDEX](RECOVERY_INDEX.md)定位 |
| 准备Run / 写证据 | 昂贵执行前查[REUSE](REUSE.md#候选检查)；[RECORDING：Reset与决策价值](RECORDING.md#入口reset与run决策价值)、[最小证据](RECORDING.md#3-run最小证据) |
| PERF_KEEP / Current / stack裁决 | [RECORDING：比较标准](RECORDING.md#4-提交与裁决)、[stack与Current](RECORDING.md#6-当前stack交互回归与产品current) |
| 功能阻塞 / 新增Runtime模块 | [RECORDING：双轨与模块约束](RECORDING.md#7-双轨backlog模块约束与完成) |
| 委派 / 服务操作 | [模型策略](docs/research/AGENT_MODEL_STRATEGY.md#zcode执行层与sol直接研究责任)、[Job/Result协议](docs/ZCODE_PROTOCOL.md) |
| checkpoint汇报 | [账本汇报规则](CODE_PERFORMANCE_LEDGER.md#重要checkpoint汇报)；无新增时写“本阶段没有新增代码级性能 KEEP。” |

## 12. 阶段性完成条件

准备交付时读[PLAN：阶段性完成条件](PLAN.md#阶段性完成条件)。Run数、commit数、文档/模块数不评价研究质量；评价可重复E2E Product Gain、Current提升和剩余真实性能Gap。Run失败或预算耗尽不等于完成。
