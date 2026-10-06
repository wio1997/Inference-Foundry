# GLM Extreme

为GLM构建完整功能的专用推理框架，研究现有算子下多机多卡、动态负载及可变PD的调度极限。当前用GLM-5.2、两台910C；版本/资源口径见MISSION。

**从checkpoint158之后以[AGENTS](AGENTS.md)恢复研究规则，再读HANDOFF+当前活动点/Git恢复事实。** HANDOFF/历史Run/Summary/next_action仅作证据。Sol high是性能架构师，Astra按需独立Challenger，Zcode / DeepSeek是执行层；恢复后先检查Goal Review，不机械继承旧路线。

当前规则`GLM-RESEARCH-RULES-v2`在`df7399c28791`上增量完善。研究启动先核验实际branch / HEAD / 最新rule version与rule commit并报告active point；不匹配先恢复，不能旧main研究。新会话或强制Goal Review后，第一项GPU/NPU性能Run前先Performance Research Reset；只留一个主假说加一个区分diagnostic，每个Run先证明决策价值，Gap未知只做一个最小补证据动作。

| 入口 | 按需用途 |
|---|---|
| [MISSION](MISSION.md) / [PLAN](PLAN.md) | 目标/合同、极限标准与逼近逻辑 |
| [RECORDING](RECORDING.md) / [优化点索引](records/points.jsonl) | matched A/B、完整E2E、Git提交与不可改写的Run证据/裁决 |
| [代码性能账本](CODE_PERFORMANCE_LEDGER.md) | 代码候选Baseline/Patched/Gain/Correctness/Verdict，代码KEEP进展 |
| [当前性能stack](CURRENT_PERFORMANCE_STACK.md) | 当前active PERF_KEEP patch、独立Gain、完整stack收益与交互回归 |
| [REUSE](REUSE.md) / [复用索引](records/reuse.jsonl) | 旧成果适用性、失败原因与去重 |
| [研究索引](docs/README.md) | GLM特性、源码、prof、历史迁移与模型成本 |
| [新对话启动](docs/START_NEW_CHAT.md) | 短交接指令、模拟演练与实际环境接入 |

交给执行Agent的短prompt：

```text
在glm5-3/按当前AGENTS.md接续checkpoint158之后的研究，HANDOFF和next_action仅作证据。Sol high作为性能架构师先做Goal Review，亲读必要源码/diff/profile/raw，选择最大可信代码Gap；Astra按需独立挑战，Zcode只做执行。代码优化经Correctness、matched A/B与真实完整E2E重复裁决，配置/部署/代码分账，无完整E2E不提升Current。维护代码性能账本，无新增时写“本阶段没有新增代码级性能 KEEP。”保留所有历史Run/raw/裁决，当前两台可行域内收敛。
先核验研究branch/HEAD/最新AGENTS版本与rule commit并报告active point；新会话/强制Review后先按START_NEW_CHAT输出Performance Research Reset，第一项GPU/NPU性能Run前完成。默认一个active假说加一个区分diagnostic，每Run写Hypothesis/Distinguishing evidence/Decision table。按Type记账，仅PERFORMANCE满足噪声感知matched与完整E2E得到PERF_KEEP，维护active stack并重测累计Gain；按AGENTS硬阈值Goal Review，功能问题分lane/backlog，按阶段完成条件交付。
```

当前事实由HANDOFF与points.jsonl恢复（checkpoint158活动点为GLM-OPT-0002，Current=None）。本次规则更新没有新增代码级性能KEEP或GPU/NPU实验。标准PD包含P Prefill→KV Transfer→D Decode；各副本完整Prefill+Decode必须标Full Replica / Complete-request Placement。

代码量不代表性能进展，Current必须连同产品代码commit、active stack、功能合同、标准workload、E2E与剩余Gap恢复；阶段完成标准见AGENTS，不把Run跑不动当完成。
