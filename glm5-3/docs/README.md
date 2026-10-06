# 按需资料索引

研究资料用于查事实与来源，不是启动上下文。实际运行事实以HANDOFF/Run为准；研究路线以checkpoint158之后的[AGENTS](../AGENTS.md)为准，旧候选/next_action不绑定后续。下面公开main和0.27源码不是用户安装确认。

| 当前问题 | 只读相关资料 |
|---|---|
| 新会话路径 / 当前产品版本 | [RECOVERY_INDEX](../RECOVERY_INDEX.md) / [CURRENT_PRODUCT_MAP](../CURRENT_PRODUCT_MAP.md)，按[恢复顺序](START_NEW_CHAT.md#恢复顺序)进入 |
| 极限范围、L/U、动态容量、多配置 | [完整逻辑](research/MULTINODE_PD_SCHEDULING_LIMIT_PLAN.md) |
| GLM层/专家、MLA/DSA、Full/Shared、MTP | [模型分析](research/GLM5_2_MODEL_AND_RUNTIME_ANALYSIS.md) |
| V1/V2、MTP、PD/Graph/量化代码入口 | [源码索引](research/GLM5_2_VLLM_ASCEND_PATHS.md) |
| 动态状态、slot/KV/Graph/PD契约 | [Runtime候选设计](research/GLM_SPECIALIZED_RUNTIME_DESIGN.md) |
| profiling入口、0.27参数、多实例与观察器 | [Profiler核验](research/VLLM_ASCEND_PROFILING.md) |
| 原12份MD职责与冲突 | [迁移映射](research/ORIGINAL_MD_MIGRATION_MAP.md) |
| Sol high性能架构师、Astra独立Challenger、Zcode执行层 | [模型策略](research/AGENT_MODEL_STRATEGY.md) |
| 代码KEEP、matched A/B、完整E2E与Goal Review | [研究规则](../AGENTS.md) / [记录规范](../RECORDING.md) / [代码性能账本](../CODE_PERFORMANCE_LEDGER.md) |
| 当前active PERF_KEEP、交互回归、完整stack累计Gain | [CURRENT_PERFORMANCE_STACK](../CURRENT_PERFORMANCE_STACK.md) |
| Zcode/subagent格式、服务/监控与大日志归约 | [Job/Result协议与桥接工具](ZCODE_PROTOCOL.md) |
| 开始执行、新对话交接与本地模拟 | [启动指令与演练](START_NEW_CHAT.md) |
| 全部讨论的背景 | [完整方案快照](research/GLM5_3_W8A8_PD_PLAN.md) |
| 固定来源与hash | [Source audit](research/SOURCE_AUDIT.json) |

原Foundry方法、TaskCtl和知识库继续引用根目录，不复制其历史Current。文档中的对象/候选名是建议，执行Agent可以改变路线。

v2研究入口先核验branch/HEAD/最新rule version与rule commit/active point；新会话或强制Goal Review后先Reset，再进入性能Run。默认一个active假说加一个区分diagnostic，Run前决策价值、PERF_KEEP噪声/stack门槛、5/3/3等Goal Review阈值和阶段完成条件均以AGENTS为准；旧研究方案的组合/并行建议不能覆盖这些门槛。
