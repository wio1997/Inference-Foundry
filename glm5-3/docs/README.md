# 按需资料索引

研究资料用于查事实与来源，不是启动上下文。实际运行事实以HANDOFF/Run为准；研究路线以checkpoint158之后的[AGENTS](../AGENTS.md)为准，旧候选/next_action不绑定后续。下面公开main和0.27源码不是用户安装确认。

当前目标已切换到GLM-5.3 W8A8标准PD，权重`/data/tiankuan/wio/GLM-5.3-w8a8`，新服务名`glm-53`；用户2026-10-06上传中，完成并核验前不加载/测试。旧模型分析、SOURCE_AUDIT与Run是历史资料，不是5.3结构、支持或收益证明。

| 当前问题 | 只读相关资料 |
|---|---|
| 新会话路径 / 当前产品版本 | [RECOVERY_INDEX](../RECOVERY_INDEX.md) / [CURRENT_PRODUCT_MAP](../CURRENT_PRODUCT_MAP.md)，按[恢复顺序](START_NEW_CHAT.md#恢复顺序)进入 |
| 极限范围、L/U、动态容量、多配置 | [完整逻辑](research/MULTINODE_PD_SCHEDULING_LIMIT_PLAN.md) |
| GLM-5.3层/专家、MLA/DSA、Full/Shared、MTP | [当前模型核验](research/GLM5_3_MODEL_AND_RUNTIME_ANALYSIS.md) |
| GLM-5.3 V1/V2、MTP、PD/Graph/量化代码入口 | [当前源码入口](research/GLM5_3_VLLM_ASCEND_PATHS.md) |
| 动态状态、slot/KV/Graph/PD契约 | [Runtime候选设计](research/GLM_SPECIALIZED_RUNTIME_DESIGN.md) |
| profiling入口、0.27参数、多实例与观察器 | [Profiler核验](research/VLLM_ASCEND_PROFILING.md) |
| 原12份MD职责与冲突 | [迁移映射](research/ORIGINAL_MD_MIGRATION_MAP.md) |
| Sol high性能架构师、Astra独立Challenger、Zcode执行层 | [模型策略](research/AGENT_MODEL_STRATEGY.md) |
| 代码KEEP、matched A/B、完整E2E与Goal Review | [研究规则](../AGENTS.md) / [记录规范](../RECORDING.md) / [代码性能账本](../CODE_PERFORMANCE_LEDGER.md) |
| 当前active PERF_KEEP、交互回归、完整stack累计Gain | [CURRENT_PERFORMANCE_STACK](../CURRENT_PERFORMANCE_STACK.md) |
| Zcode/subagent格式、服务/监控与大日志归约 | [Job/Result协议与桥接工具](ZCODE_PROTOCOL.md) |
| 开始执行、新对话交接与本地模拟 | [启动指令与演练](START_NEW_CHAT.md) |
| 当前GLM-5.3标准PD方案 | [方案入口](research/GLM5_3_W8A8_PD_PLAN.md) / [PLAN](../PLAN.md) |
| 历史模型讨论与固定来源 | [旧方案](research/GLM5_2_PD_PLAN_REFERENCE_20261001.md)、[旧模型分析](research/GLM5_2_MODEL_AND_RUNTIME_ANALYSIS.md)、[旧源码索引](research/GLM5_2_VLLM_ASCEND_PATHS.md)、[历史Source audit](research/SOURCE_AUDIT.json) |

原Foundry方法、TaskCtl和知识库继续引用根目录，不复制其历史Current。文档中的对象/候选名是建议，执行Agent可以改变路线。

启动范围与读取时机统一见[AGENTS](../AGENTS.md)和[START_NEW_CHAT](START_NEW_CHAT.md#恢复顺序)，本索引不复述研究门槛。
