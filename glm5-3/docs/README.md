# 按需资料索引

研究资料用于查事实与来源，不是启动上下文。实际运行事实以HANDOFF/Run为准；下面公开main和0.27源码不是用户安装确认。

| 当前问题 | 只读相关资料 |
|---|---|
| 极限范围、L/U、动态容量、多配置 | [完整逻辑](research/MULTINODE_PD_SCHEDULING_LIMIT_PLAN.md) |
| GLM层/专家、MLA/DSA、Full/Shared、MTP | [模型分析](research/GLM5_2_MODEL_AND_RUNTIME_ANALYSIS.md) |
| V1/V2、MTP、PD/Graph/量化代码入口 | [源码索引](research/GLM5_2_VLLM_ASCEND_PATHS.md) |
| 动态状态、slot/KV/Graph/PD契约 | [Runtime候选设计](research/GLM_SPECIALIZED_RUNTIME_DESIGN.md) |
| profiling入口、0.27参数、多实例与观察器 | [Profiler核验](research/VLLM_ASCEND_PROFILING.md) |
| 原12份MD职责与冲突 | [迁移映射](research/ORIGINAL_MD_MIGRATION_MAP.md) |
| 模型成本与按需升级 | [模型策略](research/AGENT_MODEL_STRATEGY.md) |
| 全部讨论的背景 | [完整方案快照](research/GLM5_3_W8A8_PD_PLAN.md) |
| 固定来源与hash | [Source audit](research/SOURCE_AUDIT.json) |

原Foundry方法、TaskCtl和知识库继续引用根目录，不复制其历史Current。文档中的对象/候选名是建议，执行Agent可以改变路线。
