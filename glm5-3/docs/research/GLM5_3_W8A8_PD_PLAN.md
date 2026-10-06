# GLM-5.3 W8A8标准PD方案入口

当前执行方案唯一维护在[PLAN](../../PLAN.md#当前执行方案标准pd优化)，目标与约束见[MISSION](../../MISSION.md)及[AGENTS](../../AGENTS.md)。

- 两机目标权重：`/data/tiankuan/wio/GLM-5.3-w8a8`；新服务模型名：`glm-53`。
- 2026-10-06用户正在上传；只读确认两机独立目录存在，检查时尚无config.json。上传完成、index引用分片、tokenizer、量化配置、安装支持与实际加载均待核验。
- 标准链路：P Prefill→KV Transfer→D Decode/MTP→有效输出。P/D分别组织DP/TP/EP；参考DP1TP16PP1不是新模型fit或最优证明，DP2TP8只在证据支持时进入角色内对照。
- 完整功能、有效计数、SLO和算子边界保持。先定位一个真实PD关键路径Gap，证据支持后做最小native patch，再correctness、matched重复完整E2E裁决。
- 上传期间不加载、不启动模型或测试；不回退旧权重。当前没有GLM-5.3 Current、PERF_KEEP或完整E2E基线。

模型资料见[GLM-5.3模型核验](GLM5_3_MODEL_AND_RUNTIME_ANALYSIS.md)，源码入口见[GLM-5.3运行路径](GLM5_3_VLLM_ASCEND_PATHS.md)。此前以旧权重完成的讨论保留为[历史方案](GLM5_2_PD_PLAN_REFERENCE_20261001.md)，其结构、版本支持、功能和收益不能直接赋给5.3。
