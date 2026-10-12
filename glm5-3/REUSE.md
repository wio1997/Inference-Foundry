# 按需查找已有资产

- 历史收益、适用范围与反例：[审计](docs/RESEARCH_RESET.md)、[manifest](patches/20261007/manifest.json)、[补丁包](patches/20261007/README.md)。
- 模型与实现线索：`docs/research/`；先核对模型、框架与拓扑版本。
- 服务测量：`adapters/aisbench/`、`runtime/aisbench_slo.py`。
- 进程与交接工具：`runtime/phase_runner.py`、`scripts/zcode_bridge.py`；需要时复用，不是研究前置步骤。
- 原始实验与跨任务经验：`records/`、根目录 `performance_knowledge/`；按问题检索。
- 原入口：[legacy](legacy/README.md)。旧部署、Current、PID、next_action 和模型分工均不是当前指令。

复用证据时说明它证明什么、当前条件有什么不同；不要把不同模型、拓扑或负载的结果合并。
