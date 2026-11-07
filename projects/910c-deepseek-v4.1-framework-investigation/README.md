# DeepSeek V4.1 Flash 推理框架算子优化排查

本目录是持续更新的排查档案。固定 DeepSeek V4.1 Flash W8A8、DSpark、Engram、CANN 9.2、910C 及既定 C8 负载，以实际推理顺序组织算子优化分析。整体任务仍在进行。

远端唯一发布位置：`wio1997/Inference-Foundry` 的 `910c-deepseek-v4.1-flash` 分支，`projects/910c-deepseek-v4.1-framework-investigation/`。该目录记录本次 171 分析，不覆盖同分支已有的 172 DeepInfer 结果或冻结 vLLM 基线。

| 入口 | 内容 |
|---|---|
| [EP16 专家布局的冻结 C8 对照](baselines/ep16-interval-c8.json) | 首 token、整批、自然命中、数值检查和原服务恢复 |
| [按专家布局重算矩阵计算下限](baselines/ep16-compute-floors.json) | 原布局、每层调入两个专家和自由分配在相同矩阵工作下的下限 |
| [QLI 在批量 slot 基线上的增量](baselines/qli-batched-slot-c8.json) | 171 同基线前后参照，输出一致性、首 token、整批耗时和恢复 |
| [EP16 专家布局的实际命中与所需收益](baselines/ep16-layer-guard.json) | 两轮参照的传递门槛、一次零命中的批次差额、诊断轮的覆盖 |
| [单算子与联合范围的价值初筛](baselines/fusion-priority.json) | 局部占比、请求级上界、无需单独深查的范围和重新打开条件 |
| [末层 FFN 原系数数值对照](baselines/terminal-dense-coefficients.json) | 保持真实路由权重后，逐段核对专家、mHC 和读出 |
| [末层 FFN 分支收益预算](baselines/terminal-guard-budgets.json) | 哪些 DP 分支必须同时覆盖、不同批次替代成本怎样分摊 |
| [四层 QLI 三组服务对照](baselines/qli-service-c8.json) | 首 token、整批完成时间、各组样本与原服务恢复 |
| [四层 QLI 模型内实测](baselines/qli-model-window.json) | 原版与修正版短窗口、实际满批命中和整批收益判定 |
| [融合方案怎样快速筛选](baselines/decision-workflow.md) | 按可消除的整份工作归并，只补可能改变取舍的证据 |
| [逐请求的时间预算](baselines/graph-request-prefixes.json) | 58 类目标图任务在每个客户端截止时点前的总量，可直接查询联合范围 |
| [请求终点与融合筛选](baselines/c8-endpoint-screen.json) | 两个服务指标、必须提前的请求、实际命中次数和整组可省时间 |
| [共用测量目录](baselines/shared-measurements.json) | 通信、计算、搬运和调度的可复用实测条件与集中补证条件 |
| [完整排查报告](REPORT.md) | 固定范围、执行顺序、价值筛查、实验裁决、理论边界和重要缺口 |
| [模块分析](modules/README.md) | M00–M11 的分支、改写空间、证据、处置和下一步 |
| [最终 FFN 与词表行需求](baselines/terminal-sparsity.json) | 171 功能对照、按原行归属的紧凑投递、替代成本上限 |
| [八请求批量价值筛查](baselines/request-region-screen.json) | 16 种改写范围、各自的首 token P90 上界、会改变裁决的假设 |
| [共享单位成本速查](baselines/unit-costs.md) | 真实消息尺寸在两组 TP8 的单组和并发延迟、跨模块复用和集中补测条件 |
| [统一成本基线](baselines/README.md) | 已有频次/成本统一查询、通信等公共缺口、集中补测与停止规则 |
| [进度状态](progress.json) | 当前模块、最新结果、未完成范围和恢复入口 |
| [固定范围与版本](scope.json) | 模型配置、硬件/负载、选中源码的精确哈希及绑定限制 |
| [分析日志](analysis-log.jsonl) | 可审查的问题、观察、判断依据、改进和后续动作；追加保存 |
| [证据索引](evidence-index.json) | 原始位置、内容哈希、来源 Git 版本和证据适用范围 |
| [维护约定](MAINTENANCE.md) | 更新顺序、状态口径、发布方法和完成标准 |

原始 profile、模型、编译缓存和完整日志留在现有服务器/分析库，不重复上传。这里保存可阅读报告、小型结构化摘要和精确证据引用，不保存凭证。
