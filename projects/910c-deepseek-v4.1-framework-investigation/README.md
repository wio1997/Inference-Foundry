# DeepSeek V4.1 Flash 推理框架算子优化排查

本目录是持续更新的排查档案。固定 DeepSeek V4.1 Flash W8A8、DSpark、Engram、CANN 9.2、910C 及既定 C8 负载，以实际推理顺序组织算子优化分析。整体任务仍在进行。

远端唯一发布位置：`wio1997/Inference-Foundry` 的 `910c-deepseek-v4.1-flash` 分支，`projects/910c-deepseek-v4.1-framework-investigation/`。该目录记录本次 171 分析，不覆盖同分支已有的 172 DeepInfer 结果或冻结 vLLM 基线。

| 入口 | 内容 |
|---|---|
| [压缩 Attention 的首段 FP32 交付](smla-partial-handoffs.md) | 首块直写、KV 面板净流量、长提示首 token 与普通请求的不同收益 |
| [从热点定位实际可省的算子工作](profile-boundary-audit.md) | 96 个局部热点签名、固定 MoE 布局的对照、Attention 元数据发布与真实读者 |
| [四条条件提案的请求收益](four-mixed-fusion-edges.md) | 两层 Engram 合包、QLI 分数及八层 token 归并的收益门槛、草稿 OB 补齐、同波专家 GMM |
| [融合收益归因与混合提案处置](fusion-attribution.md) | 37 条混合提案的复用、归档与条件项，以及普通 slot 的设备收益预算 |
| [按模型执行顺序的价值表](module-value-map.md) | 12 个模块分别筛查环节价值、融合算子、适用分支和剩余空间 |
| [B1–B4 批写与最晚参与卡](slot-batch-by-population.md) | 13/26 项的实际分布，以及 Attention 和 slot 在普通请求的 153 个目标图波次上能否共同过线 |
| [普通请求的输入与 slot 批写](ordinary-input-and-slot.md) | 固定通信的融合价值、真实 B1 与 24 行图、13/26 项分支及收益门槛 |
| [Attention 融合的同批估算](joint-attention-frontiers.md) | 四项联合、QLI 净增量、普通代码首 token 与整批收益 |
| [融合候选与预计收益](fusion-candidates.md) | 具体算子、适用分支、普通请求与 C8 收益、批量排除及剩余范围 |
| [按执行顺序查询分支台账](branch-coverage.md) | 12 个模块、21 个已登记分支族、360 条历史语义记录，以及每组对应的当前证据 |
| [图内外词嵌入、Engram 与请求估算](baselines/embedding-engram-fusion.json) | 双查表的读者、跨卡等待和普通请求的样本差异 |
| [三次检查点归集与草稿选行的联合估算](baselines/dspark-checkpoint-packet.json) | 原生均值、三次归集与按行交付 |
| [末层输出到词表投影的实测区间和请求估算](baselines/terminal-head-delivery.json) | raw 读者、分片行数与连续区间上限 |
| [路由分数、TopK 与分派入口](baselines/router-score-delivery.json) | 路由分数、同代 ID、关键路径与一次主机晚提交 |
| [可选路径与保留完整 logits 的选词](optional-paths-and-raw-head.md) | 最后四条旧候选与输入缓冲、草稿全宽 RMS 的联合估算 |
| [C2 投影、池化和选择到 Attention](c2-selection-pipeline.md) | 固定排程可省的直接交付、按查询分块的额外收益与条件 |
| [首层四路展开和两层 Engram](initial-hc-engram.md) | 两处残差读者、哈希与投影中间面、真正制约 TP8 的 padding 交付 |
| [草稿状态与下一轮输入](draft-state-handoffs.md) | 验证与五步草稿 ID、下一轮状态、计数等待和六个普通请求的估算 |
| [末层读出、词表投影与提示词评分](target-readout-and-prompt-scores.md) | 冻结 greedy 的联合上限、可选有限 K 的本地与跨卡融合、通信盈亏平衡行数 |
| [验证后分数输出、惩罚与五步草稿投影](score-readers.md) | 有限 Top-K 的拼行、归一化与排名，W1/W2 缓存的请求收益范围 |
| [FFN 输出合并、HcPost 与下一层 premix](ffn-merge-premix.md) | 固定专家分布下的中间写读与 GMM2 的 20 个列块读者 |
| [OA、OB 的共享 Cube 与接收端旋转](oa-ob-shared-cube.md) | 逐卡矩阵与搬运预算、分块门槛和不同形状的处置 |
| [逐项融合归因](scope-reconciliation.md) | 分支适用条件、重复预算与退出融合筛查的原因 |
| [固定路由的两处中间结果](baselines/routed-expert-handoffs.json) | 路由专家的 FP16、INT8 中间写读和 C8 请求收益预算 |
| [索引交付、暂存与两次矩阵读取](baselines/sparse-selection-delivery.json) | QLI 索引发布、38 层 SMLA 的稀疏读、临时缓冲和临界速率 |
| [Attention 输入、旋转与缓存写入](baselines/attention-input-fusions.json) | Q/KV 实际汇合、C2 与索引发布、六个普通请求与冻结 C8 的联合收益 |
| [OA 与 OB 的逐请求估算](baselines/oa-ob-fusion.json) | 固定权重分片和归约的搬运收益、流水敏感性和普通请求 |
| [mHC 两处融合与同排程对照](baselines/hc-paired-boundaries.json) | 171 SDK 的实际数据通路、两处残差共用的请求收益与扩宽写入成本 |
| [采样形状、返回分数和请求估算](baselines/random-sampling-fusion.json) | Ascend 双阈值掩码、普通与投机采样、冻结和普通请求的条件收益 |
| [上一步有效 token 数与下一轮输入等待](baselines/draft-submit-waits.json) | context KV 的主机收益、选词直接产出种子与计数、未绑定事件的保守计价 |
| [三层草稿 context KV 与跨模块联合收益](baselines/draft-context-fusion.json) | 共享位置和地址、旋转与缓存直写，逐卡主机提交和设备执行的重叠 |
| [目标与草稿的词表选词](baselines/vocab-selection-fusion.json) | 真实形状、普通请求与八请求整批的融合估算 |
| [普通负载在 171 的实测](baselines/workloads.md#171-的普通请求已实测) | 逐卡 guard 命中、首文本前的图回放、跨 DP 缓存和对照输出 |
| [常见推理场景与候选适用范围](baselines/workloads.md) | 真实对话、短代码回复、多轮前缀复用，以及本机还需要哪些证据 |
| [DSpark 三层非专家工作的整组收益](baselines/draft-fixed-expert.json) | 保留专家耗时，比较具名计算与全部剩余时间，识别决定判断的请求 |
| [mHC 循环提前退出的价值筛查](baselines/mhc-iteration-cost.json) | 真实调用次数、两侧的合计收益、提前退出的最晚时点与开销 |
| [QLI 在批量 slot 基线上的增量](baselines/qli-batched-slot-c8.json) | 171 同基线前后参照，输出一致性、首 token、整批耗时和恢复 |
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

## 历史对照

以下记录涉及专家放置与负载均衡，保存用于复核旧结论。当前融合候选见[融合候选与预计收益](fusion-candidates.md)。

| 历史记录 | 内容 |
|---|---|
| [EP16 加到批量 slot 的实际增量](baselines/ep16-batched-slot-c8.json) | 各指标的独立裁决、同一代服务上的九轮计时和恢复 |
| [EP16 专家布局的冻结 C8 对照](baselines/ep16-interval-c8.json) | 首 token、整批、自然命中、数值检查和原服务恢复 |
| [按专家布局重算矩阵计算下限](baselines/ep16-compute-floors.json) | 原布局、每层调入两个专家和自由分配在相同矩阵工作下的下限 |
| [EP16 专家布局的实际命中与所需收益](baselines/ep16-layer-guard.json) | 两轮参照的传递门槛、一次零命中的批次差额、诊断轮的覆盖 |
