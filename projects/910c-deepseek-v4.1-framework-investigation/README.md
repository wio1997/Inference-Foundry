# DeepSeek V4.1 Flash 推理框架算子优化排查

本目录是持续更新的排查档案。固定 DeepSeek V4.1 Flash W8A8、DSpark、Engram、CANN 9.2、910C 及既定 C8 负载，以实际推理顺序组织算子优化分析。整体任务仍在进行。

远端唯一发布位置：`wio1997/Inference-Foundry` 的 `910c-deepseek-v4.1-flash` 分支，`projects/910c-deepseek-v4.1-framework-investigation/`。该目录记录本次 171 分析，不覆盖同分支已有的 172 DeepInfer 结果或冻结 vLLM 基线。

| 入口 | 内容 |
|---|---|
| [完整排查报告](REPORT.md) | 固定范围、执行顺序、价值筛查、实验裁决、理论边界和重要缺口 |
| [模块分析](modules/README.md) | M00–M11 的分支、改写空间、证据、处置和下一步 |
| [进度状态](progress.json) | 当前模块、最新结果、未完成范围和恢复入口 |
| [固定范围与版本](scope.json) | 模型配置、硬件/负载、选中源码的精确哈希及绑定限制 |
| [分析日志](analysis-log.jsonl) | 可审查的问题、观察、判断依据、改进和后续动作；追加保存 |
| [证据索引](evidence-index.json) | 原始位置、内容哈希、来源 Git 版本和证据适用范围 |
| [维护约定](MAINTENANCE.md) | 更新顺序、状态口径、发布方法和完成标准 |

原始 profile、模型、编译缓存和完整日志留在现有服务器/分析库，不重复上传。这里保存可阅读报告、小型结构化摘要和精确证据引用，不保存凭证。
