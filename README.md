# Inference Foundry

本分支记录 **172 上的 DeepSeek V4.1 Flash W8A8 部署与 DeepInfer 910C 适配**。

**真实DSpark固定5：521.59 token/s、平均首token795.65ms，16/16完整输出一致。** 对照冻结vLLM518.42/831.25ms，吞吐仅领先0.61%，只代表指定两轮热文本负载。172当前被他人训练占用，按用户条件暂停；置信度变长路径仍待最终验收，当前无本项目NPU服务。 此前提示查找实验的完成结论已撤回，见[官方DSpark进度](projects/910c-deepinfer-172/optimizations/20261027-dspark/README.md)。

| 入口 | 内容 |
|---|---|
| [当前 DeepInfer 项目](projects/910c-deepinfer-172/README.md) | 原生W8A8源码、配置、两轮结果、适用范围和回切入口 |
| [冻结 vLLM 基线](projects/910c-deepseek-v4.1-flash/README.md) | 原始调优与基线证据，按冻结时现场保留 |
| [仓库导航](docs/REPOSITORY_MAP.md) | 当前资料与历史研究的位置 |
| [完整历史快照](archive/legacy-20261019/README.md) | 整理前分支的全部文件，内部目录结构保持原样 |

当前native状态以 `projects/910c-deepinfer-172/STATUS.json` 为准；原vLLM容器已停止并保留。历史记录不代表当前服务状态。

本次整理发生在 `910c-deepseek-v4.1-flash`，来源为
`codex/ascend-sglang-vllm-comparison@59dde6baad3e55feb9338db02b71d28e740d38e5`。
原分支和 main 未修改。历史文件整体放入归档，Git 历史继续保留。

仓库保存脚本、配置、摘要和小型证据；模型、编译缓存、完整运行日志保存在服务器的 `<WIO_ROOT>` 下。

历史提示查找实验：[差距排查与固定场景验收](projects/910c-deepinfer-172/optimizations/20261027-gap-closure/README.md)。[此前v10–v13记录](projects/910c-deepinfer-172/optimizations/20261027-ttft/README.md)完整保留。
