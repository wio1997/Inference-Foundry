# Inference Foundry

本分支记录 **172 上的 DeepSeek V4.1 Flash W8A8 部署与 DeepInfer 910C 适配**。

当前 DeepInfer 优化结果 **737.99 token/s**，平均首token **4.32s**，原冻结 vLLM-Ascend 基线 **518.42 token/s**；16/16全文一致。首token延迟仍更高，性能范围见独立项目。

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
