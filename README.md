# Inference Foundry

本分支用于记录 **172 上的 DeepSeek V4.1 Flash W8A8 单机部署**。

| 入口 | 内容 |
|---|---|
| [当前项目](projects/deepseek-v41-172/README.md) | 模型、镜像、启动命令、现场状态与验证证据 |
| [仓库导航](docs/REPOSITORY_MAP.md) | 当前资料与历史研究的位置 |
| [完整历史快照](archive/legacy-20261019/README.md) | 整理前分支的全部文件，内部目录结构保持原样 |

当前状态以项目内 `STATUS.json` 为准。历史记录不代表当前服务器或服务状态。

本次整理发生在 `codex/172-deepseek-v41-inference-records`，来源为
`codex/ascend-sglang-vllm-comparison@59dde6baad3e55feb9338db02b71d28e740d38e5`。
原分支和 main 未修改。历史文件整体放入归档，Git 历史继续保留。

仓库保存脚本、配置、摘要和小型证据；模型、编译缓存、完整运行日志保存在服务器的 `<WIO_ROOT>` 下。
