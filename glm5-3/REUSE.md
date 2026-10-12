# 历史资产与边界

先读 [重置审计](docs/RESEARCH_RESET.md)。旧研究没有被整体否定；本次改变候选的表达与实验入口。

| 资产 | 用途 |
|---|---|
| [manifest](patches/20261007/manifest.json) / evidence | H6/H5 限定收益、MTP/其他候选反例 |
| [patch 包](patches/20261007/README.md) / fixtures | 源码定位、可回滚对照与实现身份；不自动部署 |
| `docs/research/` | 模型消费者、状态依赖、CANN/vLLM 路径线索；先核验版本 |
| `adapters/aisbench/`、`runtime/aisbench_slo.py` | 后续服务验收；不作为独立计划的底层执行器 |
| `runtime/phase_runner.py`、`scripts/zcode_bridge.py` | 按需复用的进程/交接工具；不是研究前置任务 |
| `records/`、根 `performance_knowledge/` | 按机制查询，不通读或复制全历史 |
| [旧入口](legacy/README.md) | 解释历史规则，不恢复 Current/next_action |

未发布到 main 的研究分支以不可变 commit 链接引用，不在 main 中引用不存在的本地路径。GLM-5.2/5.3、两机/六节点、不同精度和负载分别保留，不自动迁移。

复用前只回答：原件证明什么；当前合同有何不同；还缺哪条影响决定的证据。旧事实足够则引用，不为增加 Run 重采。
