# Ascend 上 SGLang / vLLM-Ascend 对比与优化

分支：`codex/ascend-sglang-vllm-comparison`。建立于 2026-10-16，基于原研究分支 `codex/glm-execution-plan-reset` 的 `6b44d8b2755f9527b3066fdd7a014ed5cdc062bb`。

目标：在用户实际 A3 / SDMA 硬件上，以 vLLM-Ascend 为正确性参考和性能对照，补齐 SGLang GLM-5.3-Flash 适配，并优化原 GLM-5.3 的完整服务性能。两个模型的结果分开，有限实验只能选出已测条件内更优的实现。

| 实验线 | 最新机器 / 角色 | 第一交付 |
|---|---|---|
| Flash 1P1D，vLLM-Ascend | 14=P，15=D | 同权重的真实PD正确性参考 |
| Flash 1P1D，SGLang | 46=P，47=D | 先单机文本正确，再真实1P1D |
| 原GLM53四机2P2D，vLLM-Ascend | P=30/31，D=43/45 | 新SDMA、独立前缀性能基线 |
| 原GLM53四机2P2D，SGLang | P=34/35，D=36/41 | 同合同基线与受控优化 |

备用32/33/37；排除38/42。旧SGLang相关33容器已按授权停止，镜像/权重/配置和容器对象保留；父独立复核16台256die健康、无NPU进程。42曾临时打通管理通道，用户排除后已撤除；不再用于本批。新2P2D与Flash环境尚未部署。起初错峰压测，并按实际拓扑复核物理组差异。

**最新要求：不考虑共享前缀。** 新测试每档使用 C 条相互独立的 74,400-token 前缀，分别预热，连续复用四轮，每轮全新后缀，得到 4C 条不同的完整 80,000 输入 / 600 输出请求。C=16/32/64 时分别是 64/128/256 条。预热不计吞吐，正式保留自然本地缓存。旧共享前缀结果只归档，不能作为新负载的框架性能差距。

最新交接入口：

- [服务器分工与现场占用](SERVER_ALLOCATION.md)：每台IP、P/D用途、旧作业、资源申请与角色冲突。
- [权重/镜像就绪与材料总表](ASSET_READINESS.md)：哪些已在Git、哪些二进制在外部、哪些新目标尚未具备。
- [只读现场证据JSON](inventory/20261016/readiness.json)：各节点采样时间、权重清单/小文件SHA/索引、镜像ID和检查边界。

入口：

- [实施方案](PLAN.md)：先做什么、适配范围、如何选择优化。
- [测试合同](BENCHMARK_CONTRACT.md)：数据、预热、缓存、全程/稳态和 SLA 口径。
- [Flash 多模态测试方案](MULTIMODAL_TEST_PLAN.md)：单图/多图/多轮/视频、视觉展开长度、缓存身份、PD/SDMA和端到端计时；当前为计划，未执行。
- [当前状态](HANDOFF.md)：计划与已完成事项的区别。
- [Flash 适配任务](tracks/flash-adaptation/README.md) / [接手提示](tracks/flash-adaptation/START_PROMPT.md)。
- [原模型2P2D对比任务](tracks/glm53-2p2d/README.md) / [接手提示](tracks/glm53-2p2d/START_PROMPT.md)。
- [现有环境与优化归档](archive/20261016/README.md)：实际源码、启动配置、权重校验、已验证与未验证优化。

建议两个长期对话分别负责两条实验线；共用本目录合同和资源表，不要求开两个自主执行者同时抢设备。未创建新对话或恢复定时任务；已按授权停旧SGLang，尚未启动新推理部署/模型测试。

权重和镜像二进制不提交 Git。权重原件在现场保留；本目录提交其来源、路径、分片大小/历史 SHA 和已有验证证据。镜像固定 digest，SDK 保存源码版本、实际映射库 SHA 和离线资产位置。
