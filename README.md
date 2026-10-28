# DeepInfer · DeepSeek V4.1 · Ascend 910C

分支 `deepinfer-dsv41-910c` 用于 DeepSeek V4.1 Flash W8A8 在 Ascend 910C 上的 DeepInfer 适配与部署，保存已验证源码、部署步骤、启动配置及性能记录。

| 文档 | 内容 |
|---|---|
| [部署文档](projects/deepinfer-dsv41-910c/README.md) | 环境要求、目录准备、容器创建、服务启动、检查与复测 |
| [适配说明](projects/deepinfer-dsv41-910c/ADAPTATION.md) | W8A8、注意力与缓存、并行通信、DSpark、图执行、Engram 的实现与边界 |
| [测试结果](projects/deepinfer-dsv41-910c/RESULTS.md) | 固定负载、两轮原始指标、vLLM Ascend 对照及结论范围 |
| [源码与版本](projects/deepinfer-dsv41-910c/SOURCE.md) | 源码归档、文件校验、依赖版本和许可 |

部署配置为 **8 张物理卡、16 个逻辑设备，DP2 × TP8、EP16**。仅 Engram 表卸载至 CPU，主干和专家权重常驻 NPU；草稿生成采用学习式 DSpark。

2026 年 10 月 28 日归档版本的指定两轮测试为 **551.11 输出 token/s、784.90 ms 平均首 token 延迟**，16 个完整输出均与冻结参考一致。该结果限定于文档中的 32K 输入、1K 输出、并发 8 和真实约 90% KV 命中负载。全部上游优化的迁移与同期跨引擎对照仍在进行。

上述部署记录对应冻结源码版本 `allopt-mc2-cap4096-v1-20261028`。后续设备 hash 与共享专家多流的实验结果及源码补丁见[补充记录](projects/deepinfer-dsv41-910c/optimizations/20261028-device-hash-overlap/RESULTS.md)；旧版部署脚本继续对应冻结版本。

`projects/910c-deepinfer-172` 保存此前的适配与实验记录，`projects/910c-deepseek-v4.1-flash` 保存冻结的 vLLM Ascend 基线；其中的运行状态按各自记录日期解释。本分支的部署入口为上表所列文档。
