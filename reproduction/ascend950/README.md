# Ascend 950DT 部署与性能复现

对象：DeepSeek-V4.1-Flash，单机 8×Ascend 950DT，DeepInfer 与 vLLM Ascend。材料整理于 2026-10-23。

| 入口 | 用途 |
| --- | --- |
| [DeepInfer 部署手册](../../docs/ascend950/README.md) | 下载、编译、兼容补丁、拓扑、启动与功能验收 |
| [DeepInfer 部署脚本](../../deploy/ascend950/) | 固定源码、依赖及已保存的验收证据 |
| [vLLM Ascend 部署](../../deploy/ascend950/vllm-ascend/README.md) | 官方 ds41 镜像、补充算子构建、DSA 补丁、服务验收 |
| [性能测试复现](../../docs/ascend950/PERFORMANCE_REPRODUCTION.md) | 32K/128 与 80K/600，普通组与实际 KV 预热组 |
| [全链路分析状态](../../docs/ascend950/FULL_CHAIN_STATUS.md) | 已采集项、代码入口、缺失证据与尚未完成的对照 |
| [实验代码](performance-20261021/) | 原服务适配器、AISBench 映射、采集和分析工具 |

## 已实测的配置与改动

| 项目 | 内容 | 验证范围 |
| --- | --- | --- |
| DeepInfer 原生执行 | DSpark 5、图模式、ZOS、MegaMoE、NPU Engram 存储 | 在原服务器运行；这些是框架原生路径，不计作本项目新增移植 |
| DeepInfer 服务适配 | DP8 独立请求、原生序列保留与 fork、有限前缀父序列 | 真实 HTTP 测试，保留每请求实际输出 ID、缓存工作量和排空检查 |
| vLLM 原生执行 | DSpark 5、异步调度、Decode 图、静态 Kernel、共享专家重叠、Engram 卸载 | 使用固定官方镜像及记录的源码版本；不声称达到官方峰值 |
| vLLM 适配 | 官方 moe_gating_top_k_hash provider；原生 mixed-batch reorder 初始化 | 构建、ABI 和实际服务记录；补丁有基线 SHA 检查 |
| AISBench 适配 | 固定 token ID、DP 路由、缓存命名空间、80K SSE 缓冲 | 原 AISBench 负责并发和计时；服务响应仍存在首包额外字段差异 |
| 全链路探针 | 八卡完整 cohort、CPU 线程、内存快照、原生工作量、算子与图分析 | 已采集和部分分析；因果归因未完成 |

没有开展 NVIDIA 优化向 950 的移植。代码中的启动预热工具包含早期 512-token 配置，当前 8192 预算测试默认关闭；不能把旧预热覆盖率用于当前配置。

发布包含代码、固定配置、输入生成所需的词表 ID 集合和校验值。模型、镜像及多 GB profile 原始包不在 Git 中。新主机需要按部署文档准备环境，不能把本次整理当作已经执行过新主机重部署。
