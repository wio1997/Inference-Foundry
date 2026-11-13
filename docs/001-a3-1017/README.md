# 001：A3 / 910C 接入 1017 W8A8 与 DRAM Engram

## 1. 原理介绍

固定 daily 镜像已有 V4 系列模型及部分 NPU 路径，但该 1017 W8A8 checkpoint、A3 index/cache ABI 和 Engram 存储组合不能直接运行。适配保留 SGLang 的请求、pair/history 和图状态管理，将 sparse MLA/indexer 接入已有 A3 BF16/INT8 算子，按当前 CANN 重建 host 注册。

Engram 表原数据约 202.76GB，采用节点共享 CPU 内存，实测常驻 DRAM 约188.83GiB；避免把这部分放入 NPU HBM。一次调用保留约4.20GiB workspace 的问题改为按 stream 等维度复用容量池。上述是存储/生命周期机制；未单独测定它们节省了多少端到端时间。

## 2. 源码变更

[补丁说明](../PATCHES.md) 按模型、缓存、算子、Engram 和图接口列出代码。完整 SGLang 与 SDK diff 在 [patches](../../patches)，完整修改后文件在 [G30 运行时](../../compat/runtime/g30)。变更包含数据格式、门控和浮点展开路径，不能一概声称数学等价；详见 [未验证范围](../LIMITATIONS.md)。

## 3. 实测范围

单机8物理910C/16逻辑die、aarch64、固定镜像与 W8A8 checkpoint；四档输入32768/65536/131072/262144，输出1024，C16，每档warm16/formal32。最终四档总输出吞吐139.6191/78.2387/49.4048/32.1642 token/s。G29的64K/128K吞吐更高，完整不利指标和 P99 同样保留。见 [最终CSV](../../results/20261112T134338Z/validated_profile_metrics.csv) 与 [基线](../../results/20261112T124145Z/verified_results_all.json)。

## 4. 波动噪声

没有同臂重复测量，没有同臂漂移或跨批次噪声标定。逐请求离散可由原始 SQLite 时序库和恢复后的详情分析，但本次没有把单次 G29/G30 差异裁决为融合收益。不能将 TTFT/TPOT 的平均值解释成所有请求的固定耗时。

## 5. 结论与边界

部署和指定负载测量已核验；完整模型质量和数值等价性未验收。64条相同长随机输入仅1条完整预测相同，原因未验证。代码不能直接用于910B、x86_64、其它 CANN/Torch 组合或作为官方原生支持声明。当前状态适合复核与后续质量验证，不是精度已通过的产品发布。

## 6. 代码与链接

[环境](../ENVIRONMENT.md) · [复现](../REPRODUCE.md) · [测试数据](../DATA.md) · [补丁](../../patches/sglang-a3-1017.patch) · [验收记录](../../evidence/original_acceptance/final_acceptance.json)

本条目与代码、记录同处 `sglang-dsv4.1flash1017-910c` 分支，均使用仓内相对链接。后续若继续优化，优先补齐当前配置的模型质量、数值差异来源及重复测量；本次归档没有启动这些新实验。
