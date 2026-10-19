# 2026-10-19 阶段归纳

当前可接受的独立服务优化是连续 RoPE 表：固定负载下 C4 吞吐 **143.517→164.631 output tok/s（+14.71%）**，三轮均改善，p95 TTFT、完成延迟和每请求平均 TPOT 均改善；C1 TPOT 下降17.26%。代价为额外256 MiB/rank，16 rank合计4 GiB。源码保留显式开关，结论限本次有限配对负载。[分析与验收](rope-cache-layout-20261019.md)。

本轮 Indexer K 私有图与原 Prolog 重叠的结论为 **`INCONCLUSIVE_SDMA_DEFAULT_OFF_INDEXER_K_GRAPH_OVERLAP`**。功能和有限质量检查通过，但全 SDMA 前置证据未闭合，未运行正式 C1/C4，也没有服务收益结论。按用户要求，本轮后暂停，不展开新机制或追加同条件轮次。

## 部署与约束

32/33 已建立原样 MooncakeConnectorV2、官方 proxy 的 TP16 P + TP16 D 原生 PD 服务；原纯 MemCache 问题、文档及候选修复继续保留，不等同于纯 MemCache 后端已完成推广验收。[部署记录](../official-pd-native-20261017/README.md)。

所有实验保持 GLM-5.3 W8A8C8、原模型逐层 Indexer 精度、主 int8 KV 和既有算子；没有参数搜索或新 kernel。目标 D 为 FULL graph、P 为 PIECEWISE[256,1024]；本轮只改真实 MTP eager 的 K 编排。数据路径要求全 SDMA，控制与初始化通路单独记录。原生 PD 的既有 SDMA 证据不能替代新复制的物理路径证明。

## 主要结果

| 机制 | 结果与证据边界 | 处理 |
| --- | --- | --- |
| 连续 RoPE 只读源表 | C4 +14.71%，三轮与整体 p95 均改善；有限26组完整64-token输出相同 | 接受该有限负载点 |
| MoE prefix，GMM2 前恢复原生 count | C4 +4.64%，但 p95 TPOT +5.49% | 未过联合门槛，默认关闭 |
| 提前 sample D2H 影子 | all16共10,960份副本逐位相同；7个设备步骤早副本提前6.201–7.453 ms就绪 | 仅就绪时间证据 |
| 提前发布／独立退休 | 正式 C4 −1.54%，p95 TPOT +1.70%；C1 TPOT +0.36% | 拒绝，默认关闭 |
| SFA／Indexer 同步元数据复用 | 18,752次all16功能逐位通过；4个初轮完整输出变体未被有限原生对照覆盖 | 质量未闭合，未进入正式性能，默认关闭 |
| Indexer K 直接 eager 切流 | 局部完整提交成本增加约21%–26% | 拒绝该编排 |
| Indexer K 私有图＋三份输入复制 | 局部片段有改善；整模型功能／有限质量通过，45份复制缺少物理任务关联 | SDMA证据未闭合，未进入正式性能，默认关闭 |

各轮比例独立，不相乘；局部图回放和 CPU 提交节省不换算成整模型服务收益。原始慢块、失败和差异均保留。

## 本轮 K 子图核验

首个整模型构建实际未命中候选：16 rank各2660次调用都是原生回退。精确镜像源码确认模型持有 `IndexerWrapper`，计算 backend 在 `.impl`。修正委托绑定后，使用新的不可变构建，只替换自有 D，保留已加载 P 与原代理；无效构建日志、源码和快照保留。

修正版 `bd6889091318…` 的两条短 PD、功能16、四并发8、缓存16均数值通过。all16共 **22,880** 次实际候选 K 调用与当次原生 K 逐位一致，独立输入所有权和单次消费通过，每个 rank覆盖 N6/12/18/24。原 Prolog、缓存写、target FULL、输出与 KV 退休保持原生。[机制与过程](indexer-prolog-overlap-20261019.md)。

observer 关闭后，quality52数值通过；26组完整64-token输出中25组相同，quality-0的初始和重复候选输出均出现在同一 P/D 的有限原生对照集合中。这仅通过本轮有限质量门槛，不证明通用问答质量或差异原因。

真实 C1/C4 诊断的32份rank轨迹全部导出：**7,200份带标签新增复制中7,155份精确关联为SDMA_SQE**；C4 rank0/2/6各15份新增复制缺少可关联的物理任务。缺失关联不能推定为SDMA，也不能推定为RDMA/TCP回退。冻结合同要求全部新增复制证据闭合，因此正式性能及其取消／peer复用阶段均未执行。

最初 CPU 导出超时及恢复脚本的文件／格式判断错误保留；最后复用25份完整轨迹，只补7份缺失导出，没有追加NPU请求。标签关联仍要求 enqueue→dequeue correlation→HostToDevice flow→完全相同device PID/TID/start，未放宽门槛。[结构化硬件结论](../evidence/20261018-p-first-seed-device-v2/recovery-v2/indexer-k-graph-hardware-reviewed-v2.json)与[关闭结论](../evidence/20261018-p-first-seed-device-v2/recovery-v2/indexer-k-graph-closed-without-formal-v1.json)。

## 交付与恢复

代码、有限合同、判定脚本、结构化结果和SHA备份收据同步到现有草稿 [PR #2](https://github.com/wio1997/Inference-Foundry/pull/2)，不合并。完整原始profile、trace、copy-chain与模型源码／日志快照保存在本机证据备份目录；Git保留可核对收据。

现场最终状态及精确CID／UID清理结果统一见 [HANDOFF](../HANDOFF.md) 和 [暂停收据](../evidence/20261018-p-first-seed-device-v2/recovery-v2/indexer-k-graph-pause-receipt-v1.json)。恢复工作需用户重新指示，不能把准备好的正式脚本当作已执行结果。
