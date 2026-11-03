# 厂商资料检索：9382 计算、内存与互联上限

检索时间：2026-10-28 UTC。使用公开 Web 检索和本地已保存记录，没有打开浏览器或访问设备。三份厂商页面的文本检索快照及 SHA-256 见 `vendor_sources_9382.json`；产品页面未标明固定文档版本，复核时应同时核对页面更新日期。

## 找到了什么

| 厂商原始资料 | 厂商列出的对象和参数 | 对 169 的适用性 |
|---|---|---|
| [Atlas 800I A3 产品规格](https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a3) | 10U 服务器、单机 8×昇腾 910，INT8 算力最大支持 **8.96 POPS**；内存字段写 8×128GB、带宽 **3.2 TB/s**；D2D 字段写 **双向 784 GB/s** | 整机产品级公开数据。169 的服务器型号、8 卡具体 SKU 是否与此配置完全一致，以及各数字的测量边界尚未证实；不能把它当单逻辑设备或某个指定 HCCL cut 的硬上限。 |
| [Atlas 800T A3 产品规格](https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3) | 同为 8×昇腾 910 的服务器页面，INT8 算力最大支持 **12.0 POPS**；内存和 D2D 字段分别为 **3.2 TB/s** 与 **双向 784 GB/s** | 产品类型与 800I 不同，INT8 数字亦不同，不能只凭 A3/910 名称选其中一个作为 169 峰值。页面没有给出 9382 到具体整机配置的映射证明。 |
| [CANN 9.1.X GetSocVersion 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/Utils-API/platform_info/PlatformAscendC/GetSocVersion.md) | 对 A3 产品，指定读取 `npu-smi` 的 `Chip Name` 和 `NPU Name` 拼出实际配置名；建议以 `aclrtGetSocName` 区分 A2/A3 | 结合现有 `analysis/operator-ceiling-20261028/model-source-inventory.json` 的 board0 输出 `Chip Name: Ascend910`、`NPU Name: 9382`，**在该设备属于 A3 的前提下**，按厂商命名规则可推得查询对象的配置名 `Ascend910_9382`。这是型号匹配，不证明其隶属的服务器具体型号或完整硬算力/带宽。 |
| [MindCluster 7.0.RC1 带宽/算力测试说明](https://www.hiascend.com/document/detail/zh/mindcluster/70rc1/toolbox/toolboxug/toolboxug_0006.html) | A3 产品中一张 NPU 卡包含两个 device；P2P 分卡内与卡间；算力测试以给定矩阵计算量除以耗时 | 可以帮助正确区分逻辑设备与物理卡和设计后续测量；测试结果仍是可达带宽/算力，不能当物理硬上限。 |
| [HDK 26.1 A3 HCCS 带宽字段说明](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3npu/npusmia3_0080.html) | 介绍按 HCCS 链路分别报告发送/接收瞬时带宽 | 明确哪些计数是方向和观察值；示例数值不适用于 169，也不是该链路硬最大值。 |

产品页的 3.2 TB/s 紧跟“8×128GB”的内存字段，但网页没有把统计口径清楚映射为本机某个逻辑设备或整机总和；因此不自行乘除。D2D 的 784 GB/s 明确是**双向**口径，网页未确认用户选定的通信 cut、单方向速率或并发共享规则；不能直接代入 169 的有向边界。两页均未明确 INT8 稠密/稀疏、时钟或数值累加的完整口径，`最大支持` 也需要结合规格定义才能用于形式化物理界。

## 与当前条件估算的关系

本地 `int8_ceiling_certificate.md` 根据厂商分析工具公式和 profiler 元数据给出每逻辑设备 `707.7888 TOPS` 条件估算，16 项算术和是 `11.3246208 POPS`。它比 800I 产品页整机标称 `8.96 POPS` 大，但比 800T 页的 `12.0 POPS` 小。**不能由此断言工具或厂商资料错误**：尚缺精确整机型号、产品 SKU、统计口径、核数/频率假设和资源映射。这个差异说明必须先对齐对象和单位，不能选择看起来更符合预期的数字。

目前所需的严格参数状态：`P_INT8,169,max = UNKNOWN`；`B_HBM,169,max = UNKNOWN`；特定 `B_D2D,cut,max = UNKNOWN`。因此全模型无条件数值 TPS 上界仍未知。

## 后续资料和可访问性

1. 从 169 的已存资产信息确定厂商、完整整机型号、板卡 SKU/数量、物理与逻辑 device 映射；当前原始 board 查询只覆盖 chip0。对照对应**确切型号**的厂商规格文档，明确 dtype、稠密/稀疏、算子/累加类型、频率和资源统计边界。
2. 查该型号内存和互联规格的单向/双向定义、链路及 cut、是否可共享或聚合，以及额定值是否覆盖所有运行状态。再结合必要数据量，才能写出数值时间界。
3. 产品页的白皮书链接文本称版本 04，点击后落地页标题为 [技术白皮书 07](https://e.huawei.com/cn/documents/products/computing/5f99170196404c5f90a7d0638c613652)，页面提示无法预览并显示注册/联系信息表单；本文没有填写表单、使用浏览器或假装读过白皮书正文。若后续已有授权可直接访问的厂商技术文件，优先核对文件中精确型号和口径。

执行 `python3 agents/bounds/vendor_sources_9382.py` 可校验三份已保存检索快照的关键词和 SHA，生成同目录 JSON；它不验证厂商页面对 169 的适用性，也不访问网络或设备。

## 2026-10-29：精确名称与公开资料再核对

只读现有 `actual-sku-and-cp.json`、已存 CANN 头文件和 SKU 审计，并直接查询公开文档，没有访问 169 或运行设备。新查到的 [Triton-Ascend v3.6.0 性能分析资料](https://triton-ascend.readthedocs.io/en/v3.6.0/debug_guide/profiling.html) 把 `Ascend910_9382` 列在 CANN 9.2.0 处理器名称清单中；它支持该名称是工具识别目标的判断，不给出与 169 板卡和所有运行状态绑定的硬算力、最高频或 cut 带宽。[PTO Virtual ISA 的 A2/A3 MegaMoE 示例](https://pto-isa.gitcode.com/kernels/manual/a2a3/dispatch_mega_combine/) 显式将 `Ascend910_9382` 列为支持目标，并描述按专家重叠计算；该文档是示例的支持范围，不证明 169 历史 MC2 使用了该实现、达到了任何示例速度或满足某个硬上限。

已存 `Ascend910_9382.ini` 的 `cube_freq=1800`、`ddr_rate=32` 等仍缺字段单位、覆盖物理路径和所有工作状态下上限的证明；本地快照的头文件也没有提供可直接验证其为完整设备 HBM 硬带宽的定义。因此本次没有新的可代入 169 数值硬资源上限，`P_INT8,169,max`、`P_BF16,169,max` 和指定方向的 `B_cut,max` 继续为 UNKNOWN。上游获取精确物料号及对应原厂规范的资料路径仍是最短补证，不能借相似机型标称数据、工具估算或示例测量填补。已有 BF16、MC2 窄域计算对照及适用性分别见 `agents/diagnosis/ENGRAM-MATMUL-DISPATCH.md` 和 `agents/bounds/hardware_sources_audit.md`；当前没有新的同域必要工作或真实动态状态证明，可将其提升为全局数值界。
