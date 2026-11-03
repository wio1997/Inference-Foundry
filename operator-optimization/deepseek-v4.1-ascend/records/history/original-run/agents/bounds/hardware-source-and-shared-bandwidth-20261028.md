# 169 精确硬件规格与共享带宽条件界：公开资料复核

检索日期：2026-10-28，使用公开 Web 查询和既存本机快照；未使用浏览器、未连接 169、未操作设备。

## 资料及其适用范围

| 资料 | 一手出处和能确认的范围 | 能否用作 169 的严格极限 |
|---|---|---|
| OEM 名称 | [软通华方产品目录](https://tongfangpc.com/zh-cn/htmls/chanpin/index.html) 的搜索索引列出“软通华方超强A800T A3”。本机既存 `stage-dmi-audit.md` 记录容器可见固件产品名 `ChaoqiangA800T A3`，但未证明该 DMI 与公开目录的精确 SKU/BOM 映射，或容器看到的是裸金属信息。| 不能。索引不含该机精确型号的 INT8 吞吐、HBM 和链路物理上限。官网正文和 [OEM 产品系列 PDF](https://www.tongfangpc.com/themes/default/images/chaoqiang-A-group/chaoqiang-A-Group.pdf) 本次直接抓取均超时；不要把搜索摘要伪装成已读完整规格。|
| 其他厂商相似产品 | 华为 [Atlas 800T A3](https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3) 和 [Atlas 800I A3](https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a3) 的整机产品页分别列 INT8 最大 12.0 POPS 和 8.96 POPS；也列出 HBM 与 D2D 带宽。先前 `vendor_sources_9382.md` 已记录 3.2 TB/s 和双向 784 GB/s 的页面字段及单位歧义。 | 不能直接代入。厂商、产品系列、BOM、dtype 及稠密/稀疏和设备/卡粒度未匹配；3.2 TB/s 未明确对应本机单 device/物理卡/整机，双向互联数不能作为指定单向 cut。 |
| 精确名称和容量示例 | 昇腾生态 [HY-Pano-2.0 部署记录](https://ai.gitcode.com/Ascend-SACT/HY-Pano-2.0) 将另一台型号 IT22HMDA_2_S 的 4×`Ascend910_9382` 写作“每卡 64GB HBM”。本机已有 9382 命名证据见 `vendor_sources_9382.md`。| 此记录仅描述其部署环境，**不是 169 BOM/带宽/算力证明**；且“卡”和 CANN 逻辑 device 的映射需另核验。容量也不是带宽。 |
| 共享资源的方法论 | [Triton-Ascend 性能分析文档](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/debug_guide/profiling.md) 明确提示 GM 搬入 MTE2 与搬出 MTE3 可能共享带宽，不能各自假定同时享有整个带宽；其 1.8 TB/s、11.06 TOPS 均是“某款处理器”的教学示例。 | 作为需要检查资源共享的线索可用；不能证明所有 9382 互联/读写路径共享同一峰值，也不能套用示例数字。 |

本次未找到 OEM 一手来源将 DMI 名称、硬件版本、逻辑 device/卡映射与全部 INT8、HBM、互联极限严格对应。`P_INT8,169,max`、`B_HBM,169,max`、指定有向 `B_link,cut,max` 均仍为 **UNKNOWN**。之前 profiler 工具公式的条件估算及厂商其他机器的标称值不是设备在一切合法频率和操作条件下不可超越的证明。

## 不引入错误重叠的条件公式

固定某个执行域 `A` 和测量时窗。设 `R_min` 与 `W_min` 是**对 A 中所有允许的等价实现均已证明**必须通过同一个物理边界的总读取和总写入字节数；固定权重可预驻留、缓存/复用、融合和表达式重写等必须允许，不能把某条 trace 搬运量直接当 `R_min/W_min`。若另证明该同一共享资源在窗口内的任意前缀满足 `Q_rw(t) ≤ β_shared + B_shared,max·t`，其中 `Q_rw` 是读写合并的同口径累计字节、`β_shared≥0` 覆盖短窗突发及此前在途但窗口内计入的工作，`B_shared,max>0` 覆盖该资源所有并行路径及合法工作状态，则守恒给出

`T ≥ max(0, R_min + W_min − β_shared) / B_shared,max`。

这里 R 和 W 必须确为两个不可由一次被计工作同时满足的必要传输量；若同一次传输同时满足多个逻辑消费者或计数口径重复，不能重复计数。初态中已完成、无需再次跨边界的工作应先排除，不能既计入必要工作又将其当作突发抵消。只有另证共享资源的初始信用为零，才能简化为 `(R_min+W_min)/B_shared,max`；原文未明确短窗包络，现予补正。

这个式子允许搬入、计算、搬出重叠；它把读写通过同一资源的必要工作量加在分子，而不是将观测的读时间和写时间机械相加。若无法证明共同受限资源或允许独立全双工，且分别已证明 `Q_read(t)≤β_read+B_read,max·t` 与 `Q_write(t)≤β_write+B_write,max·t`，则只能写 `T ≥ max((R_min−β_read)_+/B_read,max, (W_min−β_write)_+/B_write,max)`，其中 `x_+=max(0,x)`，两个 B 均严格为正。若共享与分方向约束都已证明且口径相同，可取各下界的最大值；若 `R_min` 或 `W_min` 未证明非零就保留符号或 0，不能自行代入模型权重文件大小。

同理，对必要计算、通信和依赖链，若各自的最小工作和对应物理容量与链间依赖均已证明，端到端下界取共同满足的约束最大值并用 DAG 的关键依赖链收紧；不能因 trace 上顺序出现就假设所有模块必须串行，也不能因理论可异步就掩盖真实数据依赖。任何 `T_min > 0` 的有效下界转为固定请求的 `TPS ≤ Q/T_min`，其中 `Q` 必须与计时窗内的成功有效输出定义一致；缺任一条件不能生成数值上界。

## 最短下一步

向资料持有人取得对应系统/板卡精确 SKU 和版本的一手技术规格或已保存资产/BOM，涵盖稠密 INT8 MAC 计数口径、频率上限、设备/卡资源映射、HBM/片内存储的方向性与共享关系、HCCS 实际拓扑和方向性 cut。逐字段核对版本和来源；在此之前继续用符号约束，不重复调用设备获取近似“实测峰值”。

## A3 带宽测试与 PCIe 信息接口的口径补查（2026-10-30）

对昇腾官网进行有限定向检索，新增两条与建界有关的接口口径：

- [MindCluster 26.1 Ascend DMI 带宽测试说明](https://www.hiascend.com/doc_center/source/zh/mindcluster/2610/toolbox/toolboxug/toolboxug_0015.html)区分 h2d、d2h、d2d、p2p；对 Atlas A3 训练及推理产品，文档中特别说明 h2d/d2h 测试项报告 HCCS 平面总带宽。d2d 描述的是同一 Device 的介质内存到芯片寄存器，p2p 描述源 Device 到目标 Device。因而这些结果的标签不能单独证明其是某条 PCIe、单向 HBM 或指定跨芯片 cut 的不可超越容量。此版本说明尚未绑定 169 历史运行所用工具、板卡及路径。
- [MindCluster 26.0 设备状态说明](https://www.hiascend.com/doc_center/source/zh/mindcluster/2600/toolbox/toolboxug/toolboxug_0019.html)将 `LnkCap Speed/Width` 描述为 PCIe 链路能力，`LnkSta Speed/Width` 描述为当前状态。这为未来限定于某条 PCIe 物理链路的上界提供接口线索；现存资料未绑定该次任务经过的全部路径、指定链路的值、编码与计数口径、双向关系和短窗口信用。它不能作为 HCCS 或片上访存界，也不能无需排查旁路就约束端到端。
- [昇腾 MindCluster 官方固定版本的 Telegraf 数据字段说明](https://github.com/Ascend/mind-cluster/blob/c38f378dbf2070399b4ade868abfdbfcaa7c8736/docs/zh/scheduling/06_api/00_npu_exporter/02_telegraf_data_description.md)列出 HCCS 单链路及总 TX/RX 带宽字段，并将 `npu_chip_info_hccs_bandwidth_info_profiling_time` 标为 1–1000 ms；[MindCluster 6.0 的 Telegraf 使用文档](https://www.hiascend.com/document/detail/zh/mindcluster/600/clustersched/usage/monitoringug/mxdlug_monit_004.html)同样将 `hccsBWProfilingTime` 列为 1–1000 ms。这些字段是采样接口而非对应短窗口的硬容量保证；两份资料未绑定本次历史采集版本，不能将指标的最高观测值当物理上限。

更正：此前此处引用的 `dlug_api_049.html` 实际正文为 Volcano 任务信息，不支持上述字段说明；原检索摘要与打开的正文不一致，已更换为核对过正文的官方来源。

本轮未找到明确绑定 169 实际 9382 SKU、固件、合法频率状态、指定物理路径及短窗口突发信用的一手硬容量证书；这是本轮有限检索结论，不是断言此类资料不存在。不调用设备，不以其他型号例子填数。要形成可用证书，仍需对同次资源映射及允许的所有旁路定义 cut，取得相应方向和状态的严格速率及初始信用，并证明该窗口必要工作量。
## 整模型结构与系统实现：主源索引（2026-10-30）

本节是对现有硬件来源索引的补充。以下 12 项来自模型/框架作者或芯片厂商；没有把不同产品的性能数字、精度路径或实现能力直接移植到 169。访问日均为 2026-10-30；未固定版本的页面标为动态版本，页面出版日期未提供则为未知。框架公开资料中的模型名称不自动证明与本次 `DeepSeek-V4.1-Flash-1017-w8a8` 权重、镜像和运行二进制完全一致。

| 主源、版本与日期 | 机制与本案具体研究方向 | 适用条件及不可推断事项 |
|---|---|---|
| [vLLM-Ascend DeepSeek V4 官方教程](https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V4.html)，文档版本 v0.13.0，日期未知 | 将 V4 列为 mHC、压缩混合注意力和 MoE，并举 A2/A3 的 W8A8 Flash 方案；mHC 的 pre/post 可围绕跨层状态和已有复合核研究，attention 必须将滑窗和压缩路径分别记账。 | V4 教程不是 V4.1-1017 架构规范；本次 40 层、压缩比、Engram 和路径以保存的 model.py 与 config 为准。 |
| [DeepSeek Engram](https://github.com/deepseek-ai/Engram/tree/fb7f84a21f91223715394a33a1dc24bbfb7f788e)，commit 2026-01-14 | 基于 N-gram 的确定性寻址与条件记忆；可研究 CPU 表的 lookup/prefetch 与设备侧计算重叠，同时保留动态 hidden 门控、版本与时序。 | 官方 demo 不等于本次运行代码；host 带宽、页映射、正确性和重叠效果都不是源文档保证。 |
| [DeepInfer V4.1 部署问答](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md)，commit 2026-10-25 | DSpark 以 confidence head 估计各位置接受率，按有效吞吐选择可验证的前缀；有助于拆分 draft、验证工作量与最终输出 token。 | 生产示例是 5 个草稿 token 的特定系统；与当前请求分布、采样契约、NPU 功能和运行参数不可等同，不能将该实现的 TPOT 迁移为本机结论。 |
| [vLLM-Ascend 量化设计](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/developer_guide/Design_Documents/quantization.html)，v0.27.1rc1，发布线日期 2026-09-24 | W8A8_DYNAMIC 表述为 INT8 权重按通道、INT8 激活按 token 动态 scale；Q_b 切 head 时研究按通道切同一权重及 scale，按 token 保持激活量化。 | 不证明当前 checkpoint 实际 scale/offset 布局、NZ packing 或位级舍入；W4A8 条目也不能单凭表推断 910C 原生指令。 |
| [Atlas 800I A3 整机规格](https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a3)，动态产品页，日期未知；[A3 算子模型与芯片核数审计](RECAPTURE-9382-IDENTITY-FREQUENCY-20261029.md) | 产品页提供整机规格，既有审计核对设备名和条件核数；建立算力、HBM 和互联分资源工作账。 | 未绑定 169 整机 BOM、单 device 计量和链路拓扑；额定频率、整机标称带宽、双向数据都不是无条件硬界。 |
| [昇腾 2201 Cube 指令理论模型](https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html)，动态开发文档，日期未知；[msprof-analyze 固定源码](https://github.com/Ascend/msprof-analyze/blob/6b8f431d4724043f44a39eef000fd5c6600784d8/msprof_analyze/cluster_analyse/cluster_kernels_analysis/operator_mfu/chip_peak_flops.py) | INT8 模型列 `16×16×32`，工具以 `16×32×16×2` ops/cycle 计；可为 small-M Q_b、MoE 按逻辑 MAC 建条件容量参考。 | 理论指令形状不是 M=3 实际利用率；未证频率硬上限、历史 kernel、运算资源上限，不能称不可超越界。 |
| [PTO A2/A3 MegaMoE 官方示例](https://pto-isa.gitcode.com/kernels/manual/a2a3/dispatch_mega_combine/)，动态手册，日期未知 | 样例将 reorder、dispatch、GMM1、SwiGLU、GMM2、combine 融合并按专家粒度交错 AIC/AIV，列 910B/910C 和 9382；可指导研究 MoE 分组就绪与依赖。 | 示例的 dtype、分组、容量与通信协议需逐项对照，不能证明 169 的 MC2 使用它，也不证明逐组交付接口或可达收益。 |
| [vLLM-Ascend GLM5 官方配置](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.md)，main 动态文档，日期未知 | A3 例子解释 FlashComm 将 AllReduce 拆为 ReduceScatter/AllGather，及融合 MC2 对专家通信的整合；比较既有 MC2 与共享专家双流时检查选路。 | 该模型/版本中 `dispatch_ffn_combine` 有 W8A8、EP≤32、非 MTP/动态 EPLB 等条件，且与 shared-expert overlap 冲突；这些约束不能机械当成本镜像的全部契约。 |
| [vLLM-Ascend Graph Mode](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/graph_mode.html)，main 动态文档，日期未知 | ACLGraph/Xlite 通过捕获重放与预分配减少部分运行期开销，可检查调度、graph 地址稳定和候选通信编排。 | 当前页面的 Xlite 支持架构名单未列 V4.1，不能认定本镜像支持；capture 前置、动态 shape 和内存预算须与实际版本核对。 |
| [SGLang Expert Parallelism](https://github.com/sgl-project/sglang/blob/f96a6e42f6048fa5175e3de603d146daaaa95a29/docs/docs/advanced_features/expert_parallelism.mdx)，commit 2026-10-30 | TBO 在 microbatch 间交错 attention 与 dispatch/combine；SBO 在同 batch 安排共享专家；Ascend DeepEP 长序列分轮通信可启发 MoE 小组/跨批流水。 | 文档将 Ascend `ascend_fuseep` 限于 PD 分离的 decode，且相关 backend 要 EP=TP；通用 TBO/SBO 描述不证明适用于本 910C/模型或现有 fused MC2。 |
| [SGLang Ascend 特性矩阵](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning.mdx)，main 动态文档，日期未知 | 单列 graph、CP/TP、量化、NZ、overlap、EPLB 的兼容性，可用来筛掉本案方案之间的冲突；研究 q_b 按 head 切分时单列 CP 与量化协作。 | 兼容表是项目版本范围的概述，CP/graph 等多为部分支持；不能将 EAGLE 的 overlap-plan-stream 断言为 DSpark/NEXTN 加速。 |
| [DeepInfer Engram 运行时](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/engram.py)，commit 2026-10-25 | 具体代码包含 hash、跨 rank 分片表读取、投影和门控；可用于列出 host/通信、投影与下游 mHC 的依赖和接口，而非重复发明同一融合。 | 其实现使用 DeepEP 和 CUDA 相关组件；不能直接在 Ascend 假定有相同 RDMA/UVA 路径、FP8 数值或性能。 |

关联当前 Q_b 候选：物理 `[3,1280] × [1280,32768]` 与按头切分后 `[24,1280] × [1280,4096]` 的条件逻辑 MAC 相等，每 rank INT8 权重本体分别为 40 MiB 与 5 MiB；权重 scale、NZ 对齐、前置全行数据汇集、Indexer、attention 元数据以及 KV/cache 的代价均另计。公开文献为寻找设计空间提供依据，不为该方案的可行性、精度或加速作证明。当前镜像的 CP 代码与 F15/F32 的关系见 `agents/fusion/F15-CP-PROJECTION-REVIEW.md` 和 `agents/fusion/SP-GATHER-QKV-PROJECTION-BOUNDARY.md`。

### 执行组织与设备接口：补充 8 项一手来源

以下固定论文版本或 commit，均于 2026-10-30 读取。论文数据来自其测试设备与工作负载，不是本机的性能预测；算法名称也不证明目标版本实际没有该优化。

| 主源、版本及日期 | 核心思想与对 169 的具体映射 | 条件、已存在功能和不适用处 |
|---|---|---|
| [PagedAttention / vLLM 论文](https://arxiv.org/abs/2309.06180v1)，2023-09-12 | KV 以逻辑块到物理块的映射存放，支持块共享、动态分配；对于 32K 输入、并发 8 的 90% 前缀命中负载，可把 prefix lookup、块表、实际命中后访问、KV 生命周期分别记入依赖图。 | 基线已记录命中计数 235904/262144，冻结启动 `--block-size 128`；缓存命中和分页不应作为新增收益再算一次。查询命中数既不是实际 prefill 计算 token 数，也不是节省 90% 总耗时；特定 DSA/SWA/压缩缓存管理以实际代码为准。 |
| [SGLang / RadixAttention 论文](https://arxiv.org/abs/2312.07104v2)，2024-06-06 | 用 radix tree 管理请求间 KV 前缀复用；在高命中负载可研究调度时请求亲和性、缓存淘汰、复制与队列之间的取舍。 | RadixAttention 是 SGLang 的组织方式，不证明当前 vLLM 使用同一数据结构。90% 已是现有缓存观测；论文 GPU 倍率、树结构与原负载不能直接换算成 169 吞吐。 |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310v3)，2024-06-17 | 将 prefill 分块并与 decode 在 token 预算下协同调度；可在并发 8、前缀命中后剩余工作与 DSpark draft/verify 共存的依赖图中研究抢占粒度和 TTFT/TPOT。 | 论文本身没有覆盖本次 DSpark/Ascend；冻结启动有 `max-num-batched-tokens=8192` 与 `--async-scheduling`，不单独证明 chunked prefill 每次如何生效。需先识别现行调度，不能重复计分块收益；命中查询差值不自动等于实际待 prefill 行数。 |
| [NanoFlow](https://arxiv.org/abs/2408.12757v2)，2025-05-25 | 对独立 nano-batch 建操作级流水，考虑计算、内存、通信共享资源；与当前 MoE 专家组流水、Q_b→attention 布局和 mHC 邻接任务可以用同一依赖图分析。 | GPU 资源划分与论文 workload 不能视作 A3 能隔离 AIC/AIV/HBM 或支持分块 ready；并发 8/DSpark 不保证总有互不依赖的可交错批次，同资源竞争可能抵消重叠。论文的“optimal”依赖其模型假设，不是本机硬上界。 |
| [DeepEP 官方实现](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/README.md)，commit 2026-10-21 | dispatch/combine 返回异步事件，延后到数据实际消费时等待；可用于描述 MoE 路由、expert compute、共享专家和归并之间的 ready/资源依赖。 | 该 README 的 GPU 路径要求 Hopper/CUDA/NVLink；另列的 DeepEP-Ascend 支持 950，不是 910C 已可调用接口。当前 `DispatchFFNCombine` 已是复合任务，不能把 dispatch、compute、combine 算成必然独立、可完全隐藏的时间。 |
| [DeepGEMM 官方实现](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)，commit 2026-10-20 | 同 N/K 的专家按 M 轴分组；mask 可用于设备端有效专家行数不同的 graph decode。启发将 MoE 真实 expert 行数、padding、权重与 scale layout 纳入账目；Q_b 需独立审视小 M 到较大 M 的布局变化。 | 该实现的 grouped FP8 kernel 面向 SM90/SM100 GPU；不等同 910C W8A8 grouped GEMM。MoE 当前已使用融合路径，不能把分组本身重复登记为新优化，也不能在未知有效行时填写工作量。 |
| [FlashMLA 官方接口及支持范围](https://github.com/deepseek-ai/FlashMLA/blob/395cb3afbb331fef7b1b21e9340861b9436b42d8/README.md)，commit 2026-10-27 | sparse attention 接收 indices 与 KV block 信息；可比较 Q_b 的 head 分片、跨 rank index 分发及 DSpark 多 query 行的 attention 数据依赖。 | README 的 Ascend 版本面向 **950**，CUDA 的 norm→RoPE→attention→RoPE→cast 融合声明不支持 Ascend；不能视为 910C 已具备此 kernel。其 V4.1 FP8/FP4 KV 格式和本案实际缓存 dtype、量化语义亦不可互换。 |
| [FlashMLA Ascend 稀疏注意力设计](https://github.com/deepseek-ai/FlashMLA/blob/395cb3afbb331fef7b1b21e9340861b9436b42d8/docs/20260930-ascend-prefill-deep-dive.md)，发布 2026-09-30 | 讨论 Cube/Vector 分工、异步拷贝、管线 issue 顺序与内存资源，适合作为分析 attention 和通信/计算能否交叠的方法参考。 | 文档明确讨论 **Ascend 950** 的内核、内存与队列；其 1C2V 配置、拷贝接口和性能数值不能直接转成 910C 的硬件参数或本案理论界。当前 QLI 已有 Cube/Vector 双槽和排序归并，不能当作尚未实现的全新流水。 |

以上与现有 `evidence/frozen-serve.sh` 中的 DSpark 5、异步调度、DSA-CP、FlashComm、fused MC2 和 graph 配置是**概念关联**；配置不等于每次动态分支或物理重叠均已观测。对 Q_b 前移分片，仍需联合计入全 token 的 qr 获取、索引、cache/metadata 所有权及输出 head 布局；对 MoE 则需把当前 fused MC2 内部和可分批任务的竞争共同记账，不能叠加两个方案的理论收益。

