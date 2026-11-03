# 模型、推理系统和算子优化：一手资料索引

对应 [169 研究报告](README.md) 与 [融合和重组清单](FUSION-INVENTORY.md)。访问/整理日为 2026-10-30。目录覆盖本次实际算子和执行组织相关的模型、厂商、框架与论文；复用历史任务资料后补检索，没有声称穷尽所有历史文献。

检索主题包括 DeepSeek/DSpark/Engram、Ascend A3 的 INT8 与存储/同步接口、KV/前缀缓存、稀疏注意力、TopK、MoE、small-M 与权重复用，以及图执行、分块与通信重叠。优先一手论文、厂商文档和固定版本代码；不把二手博客中的倍率当成本机证据。

资料使用分三层：本机源码与 trace 用于实际工作账；产品/API 文档用于判断可用机制和限制；其他硬件论文用于产生或排除候选。不同版本、设备、数值契约和负载的收益不能直接相加或迁移。未能绑定历史二进制的地方在报告保留为条件。

## 官方采集入口与版本限制

- [vLLM ProfilerConfig](https://docs.vllm.ai/en/stable/api/vllm/config/profiler/)：服务启用与客户端 start/stop 的参考；冻结版本的字段和自动停止条件以本机代码为准。
- [vLLM-Ascend Service Profiling Guide](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html)：服务化及算子采集方法；在线 main 不等同本任务镜像版本。
- [Ascend PyTorch Profiler](https://www.hiascend.com/document/detail/en/mindstudio/2600/TITools/ascend_pytorch_profiler/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)：CPU/NPU trace、Level1 与流水计数的官方说明；实际 collector 是本机 9.2/torch_npu 2.10 的冻结实现。
- [CANN 8.0 最小扰动说明](https://www.hiascend.com/doc_center/source/en/canncommercial/800/devaids/profiling/atlasprofiling_16_0134.html)：跨版本的采集成本参考，不能作为 9.2 行为证明。
- [CANN 9.2 beta1 AI 任务采集](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0011.html)：相近版本参数参考，具体支持范围仍核对安装工具。
- [MrgSort A3 接口文档](https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/sort_and_merge_ISASI/MrgSort.html)：独立队列长度、对齐、重叠与同分优先级；页面为持续开发预览，不代替本机编译验证。

npu-smi 使用本机随驱动提供的版本及帮助；其 AICore/显存快照不等同 kernel 计算效率、HBM 带宽或细粒度等待分解。官方接口、采集参数和测量扰动均在报告单列。

## 整模型结构与系统实现：主源索引（2026-10-30）

本节是对现有硬件来源索引的补充。以下 12 项来自模型/框架作者或芯片厂商；没有把不同产品的性能数字、精度路径或实现能力直接移植到 169。访问日均为 2026-10-30；未固定版本的页面标为动态版本，页面出版日期未提供则为未知。框架公开资料中的模型名称不自动证明与本次 `DeepSeek-V4.1-Flash-1017-w8a8` 权重、镜像和运行二进制完全一致。

| 主源、版本与日期 | 机制与本案具体研究方向 | 适用条件及不可推断事项 |
|---|---|---|
| [vLLM-Ascend DeepSeek V4 官方教程](https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V4.html)，文档版本 v0.13.0，日期未知 | 将 V4 列为 mHC、压缩混合注意力和 MoE，并举 A2/A3 的 W8A8 Flash 方案；mHC 的 pre/post 可围绕跨层状态和已有复合核研究，attention 必须将滑窗和压缩路径分别记账。 | V4 教程不是 V4.1-1017 架构规范；本次 40 层、压缩比、Engram 和路径以保存的 model.py 与 config 为准。 |
| [DeepSeek Engram](https://github.com/deepseek-ai/Engram/tree/fb7f84a21f91223715394a33a1dc24bbfb7f788e)，commit 2026-01-14 | 基于 N-gram 的确定性寻址与条件记忆；可研究 CPU 表的 lookup/prefetch 与设备侧计算重叠，同时保留动态 hidden 门控、版本与时序。 | 官方 demo 不等于本次运行代码；host 带宽、页映射、正确性和重叠效果都不是源文档保证。 |
| [DeepInfer V4.1 部署问答](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md)，commit 2026-10-25 | DSpark 以 confidence head 估计各位置接受率，按有效吞吐选择可验证的前缀；有助于拆分 draft、验证工作量与最终输出 token。 | 生产示例是 5 个草稿 token 的特定系统；与当前请求分布、采样契约、NPU 功能和运行参数不可等同，不能将该实现的 TPOT 迁移为本机结论。 |
| [vLLM-Ascend 量化设计](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/developer_guide/Design_Documents/quantization.html)，v0.27.1rc1，发布线日期 2026-09-24 | W8A8_DYNAMIC 表述为 INT8 权重按通道、INT8 激活按 token 动态 scale；Q_b 切 head 时研究按通道切同一权重及 scale，按 token 保持激活量化。 | 不证明当前 checkpoint 实际 scale/offset 布局、NZ packing 或位级舍入；W4A8 条目也不能单凭表推断 910C 原生指令。 |
| [Atlas 800I A3 整机规格](https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a3)，动态产品页，日期未知；本机已有芯片身份/核数审计（原始记录保留在任务目录） | 产品页提供整机规格，既有审计核对设备名和条件核数；建立算力、HBM 和互联分资源工作账。 | 未绑定 169 整机 BOM、单 device 计量和链路拓扑；额定频率、整机标称带宽、双向数据都不是无条件硬界。 |
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



## 已有设备样本对应的一手文献复析（2026-10-30）

本节收敛已登记候选，不代表已经测量性能收益；下列 GPU 论文的具体性能结果和硬件指令不能直接迁移到 910C。当前仅复用保存资料、文献和官方文档，未运行 NPU。样本窗口的计时、不同类别与整请求不可相加。

1. **FlashAttention，NeurIPS 2022。** [原论文](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) 用分块和在线归一化，减少大注意力中间矩阵在 HBM 和片上存储之间的往返。关联本任务 F13 已有 `SparseFlashMla` 复合边界：28 个选定 Indexer→整理→attention 窗中 Indexer 占比中位 82.264%，该范围不能推出 attention 本身是首要瓶颈。不能将整套 dense FlashAttention 的 IO 省量重复记在已经融合的 sparse kernel 上；还须先绑定实际稀疏索引、KV 页、mask/sink、缓存及中间输出，才能判断是否仍有可消去的 GM 往返。

2. **FlashAttention-3，NeurIPS 2024。** [原论文](https://papers.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf) 利用跨迭代缓冲，在不同块之间交错矩阵、softmax 和搬运；这是跨块流水，不是取消同一块的生产者依赖。对应 F21 MoE dispatch→GMM1→激活量化→GMM2：保存代码已按专家组重叠 dispatch 与 GMM1，而 AIV 激活受 dispatch 循环及同步限制。可研究下批 dispatch 与前批激活交错，待证明 AIC/AIV 资源、共享 UB、scale、workspace 别名和 event；不能假设原实现完全没有重叠，也不能将 Hopper TMA/WGMMA 的结果套到 Ascend。

3. **昇腾 Ascend C Matmul 异步，CANN 社区版 8.1.RC1 beta1。** [官方接口说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0077.html) 对 MIX 场景列出 Matmul Iterate/IterateAll 的同步及异步模式，纯 Cube 场景只支持同步模式。该资料限定重排条件，不能据此断言实际 910C、历史 CANN 或此 MC2 二进制支持某个异步变体；应绑定设备版本、所用 API、每个队列的完成事件及 workspace，减少不必要等待必须与同步安全共同证明。

4. **RadiK，ICS 2024，arXiv 2025-01-24 版本。** [论文](https://arxiv.org/abs/2501.14336) 以 radix 选择和批处理资源调度降低大规模 top-k 对片上列表容量的压力，并讨论数据分布、直方图和中间输出开销。当前 QLI 的 tile 长至 2048、候选最终输出形状 512，源代码逐 tile `SortAll` 再保留全局 2048，最终只写 `sparseCount`。应先核定实际 sparseCount、有效长度、最终是否要求排序和 tie/无效值语义；相比先改成新选择算法，保留 tile、仅缩短 token 累加列表是更收敛的条件候选。论文部分量化实验明确不计输出排序，且主要展示不同规模的 NVIDIA GPU，不能把速度或不排序输出直接用于本路径。

5. **RTop-K，ICLR 2025。** [原论文](https://arxiv.org/abs/2409.00822) 以阈值搜索做逐行选择，可选择近似 early stopping，目标应用不要求最终 k 个元素有序。与上述 QLI/512 同属按行选择，但本实现还包括 TopK 后索引整理、跨层候选复用、有限负惩罚与 sentinel：若下游依赖顺序、tie 或逐项等价，选集合和允许近似都不够。故它是候选生成方法，不是替代当前有序精确语义的现成证明；必要代价包括阈值查找、重复扫描、边界同分处理和排序恢复。

6. **CANN MrgSort，商用版 8.0.0。** [官方 API](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0232.html) 将 `elementLengths` 定义为各输入序列的 8 字节结构个数，理论范围 0–4095 且受 UB 容量等限制；这只支持检查混合 2048 与 512 在数值范围内，并不单独证明本设备、实际版本、对齐和异长组合合法。保存 `vector.h:152–169` 中两路分别为 tile 长与累加长，输出 scratch 容纳二者总和，且只回拷累加长；满 tile 条件下输入从 4096 对降至 2560 对、回拷从 2048 对降至 512 对，最终提取可从 2048 降至 512；**不能因此说 sort tile 不再执行或内核实现只输出 512 对**。candidate block 单独保持 2048，LD 工作区/四路归并、tie、sentinel、异步回拷与消费者须保持。

7. **MegaBlocks，MLSys 2023。** [论文](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) 将不同专家的动态 token 分布表示为块稀疏矩阵运算，以减少固定容量 padding 和不丢 token 的代价。对应主模型 MC2 在三个选定 40 层主流 span 分别占 35.89%/35.88%/36.30%，以及草稿 162 窗 MC2 时间和占窗口时间和 48.47%。当前 MC2 早已是分组矩阵及通算复合路径；实际每专家 token 数、tile 及 padding 未绑定，不能据论文假定已经存在固定容量浪费或能直接改成 GPU block-sparse 提速。

8. **Tutel，MLSys 2023。** [论文](https://proceedings.mlsys.org/paper_files/paper/2023/file/5616d34cf8ff73942cfd5aa922842556-Paper-mlsys2023.pdf) 针对动态专家工作量，研究可切换并行和 dispatch、专家计算的分块通信流水。对应 F21 和 DSpark 草稿 162 窗中位 723.376 µs；只能在已存在的通信计算交叠基础上寻找进一步重叠，并考虑各 rank collective 顺序、数据就绪、workspace 活期、计算/网络共享资源及流水首尾。论文训练 GPU 场景的加速和硬件拓扑不能移植为本次数值；草稿每窗 MC2 中位 181.186 µs 与窗口中位不是同一窗口配对。

9. **vLLM 作者 Berkeley 技术报告 2025，算术强度模型。** [报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/Archive/EECS-2025-192.pdf) 指出在假设 BF16 权重每批只读一次时，矩阵乘算术强度随参与的有效输入行数增长。关联 Engram 形状相容但身份未绑定的 `[3,6144] × [25600,6144]ᵀ`，其 96 条任务中位 275.765 µs；矩阵权重逻辑大小为 `6144×25600×2 = 314,572,800 B`，单次逻辑工作 471,859,200 MAC。针对共同权重、同时就绪的多个输入行可考虑分批复用，代价是排队/TTFT、快照和 buffer 生命周期；不能假设两处 Engram 候选权重相同或从 HBM 读取，更不能用逻辑权重字节数除以耗时当实测 HBM 带宽。

数据依据：`publication-draft/README.md`、`publication-draft/FUSION-INVENTORY.md`、`agents/fusion/F21-MC2-DISPATCH-ACTIVATION-PIPELINE.md`、`agents/diagnosis/ENGRAM-MATMUL-DISPATCH.md`、`agents/diagnosis/module-priority-evidence.md:298–307`；历史 CUDA/GPU 论文、不同 CANN 版本及该机实际运行边界不同，留待逐项身份与接口核对。

## 理想模型的理论来源与使用限制

| 一手来源 | 本任务的用途 | 不作出的推论 |
|---|---|---|
| [Williams/Waterman/Patterson，Roofline，UCB/EECS-2008-134](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)，2008-10-17；后发表于 CACM 2009 | 用计算吞吐和算术强度/带宽共同筛查 Q_b、Engram 和 MoE。计算能力、带宽及字节数采用一致边界。 | 一次最快实测不是硬容量上限；不以 Roofline 单独证明调度和通信最优。 |
| [Hong/Kung，I/O Complexity: The Red-Blue Pebble Game，STOC 1981](https://doi.org/10.1145/800076.802486)；[作者发表目录](https://eecs.harvard.edu/htk/publications/) | 在固定计算 DAG、片上容量和存储初态下研究必要 I/O，检查融合是否减少必须搬运的中间量。 | 普通矩阵乘算法的下界不能直接套到任意算法；checkpoint 大小和当前 workspace 也不是普适必读 HBM 字节。 |

本研究保留适用域中的条件/符号时间下界与吞吐上界。尚无已绑定本机、所有必要资源容量的不可超越全模型数值；这不阻止基于测量给出可审查的优化机会，也不把名义计算理想值称作可兑现收益。

## 补充：独立引擎进程的 CPU 采样

- [vLLM 官方 Python profiling 说明](https://docs.vllm.ai/en/stable/contributing/profiling/)：Python CPU 与 worker 设备 profile 应区别处理；在线 stable 不是冻结版本行为保证。
- [py-spy 官方说明](https://github.com/benfred/py-spy)：可以从目标进程外读取 Python 栈，支持 PID 与非阻塞模式；非阻塞可有不完整/失败样本，等待栈的样本比例不是调度浪费或可删除时间。
- [py-spy 0.4.1 官方 PyPI 包](https://pypi.org/project/py-spy/0.4.1/)：提供 aarch64 wheel，SHA 与下载源留在任务记录。工具只解包到本人分析目录，不更改推理依赖或宿主环境；是否实际采样及覆盖以报告为准。
