# 一手资料补充与本机证据的再映射

检索日：2026-10-30（Asia/Shanghai）。使用 Web Search、论文原站、官方源码/文档；没有打开 Browser/Chrome/Computer Use。下表与[原资料库](../publication-draft/REFERENCES.md)配合使用。原库含 42 个去重 URL；数量代表链接覆盖，不能替代阅读全文或证明历史资料穷尽。

## 检索与筛选方法

按模型结构、服务调度、注意力/索引、MoE/通信、投机解码、量化、Ascend graph/profiling 七个机制域检索；同时查 GPU 论文中的可迁移数据依赖方法和 A3 的真实接口限制。纳入模型/框架作者、论文原站、芯片厂商资料；未用媒体、转载和搜索摘要支持本机性能结论。

本次代表查询包括：`COMET fine grained communication computation overlap mixture experts`、`FLUX kernel fusion`、`mHC Manifold Constrained Hyper Connections`、`Engram Conditional Memory Scalable Lookup`、`FlashInfer plan CUDAGraph workspace`、`ORCA iteration scheduling`、`speculative decoding exact sampling`、`SmoothQuant W8A8`、`vLLM Ascend profiling`、`ACL Graph host parameter updates`。

阅读深度分别注明：摘要用于筛选；具体设计结论来自对应全文小节/源码，不把仅检索到的标题视为已完成技术核查。与已有融合重复、要求更换模型/量化语义、仅提供其他硬件倍率的方案不登记为本机已验证优化。

| 一手来源、版本 | 已读取范围与机制 | 对本任务的具体取舍 |
|---|---|---|
| [FlashInfer，arXiv:2501.01005v1，2025-01-02](https://arxiv.org/html/2501.01005v1) | 全文 §3.3/3.4 与 Appendix D.1：运行时每步做 plan，兼容的层共享计划；持久 kernel 和固定 workspace 偏移保持 graph 地址稳定。 | 新发现的 `prepare input` 与 F33 应用此方法检查哪些 plan/坐标真实相同。当前 v4.1 源码已有批次和组内共享；进一步研究 A3 多 group 的批量准备。FlashInfer 的 CUDA 实现和倍率不适用于本机直接替换。 |
| [Comet，arXiv:2502.19811v1，2025-02-27](https://arxiv.org/html/2502.19811v1) | 全文 §3.1：按消费者的独立维度拆共享张量，重新组织符合 GEMM tile 的任务；输入传输与 GMM 适合 token/行方向，top-k combine 的依赖可能要求沿输出通道交付。 | 比“把 MoE 再融合”更具体：核查 MC2 的组级/通道级 ready 与缓冲生命周期。消费者完整 reduction 之前不能提前发布结果；本机有效专家行数、内部阶段和硬件资源争用仍须实测。 |
| [FLUX，arXiv:2406.06858v1，2024-06-11](https://arxiv.org/html/2406.06858v1) | 全文 §2.3 与摘要：通过 kernel fusion 缩小计算与通信重叠粒度；需用整体时间评估拆分 GEMM 的效率损失。 | 研究 output projection→reduce-scatter、query gather→GEMM 的边界。A3 的 collective/MC2 已有融合，需逐接口确认可提前消费；GPU 算子、链路和论文倍率不移植。 |
| [mHC，arXiv:2512.24880v1，2025-12-31](https://arxiv.org/html/2512.24880v1) | 全文 §4.3 的融合、混合精度、重算和 DualPipe 方法；模型通过约束残差映射保持稳定性。 | HcPre/HcPost 在主模型轨迹已存在，优化对象是现有内部阶段和跨边界状态。训练重算与 DualPipe 不能自动迁移到低并发解码；保持迭代归一化、FP32 系数及舍入语义。 |
| [Engram，arXiv:2601.07372，2026-01-12](https://arxiv.org/abs/2601.07372) | 摘要：确定性 N-gram lookup 与条件记忆，地址可预测有利于 host 预取。 | 与本机 UVA gather/大投影分别记账。预测地址不代表动态门控和后续投影可省，也不证明当前访问实现无开销；先核实真实 host 路径与预取期限。 |
| [ORCA，OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | 原论文摘要/入口：迭代级调度与选择性 batching。 | 当前已有连续调度、async batch queue；C8 封闭负载下不能将这些当新增功能。提示按完整迭代分析，不能仅以 40 层主流片段评估系统。 |
| [DistServe，arXiv:2401.09670，2024-01-18](https://arxiv.org/abs/2401.09670) | 摘要：分离 prefill/decode 资源与并行策略，减轻相互干扰，同时计入 KV 迁移。 | 本任务只有本人 169，且 90% 前缀复用；先比较单机内分工和 chunking 的真实代价，跨机 PD 分离仅作扩展方案，不能未经资源账就列为优先部署。 |
| [Fast Inference from Transformers via Speculative Decoding，arXiv:2211.17192，2022-11-30](https://arxiv.org/abs/2211.17192)；[Speculative Sampling，arXiv:2302.01318，2023-02-02](https://arxiv.org/abs/2302.01318) | 原摘要：草稿候选与目标并行验证，接受/校正须保持目标分布。 | DSpark 已在基线；优化目标是单位最终输出 token 的草稿、验证和准备成本。不能只最大化接受率或把一轮验证行数当输出 token。概率重排/融合必须保留 RNG、拒绝及回退语义。 |
| [EAGLE，arXiv:2401.15077，2024-01-26](https://arxiv.org/abs/2401.15077) | 摘要：利用目标特征预测草稿并处理特征不确定性。 | 提供不同草稿路径的比较视角；没有证明本次 DSpark 可直接换用 EAGLE，也不能把另一草稿模型的精度或速度当本机证据。保留为算法比较，不做本次实现建议的前提。 |
| [SmoothQuant，arXiv:2211.10438，2022-11-18](https://arxiv.org/abs/2211.10438) | 摘要：通过等价缩放处理 activation outlier，使 W8A8 更易执行。 | 当前权重与动态量化契约已冻结。量化融合应核实 scale/offset、约化域和数据布局，不能因论文存在 W8A8 就推断本 checkpoint 使用同一校准方法；改量化方案属于另一个精度验收任务。 |
| [vLLM-Ascend 官方 Service Profiling Guide，main 动态页](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html) | 页面明确 profiler-config、start/stop 接口、官方 analyse 导出及文件用途。 | 本研究复用安装版本产生的 CSV/trace；区分前端 trace 与实际 worker trace。本次 worker trace 已找到 cpu_op 活动，不能因前端文件只有包络就否认 worker 证据。 |
| [vLLM-Ascend RFC #13058：ACL graph host 参数更新](https://github.com/vllm-project/vllm-ascend/issues/13058) | 议题正文列出隐式 task 身份、参数/handle/event 平行列表和 backend 负责整图编排的风险。 | 帮助约束 F33 与 metadata graph 重构：稳定任务身份、校验列表长度、明确 producer/event 与混合 backend 边界。RFC 不是已合入功能或本镜像缺陷证明；以冻结源码核对。 |

## 文献使本轮排序发生的变化

FlashInfer 的价值在于把动态计划与执行分开，并以受控地址发布更新。它与本机重复准备事件直接相关，因此 F33 从一般的“调度开销”问题收敛为 group-specific slot mapping 和 metadata 生产边界。源码已共享的部分不重复计收益，跨组物理坐标不共享。

Comet/FLUX 则让 MoE 和输出投影候选更具体：拆分依据是消费者可独立执行的数据维度，粒度须同时满足计算效率和通信 ready 接口。当前 `DispatchFFNCombine` 的外层耗时只说明值得调查，不足以证明内部阶段可按论文方式重排。

投机解码资料要求以**最终被接受并输出的 token**归一化成本。新 worker 测量轮的 accepted/draft=6445/8790 是草稿 token 计数比，不等于整请求加速、每轮接受长度或每条输出 token 的完整工作量。后续需要把每次 draft、verify、prepare 与实际输出绑定。

## 后续覆盖检查

原库已覆盖 PagedAttention、RadixAttention、Sarathi-Serve、NanoFlow、FlashAttention、FlashAttention-3、FlashMLA、DeepEP/DeepGEMM、MegaBlocks、Tutel、RadiK/RTop-K、Ascend C/MrgSort/PTO 和 Roofline/I/O 理论。以上补充了 plan/run、MoE 共享张量分解、mHC/Engram 原论文、早期投机解码和量化基础。

机制域及同场景复析已整合进[主报告](../publication-draft/RESEARCH-REPORT.md)第 11–12 节；[去重检索索引](reference-index.json)保留条目出处。阅读范围与版本以各表格为准，链接数量本身不作为技术判断的证明。
