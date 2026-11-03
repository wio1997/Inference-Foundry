# EP8 attention 输出投影：单次 BF16 GEMM 条件计算模型

## 样本与能证明的范围

已保存的 EP8 第 0 层选中设备 task 43440 是 `aclnnMatmul_MatMulV3Common_MatMulV3`，采样 CSV 报告输入 BF16 `[8184,1024]`、`[5120,1024]`，输出 BF16 `[8184,5120]`，该次设备任务区间 320.893 µs。它的唯一 host-to-device flow 起点被同一 PID/TID 的 `vllm::unquantized_gemm`、`aten::linear`、`aten::matmul` 及 `aclnnMatmul` host 区间包含；`aten::linear` 报告相同的 BF16 输入形状，`aten::matmul` 报告右输入 `[1024,5120]`。这将该样本关联到一次 host 的非量化 BF16 线性调用，比仅从权重描述推断更强。Host API 类型、设备导出的形状与类型不证明内部实际执行的指令、累加类型、张量 storage、二进制版本或所有实现方式。trace 仅是同一轮重型 profiler 样本，不代表无 profiler 基线。

## 受限模型与算术

若限定采用传统的直接稠密矩阵乘、每个输出元素完整使用全部 1024 个乘加，该次逻辑工作为 `8184×5120×1024 = 42,907,729,920 MAC`。这里 MAC 是一次乘加，计作一个 MAC；若将乘与加分别计数，FLOP 数是两倍。这是指定算法的逻辑工作，不是已测物理指令数，也不是所有可能合法算法的必需工作。

进一步假设：所有工作只能由一个至多 24 个 Cube 核的同一资源池完成；每核频率的硬上限为 1.8 GHz、每核每周期至多完成 4096 个该种 BF16 MAC；对所选时间窗口没有在途工作或突发信用；不允许其他资源承担计算，且忽略调度、访存和通信，那么理想计算时间有条件地满足：

`T ≥ 42,907,729,920 / (24 × 1.8×10^9 × 4096) ≈ 242.489 µs`。

24 核、1.8 GHz 来自现有采集/设备元数据，但不是已证实的所有运行状态的硬上限。4096 是依据 [Ascend 2201 BF16 Mmad 并行度资料](https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html) 构造的单路径模型假设；参见 [硬件证据审计](hardware_sources_audit.md)。该文档本身不证明其他指令、Vector 等资源不能并行参与，也未证明总设备硬容量、指令服务间隔及所选 kernel 实际执行方式。窗口若存在在途或突发信用 `b`，必须使用 `T ≥ max(0,W−b)/P`，不能默认零信用。

在这些未验证的模型条件下，242.489 µs 与 profiler CSV 的 320.893 µs 数值比约 0.755669。**它只是理想模型时间与一个受 profiler 干扰的任务区间之比，不是 NPU 利用率、效率、可达到的提速或该任务已严格证明的物理性能界。** 该形状、候选算子和窗口不绑定业务基准的全部请求与层，不能换算成整个服务的 TPS 上界。BF16 输出舍入、权重初态、实际 tile、其他资源及更一般的计算方式均未闭合；这也不证明任何 fusion 一定可行。

复现：`python3 agents/bounds/attention_projection_ep8_ideal_model.py`。脚本校验既有两个输入 SHA、host 包含关系、选中 task、flow 唯一性、物理类型/形状以及算术；脚本 SHA-256 `b96dbf26236f563b3327a13d1dcf8e95f88ae151b83baf6acce77dcd828e0354`。不读取权重内容、不触 NPU。原始时间与内部路径保存在任务目录，不发布。

## 已存 BF16 二进制候选与 tiling 分支（2026-10-30）

对照已存同环境 `precision-paths-20261028/INSTRUCTION-EVIDENCE.json` 和本机冻结的 `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16_high_performance.o`：其元数据声明必需输入 x1/x2 为 BF16 ND、输出 y 为 BF16 ND、可选 bias 为 BF16 ND；二进制 SHA-256 `acb4812b8f02df8a6ee3c8dcc0d36e6332e318cfec3c5b984cdd731d92d3e29d` 与清单一致，保存的反汇编摘要含 `MMAD.bf162f32`。设备任务 43440 导出 x1/x2 均 BF16 ND、输出 BF16 ND；对应 `vllm::unquantized_gemm` 和 `aten::linear` 的 host `Input type` 均列两个 BF16、第三个 `None`，与无 bias 的线性调用相容。该同族二进制提供了较具体的可用实现候选，但同名、dtype、格式相容与静态反汇编**不足以证明该历史任务加载这份二进制、命中该指令或以 FP32 作实际累加**。host `None` 也不自动证明设备执行图没有其他隐式处理。

这个候选对象的保存元数据列出 **17 个不同 tilingKey**，其中 14 条 `taskRation=1:2` 且 `crossCoreSync=1`，3 条分别是 `1:0` 与 `0`；顶层 `taskRation=tilingKey`。它表明相同对象清单内有多种调度元数据，不能凭 BF16 和 ND 就唯一推定 tile、是否跨核同步、资源占用或数据搬运方式；这里仅转述字段，不把 `1:2` 解释为真实核数或共享量，也不将某值绑定选中 task。其 supportInfo 的输入 shape 是通配 `[-2]`，属性中 `transpose_x1=false, transpose_x2=false` 是该保存元数据的字段；选中 host 的 `aten::linear` 记右输入 `[5120,1024]`，同调用链的 `aten::matmul` 记 `[1024,5120]`，设备记录又是 `[5120,1024]`。这些记录不含完整 stride、view 和实际 transpose 参数，不能简单地把元数据值当历史 task 的属性，也不能据此认定形状冲突或相同张量已绑定。需要历史任务实际 kernel 标识、二进制哈希、tilingKey 及完整调用属性/布局，才可决定是否适用此对象中的哪一项。

这一核查改善了“同族可用指令与调用语义”的证据，但不把 2201 单指令理论周期升级为所有设备计算路径的硬最大速率，也不确立实际工作量、持续最高频或短窗口突发预算；`242.489 µs / 320.893 µs` 仍是前述严格条件下的理想算术对照。计算、搬运可以按 tile 交叠，不能将候选 metadata 中的同步标记或整个逻辑权重体积直接相加为不可避免耗时。反汇编摘要 SHA-256 `e810f99478fa32263ed00b0366855afe73539ddb68ccc2ddbe70d59c85bc52a3`；原始记录路径和方法见前文及 `hardware_sources_audit.md`。本次只读既有证据，无设备访问。
