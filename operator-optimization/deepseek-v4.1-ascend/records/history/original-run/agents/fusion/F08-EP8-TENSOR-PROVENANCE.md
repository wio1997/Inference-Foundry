# F08：EP8 第 14 层三条任务的调用与张量身份

本次只读检查了已绑定的 EP8 L14 scope `737-20`、stream 35、task 13274–13276；未运行新 NPU 负载。复算脚本 `f08_ep8_tensor_provenance.py`、完整聚合 `f08-ep8-tensor-provenance.json`，脚本 SHA-256 `e6baab58815fb0e7bbd759a0844f3a01b619166fa182d7e06de213fe2586e22d`。两次独立运行聚合 JSON 相同。输入是同次采集 EP8 `trace_view.json`（SHA-256 `53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23`）与 `ascend_pytorch_profiler_8.db`（SHA-256 `e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486`）。

将三个 task 的精确开始时刻分别与 torch_to_npu flow 的终点匹配，再以其唯一 flow 起点查同线程 PyTorch API 的时间包围关系：

| 设备 task | flow 起点处 API | 上层记录 |
| --- | --- | --- |
| 13274 RmsNorm | `aclnnRmsNorm`、`npu::npu_rms_norm` | `vllm::dsa_v41_forward`、`DeepseekV41DecoderLayer.forward:737-20` |
| 13275 InplacePartialRotaryMul | `aclnnInplacePartialRotaryMul`、`_C_ascend::inplace_partial_rotary_mul` | 同一个 `vllm::dsa_v41_forward`、同一层 scope |
| 13276 ScatterNdUpdateSk | `aclnnScatterNdUpdateSk`、`_C_ascend::npu_scatter_nd_update_sk` | 同一个 `vllm::dsa_v41_forward`、同一层 scope |

这证明选中三个设备任务分别具备相应的同采集上层 API/flow 关联，且 flow 起点都被同一个 attention 调用和层范围包围；比单纯名称相近、形状相容有更具体的调用层级证据。三条 CANN `launch` 均能以各自 connection ID 精确关联，但同 connection 的 PyTorch 查询均为空；这不排除通过其他 ID 表或其他机制关联。`PYTORCH_API` 仅包括输入形状、dtype、callchain 等，`COMPUTE_TASK_INFO` 有输入输出形状、dtype、format、空 attr，没有 tensor 地址或 storage identity 字段。三条匹配调用的已导出 callchain 均只有 `torch/_ops.py:1209 __call__`，并未定位到具体源码语句。

因此**仍不能证明** norm 输出与 rotary 输入、rotary 输出与 scatter 值是同一存储，也不能排除这些任务在该调用内部作用于相同形状的不同张量；不能因为同流或 task 连号替代数据依赖证明。已冻结源码中该顺序的条件分支与观察相容，但本次 trace 没定位具体分支。若将来允许采集，需记录张量的 storage 身份、offset/stride 和调用上下文，或得到直接证明运行时分支和等价数据流的证据；当前不发起新采集。全模型、所有 rank 或 graph replay 均不在本次检查范围。

## 源码条件下的视图与逻辑依赖

另查已冻结的 `sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py`，SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。普通 `_project_kv` 第 327–330 行与 multistream 第 487–490、493–494 行分别把 `kv_norm` 的 Python 返回张量经 `.view(-1, 1, head_dim)` 传给原地旋转。`device/device_op.py` 第 76–91 行（SHA-256 `01ce13ea60e46f901ad69ad4c71e4be0e31cd69fbc56ebb7d898f957d5e54f67`）将 3 维张量 `unsqueeze(1)` 后作为旋转算子的目标参数。**条件结论**：假如执行的是其中一个分支且 `.view` 成功，Python 层 `kv_norm` 返回值与旋转目标参数是同一底层 storage 的不同视图；这里不依赖 shape 相同来判断。普通路径后续的 `squeeze(1)` 以及多流分支同名操作也是视图。

在标准非 packed 写缓存分支，`dsa_v41.py` 第 297–308、239–242 行又将该逻辑值经 `values.to(cache.dtype).contiguous()` 作为 scatter 的 updates；这两个调用可以不复制，也可以发生类型转换或生成连续副本。**即使 updates 与旋转目标 storage 不同，也不反证逻辑生产者—消费者关系**；反过来，源码里的可能依赖不能证明 EP8 L14 三个具体设备 task 采用了这一分支或同一条调用链。cache 本身亦可能是较大分配的视图，形状不足以确定地址或页 stride。

额外条件：如果运行时 `kv_norm` 确系 `ops/layernorm.py` 中 `AscendRMSNorm.forward_oot` 且 residual 为 `None`，第 99–103 行将 `torch_npu.npu_rms_norm` 的第一个返回张量直接或经原地 bias 更新后返回；该文件 SHA-256 `d2fda42ad99935725c98b9505a8d2e94696d9cdb2906e5cb2f1e37bf21fb27bc`。此条件仍不足以将选中设备 task 13274 的输出绑定到该 Python 张量：运行实例及具体 API/设备输出缓冲尚未以张量身份证据绑定。

已有 EP8 已解码的一份 CANN `tensor_info` slice 只提供格式、dtype、shape，未绑定这三条调用；尚未逐字节穷尽其他 slice 和未知编码。故准确结论是**目前查验的调用/数据库字段和该解码样本没有可绑定的 storage 身份**，不是断言所有原始记录都不存在地址。无新增 NPU 采集或目标实现修改。

## EP8 L14：选中样本的数据流与时间口径（离线汇总）

下表把本报告的 flow 匹配与 `f08-ep8-pipeline-overlap.json` 的同一 scope 设备区间放在一起。形状、dtype、格式是已导出的算子参数签名；箭头表示**冻结源码所述的条件路径**，并不额外证明选中设备任务的 storage 或历史运行分支。三个设备区间使用同一 trace 的十进制时间；与 DB 整数纳秒换算可能有亚纳秒舍入差。

| 环节 | 已记录签名及目的 | 设备记录时长 | 数据来源和去向的证据边界 |
|---|---|---:|---|
| 输入及投影 | 同 host scope 的 `dsa_v41_forward` 输入 BF16 `[8184,5120]`；本表没有绑定本次具体 `wkv` 设备任务 | 未绑定 | 源码普通路径从 `cache_hidden` 按 `num_actual_tokens` 切片，经 `wkv`、`kv_norm`；仅凭 8184 与下面 8180 的形状不能证明具体裁剪结果或中间 tensor 身份。 |
| RMSNorm | ND，BF16 `[8180,512]` 与 BF16 `[512]`；输出 BF16 `[8180,512]` 和 FLOAT `[8180,1]` | 35.9195 µs | 唯一 flow 关联所列 `npu_rms_norm` API；各输出的实际地址和后续消费者未由 trace 绑定。 |
| partial RoPE | ND，BF16 `[8180,1,1,512]` 和两个 FLOAT `[8180,1,1,64]`；输出 BF16 `[8180,1,1,512]` | 36.839 µs | 对应原地旋转 API；源码路径将 norm 返回值作为 view 传入旋转，并使用位置相关 cos/sin。是否为本次相同 storage、旋转范围及位置信息未由该设备记录证明。 |
| scatter | 输入 cache 为 NCL BF16 `[17314,128,512]`、ND INT32 索引 `[8180,2]`、ND BF16 updates `[8180,512]`；输出 NCL BF16 cache 同形状 | 258.095 µs | 源码标准非 packed 路径取 metadata slot、按 cache dtype 转换并连续化 updates 后写 SWA cache；索引值、stride、是否复制及写入位置未记录绑定。逻辑 updates 为 8,376,320 字节，cache 形状对应逻辑容量 2,269,380,608 字节；均不是实测或必需 HBM 流量。 |
| Indexer | 同 scope stream 47；输入含 INT8 `[1023,32,128]`、INT8 `[17314,64,1,128]` 等 | 806.924 µs | 其第二输入与 scatter cache 的形状和精度不同，不可把两者视为同一 cache；本样本的其他来源与具体 tensor 身份未绑定。 |
| SparseFlashMla | 同 scope stream 47；输入含 BF16 `[1023,64,512]` 和 BF16 `[17314,128,1,512]` 等 | 904.442 µs | 其第二输入与 scatter cache 插入单例维后的形状相容；源码非 packed 路径从同一 `swa_cache_layer.kv_cache[0]` 读取 `ori_kv`，但这次运行的缓存 storage、相关槽与调用分支尚未绑定。 |

同流 norm→RoPE→scatter 的相邻记录间隔分别约 0.0405 和 0.040 µs，反映设备事件时间顺序，不能替代实际 tensor 依赖。所选 Q 形状相容矩阵任务与三者有交叠；scatter 结束到 Indexer、attention 开始分别为 1775.944、2803.744 µs，间隔有其他工作，不能当作等待或可省时间。源码多流分支中主流先等待 KV matmul 完成，辅助流执行 norm/RoPE 和条件缓存写入，主流在后续 `wait_stream(aux_stream)`；是否为本样本分支以及事件配对未知。`_forward_attention` 另有 metadata 就绪等待与 block table、序列长度和 sparse indices 输入，这些控制数据的实际值和与当前 cache 的槽关联仍缺失。见 `F08-EP8-PIPELINE-OVERLAP.md` 与 `../diagnosis/F08-ATTENTION-INTERVALS.md`。

融合分析可按两个边界分别考虑：norm 与原地 RoPE 接合须保持整行 RMS 归约及旋转配对和中间精度；RoPE 与 scatter 接合须保留槽有效性、缓存写入副作用和所有后续读者，并保持原有跨流就绪与可重叠部分。全链融合还要满足两类约束。这里是合法性条件，不是已经实现、历史实例已绑定或收益已量化。

## 对 RMSNorm 第二输出的源码条件审计（2026-10-30）

在已选 EP8/L14 的归一化设备记录中，导出签名除 BF16 `[8180,512]` 主输出外，还列出 FLOAT `[8180,1]` 第二输出；对应 flow 的主机侧包含 `npu::npu_rms_norm`。冻结 `ops/layernorm.py:86–105` 中，`AscendRMSNorm.forward_oot` 的 `residual is None` 分支将 `torch_npu.npu_rms_norm(...)` 的两个结果赋给 `x,residual`，可能对 `x` 原地加 bias，最终只返回 `x`，该局部 `residual` 没有被此分支后续代码使用。冻结 `attention/dsa_v41.py:327–330,487–494` 两条候选路径将 `attn.kv_norm(...)` 的返回值经 view 送去原地 rotary。**如果**历史调用实际采用这份 wrapper 的该分支、没有其他运行时 hook/隐藏消费者，第二结果在这个 Python 调用路径中不是 rotary 的直接输入；主输出则不能省，且 `bias_loaded` 时的原地加法仍须按原序保留。

因此可研究的三算子重组是：为该特定无 residual、推理路径提供仅发布归一化主结果并衔接 rotary、cache scatter 的接口，或在融合核内部保留必要的行统计而不对外发布辅助输出。输出统计的具体数学语义需匹配本版本算子契约；**不能因辅助结果未返回而省略计算归一化所必需的行规约或标量**。需要核对导出 ABI 是否要求输出二、autograd、图捕获、alias/诊断 hook、bias、所有其他消费者、动态 shape 及 cache side effect；若接口要求第二输出或其他消费者使用它，就不能省略。FP32 `[8180,1]` 的逻辑容量是 **32,720 B**，既不是实际写入 HBM 的字节数，也不是已验证可节省的流量或耗时。

此推导比仅凭输出形状推断多了 wrapper 数据依赖，但并未闭合历史实例：所选 flow 虽已匹配 `npu::npu_rms_norm`，未证明它来自上述 wrapper、该次 residual/bias 的实际值、第二输出的 storage 与所有底层消费者或编译器是否物化；也未证明后续 rotary/scatter 使用同一张量。使用 `python3 -c 'print(8180*4)'` 可复算容量；采样和调用关联来自 `f08-ep8-tensor-provenance.json` SHA-256 `7b22c498beefb88aef225d654fdf1f136b898d29d5575e0382225af8289fba2e`，已冻结源码 `layernorm.py` SHA-256 `d2fda42ad99935725c98b9505a8d2e94696d9cdb2906e5cb2f1e37bf21fb27bc`，`dsa_v41.py` SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。无新设备负载及实现修改。

## RMSNorm 跨通道依赖与有界 fan-in 的条件时间界（2026-10-30）

### 依据与独立语义

上文已记录 EP8、L14 选中 `npu_rms_norm` 调用关联的设备签名：BF16 输入 `[8180,512]`、BF16 权重 `[512]`、BF16 主输出 `[8180,512]` 和 FLOAT 辅助输出 `[8180,1]`。冻结 `evidence/model-config.json` 的 text 配置有 `rms_norm_eps=1e-20`，该文件 SHA-256 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`。**这些证据不证明历史 task 的具体 epsilon、权重值、可达输入、精确计算公式或后续 tensor 身份**。以下构造借用 512 维与配置值，不声称历史 NPU 使用了其中的具体数值行为。

定义一个独立精确接口：输入为一行 512 个 BF16 值，gamma 第 0 项取 1，其他项不影响本例所考察的输出；`ε=1e-20`。对全体输入要求第 0 个可观察输出为按理想实数计算后再 BF16 round-to-nearest-even 的

`y₀ = BF16_RNE(x₀ / sqrt((Σⱼ₌₀⁵¹¹ xⱼ²)/512 + ε))`。

限定 `x₀=1`，其余 511 个分量分别从 BF16 精确可表示的 0 或 1 中取值。令其它分量全为 0，输出为 BF16 `0x41b5`，即 22.625；对于**任意** `j>0`，只将 `xⱼ` 从 0 改为 1，输出为 BF16 `0x4180`，即 16。即在这个输入族中，每个 `xⱼ` 都能改变同一可观察输出 `y₀`，因此精确语义上它对所有 511 个动态输入均有函数依赖。

这只证明上述接口和输入族的依赖，不要求保留独立 RMSNorm kernel、额外的平方和张量或 FP32 辅助输出。融合、分块、重算、分支并行等只要保持此精确可观察结果，仍必须实现相同的输入—输出函数。**依赖不等于必须执行现有代码的一次归约，也不自动产生硬件周期或必需 HBM 读写数。** 如果只需近似结果、下游函数消除了差异、gamma 第 0 项为 0、输入被限制到不可区分的子集，或 `y₀` 不必单独可观察，这个结论不能直接用于那些更宽的契约。

### 可证明的受限电路深度

在进一步限制的独立电路模型内，假设：

1. 整个 511 输入族使用同一个有限、静态、无环的计算图；每个计算节点依赖至多 `F≥2` 个动态前驱值，常数及输入无关的模型权重可提前准备。一个拥有 511 位可变地址的巨型查表操作若被视为单个节点，**不满足**此有界动态 fan-in 限制；也不得让窗口前已知当前输入的缓存摘要免费进入初态。
2. 每个节点的动态前驱都按图中的依赖就绪后才能完成，且从任何必需前驱就绪到该节点输出就绪，存在已证明的严格下界 `δ_F>0`；图中所需路径的动态计算都在被定义的测量窗口内，不能将窗口前完成的部分当作窗内工作。该条件排除了不等待某个输入而提前完成的短路、预先计算所有分支后免费选择、将多层节点打包为未计价查表，以及其他绕过路径的情况。
3. 在此接口的整个输入族上，根节点必须输出上述可观察的 `y₀`，精度和舍入契约与第一节一致。数据相关控制或可变图若无法映射到满足以上依赖、fan-in 与延迟条件的统一图，不能套用此证书。

若根的最长动态依赖路径长度为 d，则反向展开每层最多 F 个前驱，最多覆盖 `F^d` 个动态输入；第一节已证明必须覆盖至少 511 个不同输入，故 `d≥ceil(log_F 511)`。由逐节点依赖和最短延迟，得到

`T ≥ ceil(log_F 511) × δ_F`。

例如假设 F=2，d 至少 9；假设 F=16，d 至少 3。对同一窗口，若另有独立证明的计算、通信或访存容量下界，与本项取最大值即可允许其他资源、行和分块并行；不应把整个已有 norm、RoPE 和 scatter 的墙钟时长直接相加。动态 fan-in 并非 PyTorch 操作或一次向量指令的个数；底层指令、宽向量、内存索引及硬件同步的可用并行度与实际延迟都要另证。若一部分依赖在窗口之前已完成、路径节点能以更大 fan-in 直接操作、或共享计算图允许提前枚举并通过不同机制选择，则必须调整模型及其时间账，不能沿用该数。

### 实际适用性与复算

本机资料尚无可绑定上述 task 的 gamma 值、运行时 epsilon、精度契约、后续可观察输出、合法激活可达集、所有输入相关缓存初态、硬件有效动态 fan-in 或 `δ_F`。它们任一项缺失时，该式只是独立接口的条件证书，**不是历史 35.9195 µs 的下界、利用率或真实模型 TPS 上界**。辅助输出是否对外发布也不决定这个输入—主输出依赖是否存在，但主输出若被下游等价融合消去，必须重新证明最终可观察结果仍区分这两类输入。

纯 CPU 数学复算（取理想实数后经 FP32 表示再模拟 BF16 RNE；不是验证历史 NPU 精度）：

```sh
python3 - <<'PY'
import math, struct
for k in (1, 2):
    y = 1 / math.sqrt(k / 512 + 1e-20)
    bits = struct.unpack('>I', struct.pack('>f', y))[0]
    bf = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    rounded = struct.unpack('>f', struct.pack('>I', bf << 16))[0]
    print(k, hex(bf), rounded)
PY
```

输出依次为 `1 0x41b5 22.625`、`2 0x4180 16.0`。这证明给定算术模型的两个结果可区分；不证明模型在真实目标设备逐位复现或这两种输入实际出现。本次无远端访问、设备工作或目标实现修改。
