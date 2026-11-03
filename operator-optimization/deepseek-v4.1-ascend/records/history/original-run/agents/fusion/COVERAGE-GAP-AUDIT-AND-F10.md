# 覆盖缺口审计与 F10 深查（可恢复记录）

## 范围与证据等级

基于 `evidence/FUSION-SWEEP.md` 的 F01–F31 与 `agents/fusion/LAYER-COVERAGE.md`，核对已冻结 V4.1 target、DSpark、Engram、CP 和量化/MoE 代码及短时历史 trace。**覆盖**仅表示已做相应源码/采样审查；它不代表整个计算图穷尽、已证明能融合或已完成任何性能优化。采样覆盖为历史窗口，不能推广到全部 prefill、decode、并发、rank 和请求；下表未覆盖是按当前工作材料标记，不是声明模型一定执行。

| 主要阶段 | 已覆盖的重点及证据 | 未充分覆盖或无法判断 |
|---|---|---|
| 输入、mHC 与 norm | F01–F04；F02 已进一步查 attention pre 的 y 与系数支路，F03 特别查了 post/pre、pre_mix、source tiling；复合 HC 已存在。 | 运行 binary 与真实 tile、全部形状和数值规则；norm 与相邻环节的实际中间物化/融合能力。 |
| attention Q/KV 与输出 | F05–F08、F13–F16；F06 进一步查 CP 局部/全局行域与 metadata 身份，F07/F08 查了源码与部分同窗流，F15 查 head exchange 到 projection，F16 深查两级投影/规约的 tile 与舍入。 | KV/indexer 的真实 tensor 对应与事件；QLI/SMLA、projection 内部完整 tile 图；通信真实逐 tile ready、性能重叠及资源竞争。 |
| Indexer 及共享索引 | F09、F12 已查量化 helper、跨层索引及 EP8 层 24 时序；F10、F11 已分别深查权重投影→缩放转换与 key/scale 缓存的源码候选。 | 相关 tensor、事件和实际运行分支尚未绑定；内部 QLI 数据流、逐行地址及多消费者 alias 仍 UNKNOWN。 |
| Compressor 与共享 KV | F17–F19，已查双 FP32 投影到通道池化的依赖、pooled norm、ring 状态反例、两分支消费者。 | 真正 tile 实现、每请求状态顺序/全量运行绑定；ratio 1 与 ratio 2 的所有形状及 cache 页冲突。 |
| router、routed/shared MoE | F20–F23 已查 router 和 MC2 的外部调用及数学依赖，已有复合 MC2 和 muls_add。 | MC2 内核完整 tile/通信 DAG、各阶段真实资源及跨 rank 收敛；各流共享专家 join 的具体调用绑定。 |
| Engram CPU offload 与 gate | F24–F25 查过 host UVA gather 和 FP32/BF16 投影/gate 分支。 | 真正运行权重 tensor 身份、lookup/host/NPU 带宽及完成事件，所有设备内核/物化。 |
| 最终输出、DSpark 与 metadata | F26–F31 已查尾层多消费者、DSpark 采样与 metadata；F26–F29 进一步检查 gather/collapse、辅助状态投影、首层 HC、各层 context KV，F30 检查 vocab 分块与验证。 | F30 须覆盖全部词项并保留 dtype 舍入、tie/NaN、词表映射、概率分布与 RNG；F26–F29 的运行 tensor、事件、分支及请求绑定和收益未知，metadata 缓冲生命周期亦未闭合。 |
| 框架调度、跨请求/阶段 | 部分源级 event 与 trace 的同流先后已核对；有其他模块专门审计数据依赖。 | 请求身份、全部 stream/event 对、跨轮并发、host 调度与图重放；不能从 device 任务时长简单相加得到端到端关键路径。 |

**不能称“全量已扫完”。** 现有 F01–F31 是边界清单与部分深入检查，不是所有可组合二/三/多节点的证明，也不能代表目前每个工作负载都命中。进一步探索按新增具体证据和不同边界推进，停止重复罗列。

## 本次选 F10：权重投影 → 缩放 → 类型转换 → QLI

### 冻结源码中的具体依赖

`models/deepseek_v41/indexer.py:57–64,171–172`：`weights_proj` 是无偏置 `ReplicatedLinear(hidden_size, index_n_heads)`、无 quant_config；结果通过 `_output` 取 tensor 后 `.float()` 再乘 `weights_scale = width**-0.5 * n_heads**-0.5`。冻结配置 `width=128`、`n_heads=32`，按实数算 scale 为 `1/64`；Python 数值和实际 kernel 浮点计算仍按原语义处理。不要因它是 2 的幂就忽略 BF16 投影输出舍入、FP32 中间极值/特殊值及 FP16 转换。

`attention/dsa_v41.py:393–409`：只有 `has_long_context && is_index_source` 才执行。主流先计算 index query projection，并记录 `weights_start`；辅助流等此 event 后执行上述 projection/FP32 cast/scale，记录 `weights_done`；主流同步做 query RoPE，之后等待 `weights_done` 才返回 `prepared.query/weights`。这是一处已存在的异步重叠；不应把 projection 与 RoPE 算作完整串行，也不能仅凭两个任务的时间重叠证明竞争程度或实际事件绑定。

`indexer.py:193–254,274–285`：空 query 和空 cache 分支在权重 `.to(torch.float16)` 前早返回；非 packed 的正常路径才将 weights 转 FP16，作为 QLI 第三个位置参数。packed 分支接收原先 FP32 weights，走另一后端。`dsa_v41.py:580–602` 在选择前还需等待 query quantize 完成，`indexer.py:274` 另有 metadata wait。源码中 weights 明确直接数值消费者为 select_projected 中的对应 backend；同一 `hidden_states` 还有主 attention 等独立消费者，不能因融合 weights projection 就删除其余读者。最终结果通过 QLI 决定共享索引，影响后续层 SMLA，不能把权重缩放视为对模型输出无影响。

### 已有采样与条件候选

已有 `f07-q-kv-streams.json` 记录同一 EP8 层 24 run 2，窗口起点后 `169.377–180.437 µs` 的 stream 87 BF16 `MatMulV2`：输入 `[3,5120];[32,5120]`，输出 `[3,32]`；同期 stream 89 的 `InplacePartialRotaryMul` `[3,1,32,128]` 为 `168.837–172.637 µs`。该形状、不同流及重叠与源码权重投影、query RoPE 的预期相容。`f12-layer24-run2-evidence.json` 中后续 QLI 输入包含 `[3,32] FLOAT16`。这些只是同层窗口中的强线索，**没有 tensor 地址和事件对象绑定，不能宣称已证实 MatMul 输出经转换进入 QLI**。

可研究让 projection epilogue 对每个输出先按原路径保留 BF16 投影输出的舍入、再以 FP32 乘 `1/64`、最终进行原 FP16 舍入，直接生产 QLI 可消费的权重；不允许把 BF16 乘法或对未舍入 GEMM accumulator 的 FP32 乘法直接视为等价。此候选若将 FP16 转换从主流前移到辅助流，还须正确调整 `weights_done`、量化工作与所有局部/packed/空缓存分支，并重新核对 query 和 metadata 的事件；原先重叠关系可能变化，流上没有依赖也可能争抢 Cube/Vector/MTE。输入为 `[3,32]` 时数据很小；设备 launch、图捕获和同步可能占主导，但现有材料没有关键路径和收益测量。

`float()`、乘法、`.to()` 独立 Python 表达式并不证明会产生三次独立设备 task 或真实 HBM 访问；不能承诺节省任意字节/微秒。源码中没有直接看到上述组合型 epilogue，图 lowering 和底层 fusion UNKNOWN。所需输入 `_output(linear, value)` 可能 tuple 解包；真正量化方法、投影结果 dtype 和 callback/其他消费者都需按实例确认。

### 可恢复下一步

stage150 由 ZCode `deepseek-flash` 执行有界只读查询；独立核验 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-150-f10-layer24.exit-code` 为 0、`.verified.json` 的 `model_verified=true`。输出 `f10-layer24-run2-evidence.json` SHA256 `1fbf29d27126ad4286917963b0cf9c3c4c4d2bbe13716a937b9df0eb1ee1886d`；查询脚本 SHA256 `9e24f2da86d4032f8ed22a42ce7579ca37fece22adbff918332244a6b6513c87`，CSV SHA256 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 与冻结清单匹配。未启动 NPU。

以下均是同一 EP8 层 24 run 2 相对起点的观测，时间单位 µs：

| 事件 | stream / task | 开始–结束 | 记录的物理输入 → 输出 |
|---|---|---:|---|
| indexer query 形状相容的 partial RoPE | 89 / 1270 | 168.837–172.637 | `[3,1,32,128]` BF16 |
| weights projection 形状相容的 MatMulV2 | 87 / 283 | 169.377–180.437 | `[3,5120]×[32,5120]` BF16 → `[3,32]` BF16 |
| Cast | 87 / 284 | 180.457–181.957 | `[3,32]` BF16 → FP32 |
| Mul | 87 / 285 | 181.997–183.557 | `[3,32]` FP32 和标量 → `[3,32]` FP32 |
| `_quantize_indexer_query_kernel` | 87 / 288 | 184.697–189.937 | shape/dtype N/A，model ID `4294967295` |
| Cast | 89 / 1274 | 190.537–192.117 | `[3,32]` FP32 → FP16 |
| QLI | 89 / 1275 | 193.216–691.707 | 输入中有 `[3,32]` FP16 |

这观察到与源码预期形状、dtype 和时间顺序相容的完整链，以及 MatMul 与 RoPE 时间部分重叠。它是同一窗口中分别存在对应 task 的证据，**不是**这些 task 的输入输出 storage 一致、乘数实测为 `1/64`、源码具体调用已绑定或 stream event 因果关系已证。模型复用及 graph replay 可能导致 task ID 重复，不可据此匹配跨窗口；查询只覆盖指定算子名。三次 cast/mul 小任务的时长也不能相加作为融合收益，因为现有重叠、依赖与资源争用尚未解开。下一步只有当已有证据可绑定 tensor 或 event 才能继续收紧；否则保持 UNKNOWN，不新做无目的测试。


补充：F30 深查见 `F30-VOCAB-TILE-AND-VERIFY-REVIEW.md`；这是条件粒度边界审计，不代表 F26–F31 或全模型组合已穷尽，未启动新 trace/NPU。

F06 的 CP 行域条件推导及历史采样限定见 `F06-CP-SHARED-INPUT-ROW-DOMAINS.md`；真实实例、量化配置和张量绑定未证，不代表全图融合覆盖完成。

F17 的单通道投影→池化依赖、已采窗口和共享权重平均反例见 `F17-COMPRESSOR-DUAL-PROJECTION-PIPELINE.md`；tensor 及事件绑定未知，仍不是穷尽全图融合。

F02 attention 输入混合/系数拆分和两个后继 join 条件见 `F02-ATTENTION-HC-BRANCH-EARLY-READY.md`；与 F03 共用 HC 内部分析，不可重复计入收益，完整覆盖仍未知。

F16 的两级投影条件分块、预乘 BF16 反例及 stage170 同窗 trace 限制见 `F16-TWO-PROJECTION-TILE-AND-ROUNDING.md`；数据身份、event 和收益仍 UNKNOWN。

F26 的 SP+MTP 条件下 gather/collapse 等价约束、MTP residual 消费、条件通信字节和 stage171 尾标记后有限形状证据见 `F26-TAIL-GATHER-COLLAPSE-REVIEW.md`；目标身份与张量绑定未证，未覆盖所有尾部路径。

F27 target 辅助层状态的 HC mean→草稿 main projection→norm 的条件重组和归约/生命周期限制见 `F27-AUX-MAIN-PROJ-PIPELINE-REVIEW.md`；现仅源码级分析，真实输入组装、运行 tensor 与性能均未证，不能称所有草稿路径或全图覆盖完毕。

F27 目标辅助状态到 DSpark 投影的条件分块与 stage172 形状相容记录见 `F27-AUX-MAIN-PROJ-PIPELINE-REVIEW.md`；张量身份、完整数值等价和收益仍未知。

F28 首层 DSpark 四份 embedding 和固定外部 premix 的专属结构、可并行/重组条件、浮点反例及 SP 约束见 `F28-FIRST-DRAFT-HC-STRUCTURE.md`；仍未绑定实际执行或收益，也不能将其与 F02 共用部分重复计收益。

F29 三层 context KV 投影到各层持久 SWA cache 的可打包/分块/重叠条件及范数、RoPE、group slot 和 cache 地址限制见 `F29-DSPARK-CONTEXT-KV-PIPELINE.md`；未绑定具体运行调用，不能据源码重组候选宣称性能提升。

F14 的 attention 最终归约→inverse RoPE、按 token/head 与 CP 交换可交换的精确条件、位置相关权重和 BF16 中间舍入反例见 `F14-ATTENTION-ROTATION-PROJECTION-REVIEW.md`；与 F15/F16 重叠部分不得重复计收益，真实 tensor 与 tile ready 仍未绑定。

F19 进一步划分 index `wk` 最后读取 latent 与 key/scale 发布、long cache 的不同首消时点，及 source 2/8/14/20 的跨层角色范围，见 `F19-LATENT-FORK-PUBLICATION-CUT.md`；实际 tensor/event 绑定与跨轮缓存版本仍未知。

F13 的 SWA/压缩注意力单 softmax 合流、理论可并行条件、必须保留 sink 与 buffer 状态及接口版本限制见 `F13-SWA-COMPRESSED-SOFTMAX-JOIN.md`；重组不代表当前 SMLA 已提供分支状态，也不能由既有 QLI 时间推测隐藏收益。

跨 target 采样和 DSpark 的 padded 行集合独立性、仅能前移 hidden 投影而不能猜 accepted/metadata 的条件，见 `RUNNER-PADDED-SPEC-DRAFT-PROJECTION-EARLY.md`；仅源码证据，实际 runtime/事件及收益未知。

草稿 pooled 最低 AIC MAC 代理的限定含义、legacy CP 空 query 仍需参与 head 交换的双 rank 反例及变长重组条件，见 `DRAFT-EMPTY-QUERY-CP-EXCHANGE-REVIEW.md`；历史是否为空 query 及收益未证。

### F10 缩放精度：可证明的有限值条件域

此前“不能直接视为等价”的提醒仍适用于未绑定的真实实现，但可以把一种候选的数学条件显著缩窄。冻结配置 width=128、n_heads=32 时，Python 表达式实际值 `128**-0.5 * 32**-0.5` 为 `0.015625000000000003`，并非 float64 严格等于 1/64；在 IEEE 最近舍入转换为 FP32 标量后则为 `0.015625`，即 2^-6。历史算子中标量如何编码、提升和参与计算尚未绑定，不能只凭源码证明实际采用 FP32 标量。

令 x 为已经完成原投影舍入的任意**有限 BF16 值**，并假定缩放标量及算术为 IEEE 风格、有效标量恰好是 2^-6、BF16 和 FP16 的转换使用一致的最近偶数舍入，且没有改变相关结果的不兼容 flush-to-zero；比较 `FP16(FP32(x) * 2^-6)` 和 `FP16(BF16(FP32(x) * 2^-6))`。对乘积位于 BF16 正常数范围者，乘 2 的整数次幂仅改变指数，无有效位损失；对乘积绝对值小于 BF16 最小正常数 2^-126 者，即使 BF16 转换发生下溢，其结果远低于 FP16 最小正次正规数的一半 2^-25，两条路径最终都舍入为同号 FP16 零。有限大数转 FP16 的溢出及 signed zero 也按相同转换处理。因此在上述数值模型下，两条路径对全部有限 BF16 值的 FP16 位模式相同。

只读 CPU 模型可复算：`python3 agents/fusion/f10_bf16_power2_scale_exhaustive.py`，脚本 SHA-256 `09e9b6c2f8c13b86c26bb038b9d2b382a4a7512c6547a56d8120af6dd544c2bf`。使用 NumPy 2.0.2，枚举 65280 个有限 BF16 bit pattern，FP16 位模式差异为 0；不是实际 PyTorch、CANN、Ascend NPU 运算测试。NaN payload/信号 NaN、Inf、非 IEEE 模式和实际硬件标量 lowering 不在结论内，且 packed 分支消费的是 FP32 权重，不可省略中间精度。将缩放提前到 GEMM accumulator、改变投影舍入、变更 1/64 或把投影权重预缩放都不在此结论范围。

对 F10 的分析影响是：在能证明这些条件及 QLI 只观察最终 FP16 的非 packed 路径下，“先经 BF16 缩放必定不同”不是拒绝该候选的充分理由；原文要求先保留 BF16 投影输出舍入与验证事件/所有消费者仍成立。这不证明历史运行满足条件、真实 NPU 等价、融合接口可用或性能收益。

关于标量及 dtype 的证据：EP8 层 24 run 2 的 task 285 导出 `Mul` 的输入类型 `FLOAT;FLOAT`、输出类型 `FLOAT`，task 284 是 BF16→FLOAT，task 1274 是 FLOAT→FLOAT16；它们与 Python 权重路径相容，但没有张量/调用绑定或标量数值，不能证明这就是该路径或当时乘数。[PyTorch 当前类型提升文档](https://docs.pytorch.org/docs/stable/tensor_attributes) 描述普通 Python scalar 和 tensor 算术的类别提升规则；[华为 CANN 9.1 Tensor/Scalar 推导文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/common/tensorscalar_deduction_relationship.md) 表明 `aclnnMuls` 会按 tensor/scalar 类型推导并转换，但这两个不同版本的通用接口文档均不是历史 torch-npu/CANN 编译 lowering 的证据，不能以输出 FLOAT 推断实际乘数编码及计算精度。应检查同版本 `torch_npu` scalar 构造与 `aclnnMuls` host tiling/kernel，以及当时 graph 常量；当前未找到绑定证据。

独立边界断言补查了正负 `x=2^-19` 经 2^-6 后恰为 FP16 最小 subnormal 的半值，最近偶数得到正负零；输入 BF16 正负下一表示值得到正负最小 FP16 subnormal。脚本逐位断言这四种预期值并通过，仍只验证 NumPy CPU 模型，不替代目标设备的 FTZ、舍入或 NaN 行为验证。

### F10/F09 会合处：不能直接把权重缩放搬到 query scale

此节是已登记 F10 权重路径与 F09 query 量化路径在 QLI 合流时的**舍入边界**，不是另一项可独立累计的融合收益。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py` SHA-256 `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33` 的 `:171–172,254–256,275–280` 在非 packed 路径把 weights projection 输出转 float、乘配置尺度，再转 FP16，与量化 query 和独立 query scale 传给 QLI。该配置的尺度在恰当 FP32 转换条件下为 `2^-6`；源表达式与实际设备标量的精确关系见前节。F09 已存 `agents/fusion/frozen-source/vllm_ascend/ops/triton/quantize_indexer_query.py` SHA-256 `6065b79f0f764e99bafae71347c8c8dfd0625ad52b0d695639c47e71338d0bba`：逐 token/head 计算 scale，按所存 FP16 roundtrip 和最低 scale 规则生成 INT8 query 及 FP16 scale。保存的 arch22 `csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h:730–747` SHA-256 `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8` 在 half 局部 tensor 上相乘 weights 与 query scale，再 broadcast 到后续计算。

保存的 EP8/L24/run2 的 QLI task1275 输入签名含 INT8 `[3,32,128]`、两个 FP16 `[3,32]`，与此路径相容，详见 `agents/fusion/f12-layer24-run2-evidence.json`；shape 不能绑定这两组输入与上述生产者或证明历史调用使用保存的 arch22 二进制。packed 分支消费不同格式的 weights，不自动适用下述分析。

仅在有限实数模型下，可把乘数从 `(w * 2^-6) * qscale` 改写为 `w * (qscale * 2^-6)`。然而这些参数分别跨越 FP16 接口时，改变舍入位置可能改变结果。下面是**CPU NumPy、FP16 最近偶数舍入**的构造性反例：`w=2^-19` 可用 BF16 精确表示，`qscale=64` 可用 FP16 精确表示；先缩放 weights 时 `2^-25` 是 FP16 零和最小正次正规数的中点，按最近偶数得到零，再乘 64 仍为零。先将原权重转半精度、把尺度改放到 query scale，再相乘得到 `2^-19`。

```python
import numpy as np
w = np.float32(2**-19)
c = np.float32(2**-6)
q = np.float16(64)
original = np.float16(np.float16(w*c) * q)
moved = np.float16(np.float16(w) * np.float16(np.float32(q)*c))
print(float(original), float(moved))
# 0.0 1.9073486328125e-06
```

这里两种结果都是有限值；F09 所存源码的 max/127 量化在数学模型下对单 head 最大值 8128 可得到 scale 64，但这不证明指定 BF16 数据、权重或 scale 在历史运行可达，也不证明实际设备的 FP16 转换、FTZ、Mul 舍入及 QLI 分数和 top-k 会产生相同结果。即使尝试搬移 scale，也必须让 INT8 query 的生成继续使用原本量化 scale；在量化之前更改 scale 会改变 query 整数值。各 head 的 weight 和 scale、两分支其他消费者、量化和 metadata ready 事件、buffer 寿命及完整 QLI 契约均须保持，或另证允许的数值误差。

这不与上一节证明冲突：上一节保持原 BF16 weights 投影舍入，比较的是**权重自身缩放链**是否经过额外 BF16 中间舍入；本节改变的是两个独立 FP16 输入及其相乘的位置。有限范围、特殊值、真实后端指令/编译与设备绑定均未知，不能据此认定线上 bug、方案必然不合格或有可兑现收益。仅使用已有文件和 CPU 算术，没有 NPU 工作或推理实现修改。

## G1 aux 层条件错位与 G5 context-KV 边缺口（离线有界核对）

本节检查的是图与已存源码/配置的绑定，**不是新增运行结论，也不说明所有融合已扫描完**。G1 原图 SHA-256 `9bece982c89be6320c4f3ef986ec33030a2e71327f0f0a96ae1159a19bb187cc`；既存 overlay SHA-256 `ac4b3138e8461443c5dda153553b9013c255d03cccc56bbd3b773115a4749e5f`；materialized G1 SHA-256 `b6ddd6dcc580fa4133dfa1ff937f531e20611c66c6b8567dfffaec9c952e4c44`；G5 模板 SHA-256 `a3a4e82afb8fed8894300148791a22ed1ca954288a0ed8fd79ef9a3879c56731`。

### G1：现有图的 aux 节点未反映已调查的 DSpark fallback

图中现有 `target.layers.{36,37,38}.aux`，对应的上游 `hc_ffn_post` 分别是第 35、36、37 层，三节点接到 `dspark.main`。图把这些边标为根据 one-based config ID，但已有 F27 及本轮复核表明：冻结 `evidence/model-config.json` 的 `text_config.dspark_target_layer_ids=[37,38,39]`；`agents/bounds/metadata-path-snapshot/ascend_runner.py:787–802` **先**取父类可能返回的列表，只有父类未返回非空列表、method 是 DSpark 且 draft 实例 text config 的字段也是该三项时，才执行逐项 `+1` 得 `[38,39,40]`；`:4427–4444` 还要求满足 aux 配置条件才传给 setter。已存 `f27-aux-setter-evidence.json` 显示该 setter 按 `sorted(set(layers))` 保存互异值。冻结 `models/deepseek_v41/model.py:1555–1564` 对每层检查 `layer.layer_idx+1`，在进入该层及其可能的 Engram 之前从当时 hidden 对 HC 路求 mean。

所以，**在上述特定 fallback、三个层都在该 model forward 中执行等条件成立时**，该分支应表示零基层 **37、38、39** 的入口 mean：在本图的残差版本粒度下分别来自第 **36、37、38** 层 `hc_ffn_post`。旧图有 `36.aux` 而缺 `39.aux`；旧图关于 `37.aux` 和 `38.aux` 的现有相应上游边仍与此条件相容。重连时还须考虑潜在 Engram、SP gather、PP、其他 hidden/aux 消费者和其他 aux 配置，不能把一个数值判断推广到所有模型分支。父类可优先返回另一套配置、draft 的真实 runtime config 未绑定、setter 和执行版本/实际对象未绑定，因此不能声称历史请求一定命中 37/38/39，也不能把 fallback 直接作为唯一可能路径。F27 源码结论此前已记录；**新的发现是 materialized G1 与它未对齐，进而可能使以图索引的融合候选遗漏 39.aux 边并包含条件不符的 36.aux 边**。现有候选是结构集合，不自动证明可融合或可节省时间。

### G5：draft context 位置与按 group slot 的生产者未在片段中闭合

G5 的 `DSPARK_BOUNDARIES` 当前有 `input_expand → slot_outputs`、一个没有入边的外部节点 `context_slots → kv_precompute`，以及 `query_positions → block0/1/2`；**没有表示 context positions 的节点或 `input_expand → context_positions → kv_precompute` 数据边，也没有把 `slot_outputs` 中 context-slot 子域与 `context_slots` 接起来**。`DSPARK_SHARED_KV` 中 `norm → rope`，也没有表示 context position → rope 的数据输入。`context_slots` 被明确标为外部，是片段级抽象，不代表内核没有收到该数据；问题是这些片段无法独立表达其来源和完整消费者，不能拿它们证明完整依赖或安全前移 RoPE。

源码链：冻结 `spec_decode/dspark_proposer.py:258–310` 先设置 context 数，再对每个 draft attention group 调 `copy_and_expand_dflash_and_dspark_inputs_kernel`，分别写 `_context_positions_buffer` 和该 group 的 `_per_group_context_slot_mapping_buffers[gid]`，随后按 `_layer_group_idx` 将 group buffers 映射为各层列表。冻结 `ops/triton/spec_decode/utils.py:111–122` 对 `[0,total_input_tokens)` 从 `target_positions` 和 group 的 context slots 分别复制值；query positions 则由另一段 `:125–161` 构造，不能混同。`spec_decode/llm_base_proposer.py:1480–1481` 把 context-slot 列表交给 `dflash_proposer.py:301–319`，后者按 `_dflash_num_context` 切片，将 context states、**context positions**、slots 一起交 `precompute_and_store_context_kv`。`models/deepseek_v41/dspark.py:55–68,284–309` 把 context positions 用于各层 RoPE 的 cos/sin 查找，slot 则用于各层 KV 写；无 context 或 slots 为 None 等路径有早返回。`dflash_proposer.py:56–71` 初始化的 context slots 与 positions buffer 为 INT32；不同 group 的 slots 与共同 positions 仍是不同数据，内部 producer、流事件及是否可重组均须逐分支证明。现有 F29/DSPARK-PER-GROUP 早已分析该源码链；**新的缺口是图结构没有接上这条具体边**。

完整 source SHA-256：`dspark_proposer.py` `70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c`；`dflash_proposer.py` `f3b736f44a59ec0831aa7f2849e2eae6d709951e955992b8016f49e772fce3ff`；冻结 `utils.py` `f3c121111d07f08d0de0498faa64529700dd6089ff7e46303eecf4947e516a95`；`dspark.py` `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601`；`model.py` `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；模型配置 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`；Ascend runner 快照 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`；父 runner `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51`；setter 摘录 `3dd283959f281dfc3c4c25f43f0546ab610f9950d12d9c90792bff0b29af71f4`。

### 安全版本化要求与复核入口

本轮**没有改原图、overlay、已有候选索引或推理实现**。若之后修图，应以旧图及 overlay SHA 为输入建立新的条件图版本，显式区分 fallback 与父类/配置覆盖路径；以新版本 G5 表示 context positions 与 group slot 的完整条件生产/消费映射，并对映射的缓存生命周期、数据域及事件另留 UNKNOWN。不得把更正后的顺序位置套用到原 `cross-module-G1` 或 `cross-module-G1-conditional-v1` 的旧 edge index。需要按新图重建或显式逐边映射、重新算图 SHA、重新生成受影响候选及外部输入/输出索引；逐条检查端点存在、原边精确对应、删除边标空、条件边与 shape metadata 分类正确、旧 `NEXT_INVOCATION_RAW` 排除规则和 module 规则一致，并重新检查分支互斥和实际语义。`tools/materialize_conditional_graph_index.py`、`tools/verify_conditional_graph_index.py` 当前硬编码旧 v1 输入/输出及单一删边假设，只可作为审查方法参考，**不能直接作为该新版的现成修复工具**。G5 的记录和候选若引用其旧哈希，也须单独新版本绑定并对受影响结构重扫；不能只调高组合数当成新融合或已证性能。

下列纯本地 CPU 命令复核现图节点/边与配置算术（不模拟模型执行）：

```bash
python3 - <<'PY'
import json
x = json.load(open('graphs/G1-conditional-materialized-v1.json'))
c = json.load(open('evidence/model-config.json'))['text_config']['dspark_target_layer_ids']
aux = sorted(n['id'] for n in x['nodes'] if n['id'].endswith('.aux'))
print('config draft IDs:', c, 'fallback installed:', [i+1 for i in c],
      'fallback zero-based target layers:', [(i+1)-1 for i in c])
print('G1 aux nodes:', aux)
for e in x['edges']:
    if '.aux' in e['producer'] or '.aux' in e['consumer']:
        print(e['producer'], '->', e['consumer'], e['dependency_kind'])
g = json.load(open('graphs/source-templates-G5.json'))
for t in g['templates']:
    if t['id'] in ('DSPARK_BOUNDARIES','DSPARK_SHARED_KV'):
        print(t['id'], 'nodes:', [n['id'] for n in t['nodes'] if 'context' in n['id'] or n['id'] in ('input_expand','slot_outputs','rope','slots')])
        print('relevant edges:', [(e['producer'],e['consumer']) for e in t['edges'] if e['producer'] in ('input_expand','slot_outputs','context_slots','slots') or e['consumer'] in ('context_slots','kv_precompute','rope')])
PY
```

命令执行目录为本 run 根目录；必须结合上述源码行阅读，脚本本身不会校验运行条件或证明图边能融合。

### 条件修订包与独立引用校验（本地离线）

在不改旧图或既存 records/索引的前提下，新建 `graph-revision-aux-context-v2/`：`G1.json` 是仅覆盖上节已明确的 DSpark fallback 条件的实例图变体，不是所有配置的替代；`G5.json` 在既存模板中补充 context positions 与按组 slot 的数据边，碎片仍未合成全模型执行图。该目录的 `manifest.json` 固定记录全部输入/输出 SHA、跨模板连接所需条件与未证明事项。四个 `F27/F28/F29/F30-boundary.json` 从新版 G5 原始模板节点重取内部边与外部消费者，并指回旧候选 SHA；原 `records.json` 保持原引用，若以后要采用此版本须显式选择，不可假装旧记录已自动升级。其他未受改动的片段和原 G5 历史候选继续保持原版本引用，不声称新图对全模型消费者闭合。

`G1-edge-map.json` 逐条映射旧 conditional-v1 边；两条与旧第 36 层 aux 相关的旧边标为 null，所有其他边在新位置字典值逐项完全相同。新图有第 39 层 aux 两条条件边。重建的 `G1-index.json` 对全部 2–4 节点跨模块结构重新枚举、重新绑定内部和外部 edge 索引，保持旧版 `NEXT_INVOCATION_RAW` 和 shape metadata 不参与连通但 shape 仍在引用中的规则。旧、新各 22,820 个结构集合，旧图 46 集合不再出现在该 fallback 图，新图有 46 个不同集合；这些只是结构变动数量，不是 46 个可融合机会或收益。新版 G5 的 `DSPARK_BOUNDARIES` 和 `DSPARK_SHARED_KV` 各自结构索引全量重算，分别 467 和 12 项，含控制/元数据节点，不能按数量推断可融合算子。

构建工具 `tools/build_aux_context_graph_revision.py` 先验证全部硬编码源文件哈希，对原 v1 的 22,820 项逐个检查 edge 引用、重演旧 G5 的 438 项结构，然后生成新版、旧边映射和四份边界，写前检查目标冲突、拒绝覆写不同内容；二次运行得到同一 manifest SHA。独立工具 `tools/verify_aux_context_graph_revision.py` 不调用构建脚本代码：验证 manifest 输入/输出哈希、所有旧边映射、节点端点、fallback 三组 aux 边、全部新 G1 索引与新 G5 两个索引以及四份边界；包含对旧版 F29/G2 内外部边保留和新增位置外部边的核验。此次都仅本机 CPU 离线执行，结果 PASS，没有操作设备或修改推理实现。两个工具及 manifest 的哈希由本轮执行另行记录；任何旧源文件发生变化应停止并重新生成新版本，而非强行复用。

复现（从本 run 根目录）：

```bash
python3 tools/build_aux_context_graph_revision.py
python3 tools/verify_aux_context_graph_revision.py
```

以上证明本包在指定源码条件下的**索引和引用一致性**，不证明旧实现存在 bug、历史运行采用该分支、候选数值等价、全部其他消费者被发现、逐块事件或性能收益；不同 aux 覆盖配置和其他 cache 分支需要另建带条件的图版本。

### DSpark context 跨模板条件关联和有界结构索引

在保留上述 v2 图与索引的基础上，新增独立目录 `graph-revision-cross-template-v1/`，将 `DSPARK_BOUNDARIES`（前缀 `B::`）和 `DSPARK_SHARED_KV`（前缀 `K::`）的**原有 95 条边逐条原样保留、命名空间化**，另加入三条跨模板的条件值/视图对应：

1. `B::context_buffer → K::context`：同一次 first-pass 的 context hidden 切片经 projection 后进入各层 context KV 投影。
2. `B::context_positions → K::context_positions`：同次 input expand 生成、按 context 数切片的 positions，供该层 KV RoPE；它与 query positions 是不同域。
3. `B::context_slots → K::slots`：同一 draft 层根据 `_layer_group_idx` 选中该 group 的 context slot buffer，再按 context 数切片交给本层 cache 写的 slot 处理；不可推成所有 group 或三层使用同一 slots。

具体源码链与 SHA 在上文已有：`llm_base_proposer.py:1063–1075,1480–1481`、`dspark_proposer.py:257–310`、`dflash_proposer.py:301–319`、`dspark.py:55–68,284–309`。关联均限定相同调用与缓冲版本、对应层及 group、非空 context、slot 列表有效以及相应路径执行；源码中的 slice、算子 enqueue 与 graph 阶段还不能证明真实设备中同一 storage、零拷贝或跨流 event 配对，不能从静态边推导物理同步已经完成。空 context/slots、K=0、packed cache 及 layer group 不同等分支不可忽略。三条是**条件数据映射而不是三个新计算算子**。

新 `graph.json` 保留 `B::kv_precompute` 粗粒度复合节点作为层次与外部引用，但结构组合的连通枚举将它排除，避免它与代表其内部的 `K::` 细分节点一起计作两个独立操作。`index.json` 穷举此限定图中 2–4 节点、内部至少包含一条跨模板映射的结构集合：大小 2、3、4 分别为 3、6、16，共 25；所有条目标为 `STRUCTURAL_ONLY_NOT_A_FUSION`，还包含纯元数据与 view 节点，**不是 25 个可行融合**，也不能据此断言全模型已扫描。`views.json` 按原 F27/F28/F29/F30 选定节点重算本连接图的全部内外部引用：F29 的 slots 边界多了直接来源 `B::context_slots`；context states 与 positions 在 `lineages` 中显示直至 `K::proj` 和 `K::rope` 的来源。其他 F27/F28/F30 既定选择没有直接跨模板外部边，不代表完整运行没有其他消费者。

`manifest.json` 固定旧 v2 manifest、G5 和四份候选视图哈希及本包输出哈希，绝不覆盖旧文件或原 records。构建脚本和独立验证脚本如下，均只读本机现存资料、不操作设备；独立校验遍历 95 条原边、三个新映射、所有 25 个结构集合及四份候选视图，已 PASS。构建脚本二次执行得到相同 manifest 内容：

```bash
python3 tools/build_dspark_context_cross_template.py
python3 tools/verify_dspark_context_cross_template.py
```

构建脚本 SHA-256 `719525ef9355cd5028377cc4e7a4700a4d3ac3cc56535d04682c4cc647534d61`；独立验证脚本 `460d5d99d3eb377e4576ea790e8fbdbd025cd22642efe947ae66806480ba9d06`；新 manifest `0a540f5abf0752b70f9971645a679ba2d941738dfae3495a73728931910d084a`。这些只证明所建片段与记录的结构索引一致，不证明条件在历史运行成立、全部消费者已找到、任何实现可行或性能提升。
