# F22：shared expert 两级投影与激活的分块依赖

## 范围、区别及源码支路

F22 是 shared expert `gate_up_proj → SiluAndMulWithClamp → down_proj`，与 F21 routed expert `DispatchFFNCombine` 内部的 expert 计算不同，也与 F23 的 shared/routed 输出合并不同；相同设备算子名称可能在这些路径出现，不能把候选收益相加。`evidence/model-config.json` 指定 hidden 5120、`moe_intermediate_size=2304`、1 个 shared expert、`swiglu_limit=10`。

冻结 `model.py:104–146,193–207,236–258`：当 `n_shared_experts is not None`、`mix_placement=False`，单独构造 `DeepseekV41MLP` 并交给 `FusedMoEFactory`；shared `gate_up_proj` 是 `MergedColumnParallelLinear`，在一次模块调用中生成 gate 和 up 两组，且激活使用 `SiluAndMulWithClamp(10)`，接 `RowParallelLinear` down。gate/up 的组合模块已存在，不能当成两个尚待合并的独立调用。`mix_placement=True` 时并不构造独立 shared 模块，而把 shared 纳入 factory 的另一条路。源码 `ascend_config.py:587` 默认 `mix_placement=False`，不等于历史进程 resolved 值。runner 内部何时触发 shared MLP、共享激活的设备实现、输出 tuple/tensor、各 rank 的实际 TP/SP 配置与张量地址均未绑定。

模型源码在 `is_sequence_parallel=True` 时传入 `disable_tp=True`；但 stage231 当前 Ascend `get_parallel_op` 对 shared expert 特判，是否以 TP=1 复制权重取决于 `shared_expert_dp_enabled()`，不能由 SP 单独推断。TP 分片及其后归约必须按实际实例、runner 与 `model.py:300–326` 分支处理。冻结配置曾请求 TP8/DP2/EP，SP 谓词需要 resolved backend/patch，不能单凭启动命令断定实际 SP。checkpoint 描述至少明确 `layers.0.ffn.shared_experts.w1.weight` 为 `W8A8_DYNAMIC`，169 在只读时的描述文件 SHA 与参考一致；这不是历史实例方法、全部层或激活计算 dtype 的证明。

## 采样支持及未知

`f07-q-kv-streams.json` 的 EP8、layer24、run2 同一 stream 89 中，窗口后段依次出现：`QuantBatchMatmulV3` task1300，输入 INT8 `[3,5120]`、记录输出 INT32 `[3,4608]`（1260.855–1286.835 µs，相对窗口开始）；task1302 输入 INT8 `[3,2304]` 和量化权重，输出 BF16 `[3,5120]`（1292.375–1312.954 µs）。4608=2×2304，形状与 **假设 shared 权重复制且输入 3 行**的 shared gate/up 和 down 相容，但这两个记录不识别实际层对象、scale 值或 tensor 地址；不能单凭此推断 SP、确为 shared expert 或不同 GEMM 的数据连接。目标层 24 的 48 个跨 rank/轮窗口中均出现一个 `DequantSwigluQuant` 和一个 `DispatchFFNCombine`，`captured-layer-attribution/operator_metrics.csv` 报的 count 是角色/时间约束统计，不是二者逐对关联，也不是额外实现的性能成果。

stage157 由 ZCode `deepseek-flash` 对同一历史窗口做只读查询，独立核验 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-157-f22-layer24.exit-code` 为 0，`.verified.json` 的 `model_verified=true`。结果 `f22-layer24-run2-evidence.json` SHA256 `0cae38a2a726f1471f651cd664f0dcb4f608734dbb59f8603d1315ce02084dfb`，查询脚本 SHA256 `2346652ae7371f4f4cef135da4abc95194e7fb2182c4711663f5a9b76861042e`，输入 CSV SHA256 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 符合冻结清单。均为相对窗口起点 µs：

| stream 89 事件 | task | 时间 µs | 关键输入 → 输出 |
|---|---:|---:|---|
| `DispatchFFNCombine` | 1298 | 955.061–1255.835 | BF16 `[3,5120]`，INT8 expert 权重等 → BF16 `[3,5120]` |
| `DynamicQuant` | 1299 | 1256.795–1260.815 | `[3,5120]` BF16 → INT8 `[3,5120]` 与 FP32 `[3]` |
| `QuantBatchMatmulV3` | 1300 | 1260.855–1286.835 | INT8 `[3,5120]` → INT32 `[3,4608]` |
| `DequantSwigluQuant` | 1301 | 1286.855–1291.415 | INT32 `[3,4608]`、FP32 `[4608]` 和 `[3]` → INT8 `[3,2304]` 与 FP32 `[3]` |
| `QuantBatchMatmulV3` | 1302 | 1292.375–1312.954 | INT8 `[3,2304]` 和 FP32 `[3]` 等 → BF16 `[3,5120]` |

这个窗口中观察到一个 `DequantSwigluQuant` 复合任务，且其物理输入输出形状及同流顺序与两级 projection 和中间激活/量化链相容；这是实际任务已有的组合，不能当成若干独立待融合任务。`DispatchFFNCombine` 在该候选任务序列之前结束，**这些记录**在该 stream 未重叠，但不能推断内部通信、其他 stream 或 rank 无并行。记录的 `[3]` scale 形状与逐 token scale 相容，不证明其数值、归约域或对应输入的身份；`DequantSwigluQuant` 的 exact clamp 值、参数/内部实现和 Python `SiluAndMulWithClamp` 的绑定都未被这些字段证明。不能用 task 临近、连续 ID 或形状直接断言 shared expert 已确认、张量相同或调用链逐项匹配；也不能以观察到的耗时评估可节省时间。

## 分块重组的条件和不可跳过的依赖

1. **第一次投影与 activation**：对于第 t 个 token 和通道 j，激活要求该 token 的 gate_j 与 up_j 都准备好。前投影沿 hidden 维的两个点积各自的全部必要输入与权重 tile 必须完成，不能用只完成部分 K 的中间和直接激活；但不同 token / 不同 j 可独立处理。gate 与 up 排列、clamp 上下限和位置、Silu 与乘法原有计算精度、两路投影/反量化的 scale 和 offset、饱和与舍入都必须保持。源码命名 clamp=10 不能作为精确上下限/顺序的实现证明，所需 activation 原始源码或 native 数学契约仍缺。
2. **activation 与第二次投影**：down 输出每个通道依赖全部对应的 intermediate 贡献（或者保证等价的部分累加顺序）。若中间执行逐 token 动态量化，那么完整量化域的 scale 依赖该域全部激活，单个 j 的局部最大值不能代替整行 scale；如果具体执行是 group-wise、不同域或已经生成 scale，则依实际接口而定。可以研究把已就绪块流入 down 的分块累加，但其处理 scale 的归约域、INT8 值、saturation、累计 dtype 与舍入均未知。不能直接把两组权重预乘或跳过非线性。
3. **shared 与 routed 的并行**：两分支在各自输入 ready 后源码结构上可独立运算，但最终合并不能早于各自正确输出及所需通信；如 runner 允许并行，底层仍可能争抢 Cube、Vector 和 MTE，因此 overlap 不是免费。后续已取得当前 runner 和 shared expert 的部分事件路径，见文末；完整历史 runner 事件 DAG 未绑定，不得声称历史样本中的重叠或真正完成点。
4. **已存在的融合与待证接口**：若窗口确有 `DequantSwigluQuant`，这是一个复合设备任务名称；它具体包含的 dequant、clamp、激活、quant 和 scale 流程取决于该 op 的正式定义及版本，不应重算成一串独立待融合任务。可研究连接第一 GEMM 的 accumulator 或把激活量化的结果按 tile 供 down，但需要确认对应算子接受的输出/分块接口、数值及 buffer 生命周期，不能将条件候选当成已实现功能。

本记录不改实现、不改设备、无新 NPU 测试；当前是输入/输出与可并行约束审计，性能收益和理想上界不由这份形状记录推导。

## 固定单样本的并行账本：仅说明待证问题

对上表 stage157 的同流时间直接相减：MC2 长 **300.774 µs**；1299–1302 的整体跨度 **56.159 µs**（四条任务各自时长之和 55.139 µs，中间空隙合计 1.020 µs）；1299 在 MC2 结束后 0.960 µs 才开始，因此两段现有记录均结束的时刻为 1312.954 µs，晚于 MC2 结束 **57.119 µs**。这些只是同一采样窗口的时间差，不能证明任务依赖、实际空闲或该部分处于请求关键路径。

作为严格限定的数学示例，**假设**这四项实际对应独立 shared 分支、所有输入在 MC2 开始时已经就绪、允许把整个 56.159 µs 的时间段平移到 MC2 内、资源互不干扰且各任务耗时不变，同时其他必要工作与完成判据不变，则两段都完成的时刻可移至 MC2 结束的 1255.835 µs，账面缩短 57.119 µs。这只说明在这些假设下时间段放得进 MC2，既不是证明假设成立，也不是实际可实现收益、端到端提升、整个系统的理论上界，其他调度/融合变化也不受此数值约束。真正判断前须先绑定实际 runner、tensor 与同步关系，并观察重叠引入的 Cube/Vector/MTE 和带宽争用；缺少上述证据就不能对吞吐或耗时改进作数值承诺。

## 通用 linear 源码与观察到的 INT32 设备输出之间的未解绑定

对照冻结的 `agents/fusion/frozen-source/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py`（SHA-256 `a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910`）：`AscendW8A8DynamicLinearMethod.apply` 第 102–133 行在输入非 PreparedLinearInput 时先调用 `npu_dynamic_quant`，再以 `output_dtype=x.dtype` 调用 `npu_quant_matmul`；PreparedLinearInput 分支则使用其保存的 output dtype。因此若执行的是该通用方法且输入 dtype 为 BF16，**Python 接口指定的输出 dtype 是 BF16**。stage157 的第一条 QuantBatchMatmulV3 导出的是 INT32 `[3,4608]`，而不能从通用方法直接逐项推导为同一层对象的 Python 输出。图改写、其他覆写、底层算子展开或记录语义均需另证，不能排除其中任何可能，也不能由此断言冲突或编造解释。第二条输出 BF16 `[3,5120]` 与该接口 dtype 相容，但仍无逐调用绑定。

同一文件第 407–474 行在特定条件下的 routed MoE `apply_gmm1_act_quant` 分支也包含 INT32 `npu_grouped_matmul` 后接 `npu_dequant_swiglu_quant`。因此仅凭 `DequantSwigluQuant` 名称或 SwiGLU 模式不能确认 shared expert。该 routed 源码用 grouped matmul，与所选窗口导出的普通 `QuantBatchMatmulV3` 名称并不相同；它只用于指出算子名不是 shared 独占的证据，并不证明这条替代路径实际发生或与任务一一匹配。

这两点加强原判断：两个矩阵任务与 F22 shared gate/up、down 的角色和次序相容，但仍是条件性归属，不能当作已证明的调用身份，也不能与 F21 或 F22 分别重复计算性能收益。现有局部 trace 未提供 tensor/storage 身份和 Python-to-device 调用链接，重复同一单窗提取不会补足缺口；若已有历史图编译记录、调用关联或其他冻结接口信息，可优先查其是否能唯一绑定该具体任务和实际方法，否则保留未知，不启动 NPU。

## stage229：当前 Ascend 激活 wrapper 的精确 clamp 次序

ZCode stage229 对容器固定 `vllm_ascend/ops/activation.py`、`ops/linear.py` 做只读快照，分别 SHA-256 `79185cd8bae9801aef50873cb082bb977ff8d2b3ddb27766e4f42ffbfa12b5c0`、`7654777d9252fc6a3cc2c95f92b67a42410a0de24a5912ef20e807cf64ab74e6`。`agents/bounds/sampler-patch-snapshot/adapt_patch.py:829–870` 的当前注册表将 `SiluAndMulClamp` 指向 `AscendSiluAndMulWithClamp`；这并不证明历史实例命中注册表或当前文件与历史编译对象相同。

当前 `activation.py:41–48` 的 `forward_oot` 先按最后一维等分：gate 是前半，**仅**限制上界 `+swiglu_limit`；up 是后半，限制在 `[-swiglu_limit,+swiglu_limit]`。它再 `cat([gate,up])`，然后调用 `torch_npu.npu_swiglu`。因此，对这一路的等价重组不能将 gate 误改为双边 clamp，也不能把 clamp 移至调用之后而不另行证明等价。这里仅给出 Python wrapper 操作顺序；没有证明下层实现的中间精度、逐位舍入或设备融合指令。源码模型构造使用配置值，但实际运行 limit、图改写和历史任务身份尚未绑定。

该激活 wrapper 中没有显式传入 INT32 accumulator、weight scale、dynamic scale 或重新量化的接口，不能仅凭它解释 stage157 的 `DequantSwigluQuant`；不排除其他覆写、图改写或设备调用。`linear.py:251–302,305–391` 的 Ascend MergedColumn/Row 两个 forward，在 `self.custom_op` 非空时调用 `self.custom_op.apply(input_)`，否则调用父类 `forward`，本文件内没有把观察到的 INT32 设备任务直接绑定到激活的实现。选择入口在 `vllm_ascend.ops.linear_op.get_parallel_op`；需另行定位选定分支和历史对象，不能由本次快照认定 shared expert 角色或历史性能收益。

## stage231：shared expert 线性并行选择与中间 dtype 的范围

`agents/fusion/frozen-source/vllm_ascend/ops/linear_op.py` SHA-256 `2a28240882788db363c14a0999b1cad68b4f9f63896d0ad9475f2d39051e818b`，由本机 ZCode 的 stage231 从当前固定容器只读获取，退出码为 0、model_verified=true。此文件是当前版本快照，不是历史运行二进制及 resolved 配置的证明。

`model.py:118–145,193–208` 中，独立 shared MLP 构造的两个线性前缀分别包含 `shared_experts.gate_up_proj` 和 `shared_experts.down_proj`；运行其 forward 时依次调用 gate/up、activation、down。`linear.py:276,295–302,327,383–391` 的当前 Ascend 实现使用 `get_parallel_op` 选择 custom op。

`linear_op.py:406–407,432–441` 对 shared-expert 前缀有特殊处理：`shared_expert_dp_enabled()` 为真时，返回 `(None, 0, 1)`；如果该开关为假，则不会对 shared expert 执行普通的 `elif disable_tp` 早返回，即使模型的 SP 标志传入了 `disable_tp=True`。因此本文件支持的结论是：**不能依据 SP 参数独自声称 shared expert 权重必然复制**；本记录先前将 SP 直接等同于复制的表述已经纠正。它也不证明历史共享 expert DP 开关、参数最终布局或者与该快照相同的注册实现。

如果 shared-expert DP 开关为假，当前代码继续走 `_get_column_parallel_op` / `_get_row_parallel_op`（410–429 行）：前缀匹配 gate/up 或 down、`mlp_tp_enable()` 为真并且 `is_moe_layer(prefix)` 为假时，可分别选 `MLPColumnParallelOp` 或 `MLPRowParallelOp`。`is_moe_layer` 根据 `layers.<n>.` 和运行中的 HF 配置判断，`first_k_dense_replace` 缺省为无穷（479–498 行）；不能因前缀含 shared_experts 即认定该谓词一定为真。若 custom op 没有命中，返回默认 TP group 的 rank 和大小（451–468 行）；实际结果和父类 forward、线性层实例是否采用这些实现仍未绑定。

若这两个 MLP custom op 被选择，column 路径先按 dim 0 `all_gather` 再调用 `quant_method.apply`；row 路径先取得本分片输入，调用 `quant_method.apply`，再按 dim 0 `reduce_scatter`（263–302 行）。这两个 apply 包装没有显式返回 INT32 accumulator 给激活，也没有显式将激活融合进 GEMM；具体方法由实例量化配置选定（`linear.py:165–170`）。**条件性地**，如果该实例是已存 W8A8_DYNAMIC 普通 linear 方法且输入 BF16，那么该方法请求 `npu_quant_matmul(..., output_dtype=x.dtype)` 即 BF16；这依旧不证明 profiler 的 INT32 `[3,4608]` 不可能属于底层展开，也不能判定图融合或其他实现。stage157 的张量身份、运行时量化方法、图/二进制和中间数据语义仍未知。

下一条能收紧判断的证据：既有运行工件中若有该 shared 层两个完整 prefix 的实例类、resolved `shared_expert_dp_enabled`、`mlp_tp_enable`、HF `first_k_dense_replace`/`moe_layer_freq`、量化方法和实际 TP group，连同图编译或 Python-to-device 关联，才能把条件选择与具体历史任务绑定；仅看静态默认或时间相邻不够。若历史工件没有这些信息，保持 UNKNOWN，不能把当前源码当成历史调用路径。

## 既有启动记录中的 shared expert DP 请求

`evidence/frozen-serve.sh:28` 明确请求 `--additional-config` 的 `enable_shared_expert_dp=true`；独立保存的 `benchmark-20261028/raw-results/service-log-tail.txt:24` 是基线服务 APIServer 的 non-default args 打印，同样包含该值，文件 SHA-256 `64bfa3d8523ad86e1c466fcca166cdcfe1dc4a3676573fc51b65279a69aa8aba`。`analysis/operator-ceiling-20261028/diagnostic-launch.json` 记录诊断启动脚本基于 SHA 为 `c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba` 的基线脚本，只增加 profile 参数并声明语义参数不变。上述提供了诊断配置意图的较强证据，但该基线启动日志不是采样进程的 resolved 配置或该方法实际返回值，也不证明历史 shared 实例使用 stage231 源码。若实际实例用此源码且 `shared_expert_dp_enabled()` 返回真，则 shared 线性在该选择函数中返回 TP1；不能直接把这一条件结论当作观测。

已存 `evidence/model-config.json`（SHA-256 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`）列出 `n_routed_experts=384`、`n_shared_experts=1`、`moe_intermediate_size=2304`，文件内没有 `first_k_dense_replace` 与 `moe_layer_freq`。这只说明保存的 JSON 字段，不证明采样进程实际 HF 对象未被转换、补充或覆盖。此前 checkpoint 描述的 W8A8 标签也不是 layer24 实例方法或设备任务绑定。限定搜索已保存的基线服务日志、诊断启动记录、启动脚本、相关历史配置分析及配置查询结果，没有发现实例级 `shared_expert_dp_enabled()`/`mlp_tp_enable()` 返回值、这两个 HF 字段或线性量化实例与样本的绑定；搜索范围不能证明其他未保存文件中不存在。下一步仅在找到指向诊断进程的已存 resolved 配置、编译图或对象调用工件时继续绑定，否则保留未知，不以源码默认填补。
## shared gate/up 与 MC2 路由初始化的输入量化交叉边界

此前 F22 查了 shared 内部两次投影，也查了与 routed 分支计算并行；这里单独审查**两条分支各自进入第一矩阵计算前的输入量化是否能共用**。当前保存的 `w8a8_dynamic.py:61–64,94–119` 声明普通 W8A8 linear 采用动态逐 token 激活量化：普通入口调用 `npu_dynamic_quant(x)`，另有 `prepare_input_for_overlap` 生成 `PreparedLinearInput(quantized_x, pertoken_scale, x.dtype)`，`apply` 收到该类型时复用其中数据与 scale。仅证明该方法支持预制输入，**没有证明** shared 实例已用此入口，或由此可跨 MC2 复用。

在另一分支，当前冻结 `dispatch_ffn_combine_kernel.hpp:797–807` 的已有复合 MC2 内，先 `ApplyXActiveMask`，再将 `ptrA`、`expertIdx` 及 scale/offset 指针交给 `moe_init_routing_quant_v2`，并将输出写向路由共享内存及 per-token scale 空间。`dispatch_ffn_combine.h:75–76,282–284` 里该实例初始化的两个 `moeInitRoutingQuantV2Scale/Offset` 指针为 null；这不能由名称推导它的实际量化公式或完整输出布局。MC2 已将路由初始化和量化放入一个复合调用，没有现成的外部预量化输入参数。此前本节未定位该 helper 实现，现已在另一既存证据目录找到部分完整源码，见下节；MC2 所见 `ptrA` 与 shared 普通入口的张量、行范围、运行时量化方法及当前快照与历史二进制绑定尚未证。

**可证伪的条件重组**：如果实际 runner 确认两条路径消费同一版本的相同 token 行，且 MC2 的路由初始化量化对每一行产生的 INT8 值和 scale 与 shared 线性方法完全相同，则可研究一次量化后分别供 shared GEMM 和 MC2 的路由打包使用，或让路由初始化额外发布原始 token 顺序的可用量化副本。必须取得 helper 的精确约化范围、scale 定义、offset、取整、饱和、零/NaN/Inf 语义和实际模板选择，与 `npu_dynamic_quant` 比对；分组或分片不同的行不能直接共用。路由侧可能按 expert 复制、重排或搬到 peer，shared 消费的是对应 token 行：即使数值相同，还须证明输出排列、scale 关联、通信和访问可用，不能拿 expert 排列当原顺序。若 helper 不产生相同的 `(q,scale)`，或 MC2 输入与 shared 实际不同，则该无需转换的复用条件被否证。

量化域不能只按名称相同假定等价：例如按对称 max-abs INT8 量化，整行 `[1,100]` 的缩放量级为 `100/127`，只处理第一个 feature 时为 `1/127`；对于元素 1，两者所得的整数不同。因此未经证明不能让按 feature 局部提前得到的 scale 替代整行 scale。这是一般数学反例，**不是声称这两个真实接口采用所示规则**。共享缓冲还需有明确的生产完成和两消费者结束事件，避免异步覆盖，并保持现有 MC2 原位路由 ID 行为：`dispatch_ffn_combine_kernel.hpp:360–394` 在非空 active mask 时把 inactive token 的 expert 索引写为 `expertPerRank*EP`，接着才启动 routing；不能随意重排/覆盖有其他观察者的索引。串流合并还可能争用 Cube、Vector、MTE 与网络，并破坏 F21 已有 dispatch/GMM 重叠。当前没有足够的真实绑定或各项内部时间，不能断定这条路径真的重复量化或估算节省。

## 既存 helper 源码与官方接口：公式相近但尚不能证明逐位相同

现从 `/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/` 找到该 helper 的源码；所引四个文件已按相同 SHA 复制到本任务 `agents/fusion/frozen-source/csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/`，host tiler 也已保存到该任务的 `op_host/`（SHA-256 `783f76d7869371df949fc5ee80e33dfca8cf81231a65da410fcfea40df89b6f8`）。`moe_init_routing_quant_v2.cpp`、`moe_v2_fullload_dynamic_quant.h`、`moe_v2_gather_dynamic_quant.h`、`moe_init_routing_quant_v2_tiling.h` 四文件本地 SHA-256 分别为 `1766ef19445df82378f09c93a76938362e04dee27cf1160e4ae23c208f84fe2e`、`ba554c6028122f1efa3813738cf0267c011a86543282552629030ff76e51a51a`、`7e82750ac1fee1777ae77c30042c1fff82aded2ba3bc9bf9b4b89b4a891ef504`、`ab50821407cad284205dc3bed56f36270760dd68860f90f6fb6ae4ce1bd16330`，与原 `precision-paths-20261028/MANIFEST.json` 一致；仍未将这些文件与历史 profile 的实际编译对象绑定。

`op_host/dispatch_ffn_combine_tiling.cpp:262–275` 给内部 tiler 的 `quantMode=1`、`scaleDim0=0`；上述 tiling 文件 `:128–140` 在此条件设置 `SMOOTH_NONE`。入口 cpp `:46–63,120–130` 对 `21000` 选择 full-load 动态量化，对 `11000/11010` 选择 gather 动态量化；真实 tiling key 由输入 shape、平台及 tiler 决定，不能指定历史样本走哪一支。对 full-load `:220–256`，有源码明确的 per-row FP32 数据转换、`abs`、全行 `ReduceMax`、除 `127.0f`、按此 scale 除原值、再 **float→half 的 `CAST_TRUNC`，half→int8 的 `CAST_ROUND`**；gather 的对应代码及大行分块 `:121–151,260–330,355–381` 同样展示全行最大值归约和上述两次 cast。量化精度、特殊值、平台实际编译与实现细节仍需目标版本契约，不应把其它 tiling 分支推定完全一致。

[昇腾 TorchNPU 7.3 官方 `npu_dynamic_quant` 文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_dynamic_quant.md) 将其描述为逐 token 对称动态量化，无 smooth 时按一行绝对值最大值除输出类型最大值求 scale，再对商执行 round；INT8 公式与上述 helper 的 `max/127` 在有限、非零、无 smooth 的实数表达上相容。这是**文档层面更强的相容性，非 bit-exact 证明**：该文档的 round 描述和这里所见 helper 的中间 half 转换、具体溢出/零/NaN 行为未逐项绑定；版本为 TorchNPU 7.3，不能当作历史环境已验证的执行语义。[CANN 9.2.0-beta.1 舍入模式说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/API/ascendcopapi/docs/zh/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%9F%BA%E7%A1%80API%E5%88%97%E8%A1%A8.md) 描述 `CAST_ROUND` 四舍五入、`CAST_TRUNC` 截断；beta 文档也不能直接确认为历史 9.2.0 编译对象的实际语义。

例如 BF16 行 `[5,254]` 的数学 scale 为 2，首元素商恰为 2.5：假如 MC2 依上述 `CAST_ROUND` 把 2.5 转为 3，而普通动态量化实际采用最近偶数并直接对 2.5 舍入，则得到 2。该例明确说明**数学公式相同也不能单凭接口名证明 q 相同**；普通动态量化实际使用哪种模式及本例在目标设备的结果未知，这不是实测差异。即使整数相同，还要核对 scale 计算、零值/非有限值及中间舍入。

布局也更明确：full-load `:263–303,321–324` 对同一原始行计算一次后，依据 `expandedRowIdx` 向每个有效路由位置复制；INT8 行存储步长为 `cols+UB_ALIGN`，scale 写入同一局部输出的 `outLocal[cols]`，不是普通 linear 的原始 `[T,H]` INT8 与独立 `[T]` scale。`dispatch_ffn_combine_kernel.hpp:312–352` 再将带附加区的行与 FP32 scale 拆至目的缓冲。多核边界可能重复计算同一行，屏蔽/无效/容量分支还会影响输出；没有证据证明任意 shared 行都有可直接引用的 MC2 路由副本。若仍要共享，需额外证实或提供原始顺序、独立 scale、完整有效行和消费者稳定生命周期。以上均为源码与接口条件比较，没有调用 NPU、也没有实证收益。

## runner 输入分叉、mask 和当前 overlap 顺序

继续限定本地 `precision-paths-20261028` 快照，`fused_moe.py`、`routed_experts.py`、`moe_comm_method.py`、`ascend_forward_context.py` 的 SHA-256 分别为 `2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4`、`f9db74836ae210a541432b923118696c7a15be9a10d5776834e1d3b64ce11d77`、`7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7`、`848112214c9f56371b4bf43a59d39fadbed0e0e864581e300a3a89778d4083cb`，本机计算与该证据包 MANIFEST 一致。对应历史样本的 Python 类、对象、存储身份、执行路径和设备二进制仍未绑定。

在当前源码条件下，模型 `model.py:270–298` 将 SP 条件切片后的 `hidden_states` 传给 experts，同时将可能不同的 FP32/router 输入传给 router 参数；不能把 router 的 FP32 输入误认为 shared/MC2 激活输入。`fused_moe.py:179–200` 随后允许先做 routed input transform、或由调用者提供不同的 `shared_experts_input`，还可能对 routed hidden 做 `_maybe_pad_hidden_states`。`_forward_impl:493–520` 使用 `shared_experts_input` 非空时作为 shared 输入，否则才使用 routed hidden；而 routed 路又调用 `routed_experts.forward_impl`。因此**外层输入同源不保证最终两条量化调用读取同一 tensor 或同行域**：只要实际 routed transform 改值、或只对一支 padding/gather，就不能直接复用另一支的原始顺序量化。该源码对 routed transform 的通用支持不证明 V4.1 的实际实例使用它。

更具体地，`routed_experts.py:717–746` 在量化之前调用 `moe_comm_method.prepare(hidden_states, ..., replace_allreduce=...)`，用返回的 `prepare_output.hidden_states` 替换 routed 输入，再把它交给 quant method；`w8a8_dynamic.py:203–229` 把该 `x` 放进 fused experts 输入，非 MegaMoE 的 `moe_comm_method.py:587–605` 将其作为 MC2 `x`。shared 分支则由 `fused_moe.py:502–507,527–538` 通过 `prepare_input_before_routed` / `PreparedSharedExpertInput` 交给 `AscendSharedExperts.forward`。这证明两个**准备节点分开**；stage251 已补齐当前容器 `PrepareAndFinalizeWithMC2` 与 `AscendSharedExperts` 的实现，具体条件见下节。上游 transform 和实际实例仍未绑定，无法确定 shared gate/up 与 routed 的实际输入是否相同；需要真实 resolved mode、逐层 tensor/row/scale 绑定，而不是以 shape 或两者源于 hidden 就下结论。

mask 的来源可进一步限定：`ascend_forward_context.py:222–232,281–290` 在满足条件且保留 mask 存在时，用 `num_actual_tokens` 将前段写 True、至 padded_num_tokens 的尾段写 False；`prepare` 的 `mc2_mask` 经 `routed_experts.py:724–746` 和 `moe_comm_method.py:595–605` 传为 MC2 的 `x_active_mask`。这是一条条件路径，当前 prepare 实现按条件对 mask 切片或补齐，见下节；不能直接把 context 中的原始长度等同于设备输入每个位置。若某行为 inactive，MC2 `ApplyXActiveMask` 会改其 expert ID，routing helper 可不产生对应有效路由副本；即使 shared 分支仍需要该行，也不能从 MC2 输出无条件找回它。反过来不能仅凭存在 mask 宣称某行实际 inactive 或 shared 确实消费 padding。

当前还有可证明的**时间顺序约束**：`fused_moe.py:450–482,509–520` 在特定 multistream 路径先提交 shared gate/up，随后调用 routed `forward_impl` 才提交 TopK 与 dispatch；该准备及计算的完成条件由相应 event 控制。因此在这一路中，若提出让 shared gate/up 使用 MC2 **内部稍后才产生**的量化副本，就必须新增该数据的就绪依赖并重排提交/运行，不能声称同时保留原先 gate/up 早于路由提交的依赖图，也不能将“省一次量化”直接当作省下延迟。可另行研究更早一次量化同时供两支，但必须先解决两支的 transform/prepare、数值、通信和生命周期。其他非 overlap 分支以及历史实际运行情况均未知，现有信息不证明这项复用可实施或不可实施；未运行设备。

## stage251 当前容器：SP / shared DP 的具体行域反例

本机 ZCode stage251 在固定 169 staging 容器中只读取指定文件，退出码 0、`model_verified=true`；结果记录容器 ID `42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602`、镜像 `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b` 且源路径无覆盖挂载。保存 `agents/fusion/frozen-source/vllm_ascend/ops/fused_moe/shared_experts.py` 与 `prepare_finalize.py`，本机独立重算 SHA-256 分别为 `471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9` 和 `dff4fac292200fc06f60aef48505f1ee38314ea5adb53560b325901db308eb24`，均与 stage251 和本机外部参考副本一致。此前缺当前容器这两个文件的证据已补齐，下述分支推导现在有当前容器源码依据。**当前容器快照并不证明历史 profile 当时加载该源码、配置、对象或二进制**；下述仍是静态条件推导。

`shared_experts.py:120–126,208–227` 依据 SP 和 `enable_shared_expert_dp` 选择：SP 且 shared DP 为真，使用已切分的 SP shard 和复制权重；SP 且 shared DP 为假，shared 需要先 all-gather 全序列；无 SP 且 shared DP 为真，shared 自己 pad 并按 TP rank 切分原始行。`prepare_finalize.py:252–313` 的 routed MC2 输入则依据 `replace_allreduce` 选择：为真时保留传入本地 shard，必要时按 DP-wide 目标长度补尾，并切取/补齐 mask；为假时按 DP-wide 目标长度 pad 全序列，再按 TP rank 切分 hidden、router 和 mask。`routed_experts.py:717–723` 将 `self.moe_config.is_sequence_parallel` 作为此参数。

两条路径可以出现不同 token 行：**示例**无 SP、shared DP=true、TP=2，本 rank 输入原始 T=3 行 `[a,b,c]`，另一个 DP 位置使 MC2 padding 目标长度为 6，且假设两支无 transform、输入顺序相同。shared 先补到 4 再等分，rank 1 的行是 `[c,pad]`；MC2 先补到 6 再等分，rank 1 的行是 `[pad,pad,pad]`，rank 0 的行数也分别是 2 和 3。即使两者都叫当前 TP rank 的输入，也不能逐行直接共用。该例符合参考两处 padding/split 公式，**不是观测到的基线 batch**，实际 `padded_num_tokens`、TP group/rank、prepare 实现和 transform 均须绑定。

SP=true 且 shared DP=false 的参考分支让 shared all-gather 后使用完整有效序列，MC2 路则保留本地 shard、可能额外尾 pad；两者通常不是同一完整行域，不能把单 rank 的量化结果视为完整 shared 输入。SP=true 且 shared DP=true 时，两支在分叉后同源且无 transform 的额外条件下都可能使用同一 shard 的有效前缀；若 MC2 补尾、shared 不补，则仍不能把完整 tensor 等同，需以 row/slot 和 mask 证明交集并处理 shared 需要的其余行。无 SP 且 shared DP=true 也只有当实际两边 padding 长度、TP group/rank、行顺序一致时，才可能对齐；身份/视图与数值仍各自需要证明。启动参数请求 shared DP=true 是前文的配置证据，**不是**已确认的运行 mode、SP 值或输入相同。

该参考 `shared_experts.py:286–309,312–333,363–388` 还给出事件顺序：特定 SP-only 多流路径可在 shared 流等待输入事件后发起 all-gather 并记录 ready event，普通 early gather 则在调用流完成提交；shared 的执行输入通过 `shared_input_ready` 在相应流等待，SP-only 分支还可能裁掉 gather 后的 padding。`:390–435,443–453` 允许多流下在 router matmul 前准备 shared gate/up 的量化输入，`:455–470` 等 router ready 后提交 gate/up。`:670–715` 显示输出后续还有 routed finalize 等待或 shared 输出等待。这些特定条件下的生命周期意味着一份未来共享的量化 buffer 必须从生产完成延续到两个消费流各自完成，并保持 collective 次序；不能只看数据相同就去掉 event 或把实际 MC2 内部稍后量化倒供已提交的 gate/up。

当前源码 `:425–429,471–481` 还有 A8_INT_FUSED 条件分支：先动态量化 shared 输入，gate/up `npu_quant_matmul` 请求 INT32 输出。它提供了此前 stage157 INT32 形状的**另一种具体相容解释**，但实际选择的 shared 路径、tensor 身份和历史所用源码都没有与历史记录绑定，不能据此把 task1300 定性为 shared。该证据可缩小源码可能分支，却没有证明当次复用、性能收益或实验结果；本轮只读本机资料、无 NPU 操作。

### 启动参数与 SP 的条件关系

冻结启动脚本请求 `enable_shared_expert_dp=true`、`enable_flashcomm1=true`、EP 和 TP8；诊断启动记录声明语义参数不变，但这仍是请求及启动构造证据，未提供诊断进程的 resolved 参数日志。当前 `ascend_config.py:743–769` 中 shared DP 还要求 EP 且 TP>1；FlashComm 有效时仍检查 `use_sequence_parallel_moe`，不能仅由请求 FlashComm 推出 SP。冻结上游 `parallel.py:715–729` 和 Ascend patch `patch_parallel_config.py:7–28` 的判定均要求允许的 all-to-all backend、EP、TP>1；上游另要求 DP>1，patch 不要求。若 backend 实际为冻结源码默认 `allgather_reducescatter`，且请求的 EP/TP8/DP2 均如实生效，则两种谓词都推导 SP=true，shared DP 的源码门槛也满足；这不是诊断运行中的实测属性、不是已证明选择 SP+shared DP 行域。没有找到能直接绑定诊断实例的 resolved backend、SP 和 shared DP 日志；其他服务日志不能用来填补。

冻结启动还请求 `enable_fused_mc2=1`。当前 `ascend_config.py:578,832–846` 的 `multistream_overlap_shared_expert` 默认 false；若进入校验时 fused MC2 的值为 1 且 shared 多流为 true，代码将该 shared 多流开关强制设为 false。**若**诊断实例实际按当前源码完成此配置构造，且该值之后未被其他路径改写，则本任务前述 shared 多流提前量化/发射分支不会按该开关启动；实际诊断 resolved 值和实例仍未绑定，因此不是观测结果，也不能仅据此否定所有计算与通信的内部重叠。该开关专属于 shared expert，与 `multistream_dsv4_dsa_overlap` 不同，不应把它用于判断 attention 的多流情况。

stage255 的限定查询对固定诊断容器对应的完整 `service.log` 计算 SHA-256 `51375755909e205512f1a7781627d79cca628d5970b198a51ab9de7247ee295d`，第 24 行匹配到 `enable_fused_mc2=1`、`enable_flashcomm1=true`、`enable_shared_expert_dp=true`；ZCode 验证通过、退出码 0。结果文件 `agents/diagnosis/diagnostic-runtime-config-evidence.json` 仅保留匹配字段和值，不保存原始日志行。stage256 对**相同哈希**日志的第 24 行做固定类别查询，得到 `api_server` 和 `non_default_args`，独立结果见 `agents/diagnosis/diagnostic-line24-context.json`；退出码 0、`model_verified=true`。因此可将这三个值描述为**该诊断 APIServer 日志报告的非默认请求参数**，但不能等同于各 worker 实际解析、生效的 resolved 配置或已绑定采样实例的值。其他已搜索字段及固定告警未命中，只表示本次固定键值/精确文本查询没有匹配，不能按 false、默认值或从未执行解释；也不能由未命中的告警反推 SP 或多流状态。

## 2026-10-29 prefill 重采样：量化复用的实际物理行域反例条件

对同一次重采样已有 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` 做只读 CPU 汇总，严格取**同一个 target 层 CPU 提交 scope、每个名称恰有一个经 flow/设备 task/CSV 唯一匹配任务**的 `DispatchFFNCombine` 与 `DequantSwigluQuant`。源文件 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`；16 个 rank 各 80 个 scope，满足筛选的共 **1280/1280**，结果与各 EP 数量见 `agents/fusion/f22-recapture-physical-row-domains.json`。这里的 1280 是**层提交范围内的两类设备记录组合**，不是已确立依赖的 1280 对数据对象。

| 匿名 step digest 前缀 | EP | 40 层 × EP 记录数 | MC2 第一个输入的物理 shape | DequantSwigluQuant 第一个输入的物理 shape |
|---|---|---:|---|---|
| `7ddf7ae98a00` | 0–7 | 320 | `[643,5120]` BF16 | `[595,4608]` INT32 |
| `db852bff987d` | 0–7 | 320 | `[1023,5120]` BF16 | `[1023,4608]` INT32 |
| `1b826883b782` | 8–15 | 320 | `[1023,5120]` BF16 | `[1023,4608]` INT32 |
| `8c18c293afc6` | 8–15 | 320 | `[643,5120]` BF16 | `[643,4608]` INT32 |

具体可追索示例 EP0 / layer0 / `7ddf...`：MC2 flow `1793241333374667046`，DequantSwigluQuant flow `1793241333401242975`。上表来源为导出的物理 shape，**不是有效 token 数，也不能直接解释 48 行差为 padding、缺失数据或浪费**。同提交 scope、相邻顺序及 shape 均不能证明 DequantSwigluQuant 的实例就是 shared MLP，也无法证明它与该 MC2 的张量共享原始行或直接连边；重复同名任务、其他流和未匹配事件均可能形成不同解释。

条件地说，**如果**该 595 行激活 task 已进一步绑定到 F22 所述 shared gate/up 输出，而且实际 gate/up 对 token 行保持一一对应，那么它与这次 643 行 MC2 输入无法作为两个**完整同形 tensor**直接共用一次量化结果：必须证明行映射、保留独有或屏蔽行及其 scale，并独立处理其他行。相同第一维的 960 组也只能通过形状检查，不能证明相同值、行顺序、mask、quant/scale 或 tensor 身份，更不能证明复用可行。此条件不排除二者存在 595 行子集共用，亦不排除两条任务完全无关。

源码中的明确 producer-consumer 链仍是 `DeepseekV41MLP.forward` 的 gate/up → activation → down（`sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py:144–148`），其第一投影结果必须先完成对应通道的 hidden 维归约才能激活；若中间采用逐 token 动态量化，down 的 scale 还受整行归约与舍入约束。该源码链与上述设备任务之间的具体调用、地址、视图、实际量化方法及每行顺序仍未绑定。MC2 内部本来就含 routing quant 和通信、矩阵计算，不应把它计作尚待首次融合；共享量化也须保留路由 mask、目标 expert 排序和每行 scale，以及两侧各自结束前的 buffer 可见性、生命周期；跨流重排可能破坏现有通信与计算重叠或造成 AIV/MTE/带宽争用，因此不能据此给收益或根因。

复现：在任务根目录执行 `python3 agents/fusion/check_f22_recapture_row_domains.py`，脚本只读 84 MB 既有聚合并写出该分析目录的 JSON；其中记录脚本与输入哈希。它不读取 raw trace、不重新扫描全部历史、不启动 NPU，也不修改目标实现。要把上述条件变成证据，需要已有或新授权采集中的该任务对应 stack/图节点、tensor/storage 和 row/scale/mask 映射；若不存在，保持 UNKNOWN。此节是 F22 同一复用候选的约束补证，**不新增独立融合项**。

## 现有目标层样本：F22 相关已匹配任务序列的跨 rank 复核

对既有 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` 作定向 CPU 汇总，而非新增 NPU 负载。每个已识别目标层 scope，按已唯一匹配到设备的任务、同一设备 stream 和导出开始时间排序，仅检查 `DequantSwigluQuant` 前后两个**已匹配记录**。16 rank 的 1280 个 scope 各出现一个该任务，且 1280/1280 在这个已匹配序列中呈现：`DynamicQuant` → `QuantBatchMatmulV3` → `DequantSwigluQuant` → `QuantBatchMatmulV3` → `ForeachAddListV2`；两个相邻矩阵乘与激活的物理首输出/首输入 shape 和 dtype 在 1280/1280 均相容。EP0、layer0、scope `736-51` 的示例：中间 INT32 `[595,4608]`，随后 INT8 `[595,2304]`，单个 stream 45 的五个 flow ID 与精确格式记录在 JSON 中。

这将之前只引用两层的**相容序列现象**扩大到此重采样内 1280 个 scope；它不证明这些任务属于 shared expert、共享同一 tensor、构成直接 producer/consumer，也不证明源码、实际设备二进制和数值语义已绑定。此处“邻接”只针对已匹配的 task：原提取的 96922 个 target selected start events 中有 39160 个 `csv_missing`，不能排除未匹配事件夹在这些记录之间；计数范围含所有选中的 task 类型，不能理解为仅该序列的丢失率。实际 `DequantSwigluQuant` 已作为一个设备任务导出，在 F22 讨论时不能重复把 dequant、SwiGLU、quant 当成三个已观测独立设备任务；其名字也不证明内部确切指令融合。把两侧矩阵计算纳入重组仍是条件候选，须先证明 tensor 边、全行 scale/舍入及消费者、跨流生命周期；现有采样不支持宣称可获得的时间收益或推断线上瓶颈。

复算：任务根目录 `python3 agents/fusion/check_f22_all_rank_chain.py`，CPU 只读原聚合并写 `agents/fusion/f22-all-rank-device-chain.json`。输入 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`；脚本 SHA-256 `cf3f459675105f97e9617f3493d3ca025666f04582f021962243d63c99ac1447`；输出 SHA-256 `b4f5fb5f8240303d7a75efe4efcebf29070fddcec58a60d73eb9fc507950d1c4`。rank footer 记录 16/16 完成且无 rank error，但它不消除上述任务匹配缺口和 tensor 身份缺口。


## 官网接口核查：F22 clamp 融合的排列约束（2026-10-29）

昇腾 [CANN 9.1 DequantSwigluQuant](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnDequantSwigluQuant.md) 的产品表对 Atlas A3 标为不支持，而 [同版本 DequantSwigluQuantV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnDequantSwigluQuantV2.md) 表中 A3 标为支持；V2 在 A2/A3 上还规定输入是二维 INT32/BF16、输出 INT8、activateDim=-1，groupIndexOptional 不能为 null。官网第一篇同时在约束段提到 A3，页面内部存在不一致；[9.2 beta2 接口总表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/aolapi/context/ops-nn/op_api_list.md) 的确定性列又列出 A2/A3。版本、API 与页面内部表述不能混为一谈，也不能根据这些资料否定本任务中已导出的同名设备任务。**设备 task 名不等于公共 API 版本**，本次历史 9.2 是否调用任何一个公开 API、实际内核及参数仍未绑定。

V2 文档的 mode 0 按最后轴两半做普通 SwiGLU，mode 1 则以偶数/奇数索引作为 gate/up，并使用非对称 clamp、gluAlpha、gluBias；本任务保存的 Ascend activation wrapper 先取前半作为 gate、后半作为 up，再分别 clamp 上界和双边（见本报告 stage229）。因此不能直接假定 mode 1 接受原 gate-up 排列后就与 wrapper 等价；mode 0 的文档公式又没有同一 clamp。对于理想实数激活示例 [1,2,3,4]、limit=10、alpha=1、bias=0，原连续两半输出 [silu(1)*3,silu(2)*4]；mode 1 的偶奇配对为 [silu(1)*2,silu(3)*4]，第一个分量已不同。这是排布反例，发生在量化前，不证明设备输出误差或线上错误，也不排除在满足完整契约时将投影、scale 等一起重排后融合。

后续分析 F22 的第一 GEMM → activation/quant → 第二 GEMM 时，须共同核对 gate/up 排列、weightScale/activationScale 同步重排、clamp 与量化边界、中间舍入、groupIndex 合同和其他消费者，再判断能否分块。形状与文档约束相容不足以证明相同 tensor、实际 API 被调用或性能改善。此条收紧了候选适用条件，不增加一个融合项、不改变现有运行。

## F22 同流选样的数据链与逐行量化屏障（离线复核）

以下重新读取既存 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl`（SHA-256 已在前节记录），只选 EP0、layer 0、scope `736-51`、stream 45 的五个已唯一匹配任务。时间采用该导出中的 `device_start_us` 和 `trace_duration_us`，不与主机时钟相减；“下一行”仅表示同流已匹配记录的先后顺序，不证明直接 tensor 边、同一模块归属，也不能排除未匹配任务。相对起点用十进制定点数计算：

| 已匹配记录 | 相对开始 / 持续（µs） | 导出的输入 → 输出（dtype、布局） |
|---|---:|---|
| DynamicQuant | 0 / 24.3595 | `[595,5120]` BF16 ND → `[595,5120]` INT8 ND、`[595]` FP32 ND |
| QuantBatchMatmulV3 | 24.400 / 77.5585 | `[595,5120]` INT8 ND、`[144,320,16,32]` INT8 FRACTAL_NZ、`[4608]` BF16 ND → `[595,4608]` INT32 ND |
| DequantSwigluQuant | 101.978 / 28.9195 | `[595,4608]` INT32 ND、`[4608]` FP32 ND、`[595]` FP32 ND → `[595,2304]` INT8 ND、`[595]` FP32 ND |
| QuantBatchMatmulV3 | 131.878 / 57.699 | `[595,2304]` INT8 ND、`[160,144,16,32]` INT8 FRACTAL_NZ、`[5120]` BF16 ND、`[595]` FP32 ND → `[595,5120]` BF16 ND |
| ForeachAddListV2 | 190.477 / 28.6195 | 两个 `[595,5120]` BF16 ND 和一个 FP32 标量输入 → `[595,5120]` BF16 ND |

该序列相容于“动态量化 → gate/up 投影 → 反量化/激活/量化 → down 投影 → 合并”的图；不过两个矩阵任务名称、形状与 scope 仍**不足以证明**上表相邻输出就是下条输入，也不能证明 `ForeachAddListV2` 的第二个输入来自 routed expert，或认定它是 shared 路径。五个任务时长合计 217.156 µs，首开始至末结束跨度 219.0965 µs；间隔只是该 stream 中导出记录的时间差，不能作为设备空闲或可节省时间，也不可推广到未受采集扰动的生产负载。格式、shape 是 CSV 对物理入参的标记，不自行反推权重逻辑转置或内存地址；`FLOAT` 对应导出类型，标作 FP32。首个矩阵输出 595×4608×4 = **10,967,040 字节**，激活输出主 INT8 张量 595×2304 = **1,370,880 字节**，每个 `[595]` FP32 scale 的逻辑大小是 **2,380 字节**；这些是形状乘类型大小的逻辑体积，不是实测 HBM 搬运量、可消除字节数或延迟收益。buffer 可能有复用、别的消费者及分块驻留。

**条件性融合约束：**若这组事件以后被证明确实构成上述同一计算链，且激活量化采用逐行 max-abs/127 并要求逐元素保持原量化结果，则一个 feature 块的最终 INT8 输出依赖该行所有 feature 的最大绝对值。反例：两行的第一个元素均为 1，余下元素均为 1 与“仅一个余下元素为 127”，两行该元素分别量化为 127 和 1。因此，在尚未看到行尾、又不允许回写或重算的单向流水中，不能先对该块发布最终量化值；需要等待完整行统计，或通过暂存、回写或重算维持语义。这**不要求一次物化整个 batch**，也不排除分行/分块、多遍或更一般的等价变换；由于历史任务的具体量化语义、数学精度及中间张量身份未绑定，此反例不是对现有硬件执行的实测断言。

最小增量证据是该选样的具体图节点/算子到 tensor storage 的绑定及其他消费者，实际量化 API/scale 语义与有效行对应关系；若这些信息不在已有工件中，仅凭更多同形样本不能证明相连。此节只收紧 F22 条件，不新增独立融合收益。

## 相同算子签名、不同物理行数：32 对定向时长比较

复用既存 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl`（SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`），只取 EP0–7、layer 0/12/24/36；在相同 EP 与 layer 内，对匿名 scheduler 摘要 `7ddf7ae98a00…` 与 `db852bff987d…` 分别找唯一匹配的 `aclnnDynamicQuantV2_DynamicQuant_DynamicQuant`（CSV Type 为 `DynamicQuant`）任务，得到 8×4=32 对，二者的导出 stream ID 在每对内也相同。两档**仅在第一维物理行数 T 不同**：输入 BF16 ND `[T,5120]`，输出 INT8 ND `[T,5120]` 与 FLOAT ND `[T]`；导出输入列表没有独立 scale 参数。源码动态量化一般把该 FLOAT 输出作为逐行 scale，但所选 64 个事件与调用实例、输出对象和 scale 数值尚未绑定，因此这里仅称其为第二输出，不从签名推断 scale 内容。脚本逐条检查完整输入/输出 shape、dtype、format、名字、Type 及各 scope 唯一匹配；这也不证明 trace 全部事件被采集。

物理行数之比为 `1023/595 ≈ 1.7193`。逐对的任务导出时长之比（1023 行 / 595 行）最小、中位、最大分别为 **1.030、1.242、1.388**；各自除以物理行数后再逐对相除为 **0.599、0.722、0.807**。例如 EP0/layer0 的两条记录分别为 **24.3595 µs** 和 **29.9395 µs**。这明确反驳对这 32 对记录的“物理行数增加多少倍，记录时长就必须增加同样倍数”的线性解释，且说明以单个常数每行耗时外推这两档不可靠。

这是不同批次、不同实际内容及可能不同流水干扰的重 profiler 数据，不是受控工作量实验。物理行数不一定等于有效 token；同设备不同时间的 stream ID 相同、同一算子名称和 shape 相容，仍不能证明同一 input storage、编译对象、tiling 或上下游 tensor 边。上述算术也不能单独解释为固定 launch 开销、带宽变化、每 token 线上吞吐、同一输入的可达收益或 F22 shared expert 真实身份。先前 F22 已分析该类 DynamicQuant 在所选已匹配序列中的相容位置，仍须绑定真实输入及消费者才能确认其模块归属。

CPU 复算脚本 [check_dynamicquant_crossshape_pairs.py](../diagnosis/check_dynamicquant_crossshape_pairs.py) SHA-256 `96a284105192a678d451c8727cf1a7fc6985859076300c404986174b8e53ee67`；保存全部 scope、flow、task 及签名的聚合 [dynamicquant-crossshape-pairs.json](../diagnosis/dynamicquant-crossshape-pairs.json) SHA-256 `0961cfc4ae0b12f7af70e8463b062f2abb3fdfd079a8df45bb2b9e79f7eb63e7`。只读既有导出，未执行新 NPU 负载或更改实现。

## 同采样对照扩展：四条相容前馈任务的不同形状响应

在上一节同一 32 对 `(EP, layer)` 和两组 scheduler 摘要范围，进一步对每个 scope 选各一条 dynamic quant、第一投影形状相容的 QuantBatchMatmulV3、DequantSwigluQuant、第二投影形状相容的 QuantBatchMatmulV3。64 个 scope 内，这四条选中任务都在同一 stream，按该顺序不重叠；每两档之间除行维 595→1023 外完整导出 shape、dtype、format、名称和 Type 一致。算子输出与下一任务输入的**张量身份仍未绑定**，上述顺序不证明实际连接，也不能把它们必然归为 shared expert。

| 选中任务 | 1023 行 / 595 行时长比最小值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| DynamicQuant | 1.030 | 1.242 | 1.388 |
| 第一投影相容 QuantBatchMatmulV3 | 1.076 | 1.369 | 1.597 |
| DequantSwigluQuant | 1.371 | 1.422 | 1.463 |
| 第二投影相容 QuantBatchMatmulV3 | 1.131 | 1.293 | 1.497 |

这说明在已选导出记录里，随物理行数变动的时间比例因任务不同而不同，不能用单个按行线性因子估整条相容序列。表中各中位数可能来自不同对，不能直接加减或解释为总耗时；不同批次的有效行、输入、tiling、竞争和采样扰动未知，不能推断瓶颈原因、线上吞吐或融合收益。

复算 `python3 agents/diagnosis/check_ffn_crossshape_chain.py`：校验固定输入 SHA、各 scope 唯一选择、完整签名归一化、单 scope 同流顺序后保存配对。脚本 SHA-256 `523cbb24f71db19e1dd14dd43037a8e185dcb26118afa5ee39f6e3f62d691aab`，聚合 `agents/diagnosis/ffn-crossshape-chain-pairs.json` SHA-256 `036f74db28a55a1329ab371bb3c65b6c2c2bebb90a97db76e2be93dd64e6b9ce`。仅离线 CPU 处理，不启动 NPU。

### 同 scope 两条投影的时长与逻辑 MAC 对照

使用上一节同一份 64 scope 数据，在每个 scope 内比较第一、第二投影的记录时长。第一投影的形状为 `[T,5120] → [T,4608]`，第二为 `[T,2304] → [T,5120]`；**若分别按相应标准稠密矩阵乘计算**，两者逻辑 MAC 数之比为 `5120×4608 / (2304×5120) = 2`。按记录时长第一除以第二，32 个 595 行样本的最小/中位/最大为 **1.148 / 1.316 / 1.403**；32 个 1023 行样本为 **1.190 / 1.332 / 1.472**，64/64 均小于 2。

这排除了所选记录中“这两条任务的时长严格按上述逻辑 MAC 数同比例变化”的简单描述。两者并非相同工作负载、也非相同输出 dtype：第一输出 INT32，第二输出 BF16。上述 MAC 数是直接稠密乘法的逻辑工作量，不是所有等价实现必须进行的计算量。两个任务的矩阵形状、量化参数和实际执行路径可能不同，其他任务内容、有效输入、资源状态和 profiler 扰动也未控制；不能将该差异归因于访存、计算利用率或固定开销，也不能由此估计融合收益。仅依据同 scope、同流与形状兼容，仍不能证明两条投影共享 tensor 或属于同一模块；本对照不与端到端时长混算。

在运行根目录下只读复算（输入固定为上一节的聚合文件）：

```sh
python3 - <<'PY'
import hashlib, json, statistics
from decimal import Decimal
p = 'agents/diagnosis/ffn-crossshape-chain-pairs.json'
data = open(p, 'rb').read()
assert hashlib.sha256(data).hexdigest() == '036f74db28a55a1329ab371bb3c65b6c2c2bebb90a97db76e2be93dd64e6b9ce'
rows = json.loads(data)['pairs']
assert len(rows) == 32 and len({(r['ep'], r['layer']) for r in rows}) == 32
for shape in ('595', '1023'):
    ratios = [Decimal(r[shape]['projection_1']['duration_us']) / Decimal(r[shape]['projection_2']['duration_us']) for r in rows]
    print(shape, len(ratios), min(ratios), statistics.median(ratios), max(ratios), sum(x < 2 for x in ratios))
PY
```

也可执行 `python3 agents/diagnosis/recheck_ffn_projection_pair_ratio.py`；脚本 SHA-256 `15492f9c388283e0a35a1b73ac881327c513d19751c99c68baac684ef236ab60`，核验输入哈希、完整 EP/层集合、同流顺序、形状和正时长。脚本只读本机已保存数据。

## EP0 单层样本：从主机调用栈绑定 MC2 与 shared-experts 调用（只读复核）

对上文 EP0、layer 0、scope `736-51` 的同一次重 profiler 采样，直接只读既存 EP0 SQLite，按原始键核查六条主机 API → `CONNECTION_IDS` flow → 命名设备 task。scope 唯一，六个 API 都属于同一主机线程 `globalTid=3161095930592`，并完整位于该 scope 的主机时间区间内；每个主机 connection 对应唯一 API 和唯一 flow，各表中被选命名任务的 `globalTaskId`、`COMPUTE_TASK_INFO` 均唯一，设备 stream 均为 45。其余已保存的 flow 记录中，MC2 和第二矩阵乘各有一条未命名 TASK，不能将其当成额外一次算子调用。下表的开始时间是设备原始纳秒整数，以免 epoch 浮点微秒精度损失；主机区间只用于确认 scope 包含关系，不用主机与设备时间直接相减。

| 主机 API；已记录调用栈关键帧 | 主机 connection → flow → 命名设备 globalTaskId / taskId | 命名设备起点 ns |
|---|---|---:|
| `aclnnDispatchFFNCombine`；`moe_comm_method.py:591 fused_experts`，`w8a8_dynamic.py:218 apply`，`routed_experts.py:740 forward_impl` | `496477 → 120151 → 4722 / 45443` | 1793241333374667046 |
| `aclnnDynamicQuantV2`；`shared_experts.py:429 _prepare_gate_up_input` | `496478 → 120161 → 4723 / 45444` | 1793241333401140997 |
| `aclnnQuantMatmulWeightNz`；`shared_experts.py:475 enqueue_gate_up_after_router` | `496479 → 120166 → 4724 / 45445` | 1793241333401165397 |
| `aclnnDequantSwigluQuantV2`；`shared_experts.py:534 _run_a8_int_mlp` | `496480 → 120171 → 4725 / 45446` | 1793241333401242975 |
| `aclnnQuantMatmulWeightNz`；`shared_experts.py:557 _run_a8_int_mlp` | `496481 → 120180 → 4726 / 45447` | 1793241333401272875 |
| `aclnnForeachAddListV2`；`fused_moe.py:240 forward` | `496482 → 120188 → 4727 / 45448` | 1793241333401331474 |

因此，这**单个样本**的四条 API 已有直接记录的 `shared_experts.py` 调用栈支持，MC2 则有 `routed_experts.py` 与 `moe_comm_method.py` 调用栈支持；这比上文仅凭邻接与 shape 提出的角色猜想强。既存冻结源码 `agents/fusion/frozen-source/vllm_ascend/ops/fused_moe/shared_experts.py`（SHA-256 `471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9`）相应行展示了 A8 INT 动态量化、gate/up、激活/量化和 down 的实现，`fused_moe.py` 冻结参考（SHA-256 `2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4`）展示 shared 与 routed 输出合并的源码逻辑；但历史进程加载的源码版本 hash、设备二进制、张量对象和 storage 均未绑定。调用栈中的文件名、函数及行号与冻结源码相符，是强的调用角色证据；依照冻结源码把五个实际张量连成 producer→consumer 链、认定 add 的两个实际输入来源，仍是**有版本前提的解释**，不构成已闭合的张量数据流。此升级仅适用于上述六个实例，不能推广为其余样本均已完成角色绑定。

结合既存设备导出，该 MC2 的第一个输入为 BF16 ND `[643,5120]`，而同 scope 内上表 shared-experts 动态量化的输入为 BF16 ND `[595,5120]`。两种导出完整形状不同，因此排除“两个完整输入等形、无切片或映射地直接视为同一个输入”的这一个复用方案。它**不排除**底层 storage 有重叠、存在切片、填充、重排或其他行映射；shape 也不能证明其实际有效行、量化规则、值、scale 相同或不同。能否共享量化及其真实收益仍未知，须先获得这些实际对象和语义证据。这里既不把 48 行差当作有效 token 差，也不因此声称两个路径不能融合。

复核脚本 [query_f22_ep0_host_binding.py](../diagnosis/query_f22_ep0_host_binding.py) SHA-256 `9474f552ce12efaf8251941650ab16e30fc2e7e90eeba242b3a735c1c703254b`，结果 [f22-ep0-host-binding.json](../diagnosis/f22-ep0-host-binding.json) SHA-256 `3d8f0365efe5662c0e285558ffb03511a04216ffda313cc57c64621abc7218a9`。脚本在专用 stage 容器内对保存的 EP0 profiler 数据库以 SQLite `mode=ro` 执行、只读扫描和计算其 SHA-256 `a1cf18e420823e625e6f38c42f9f16dd816500bccf9a18346c815aaa9b4f6edc`，核验读取前后文件状态一致；输入 shape 来自上文有 hash 的已存 target-layer 导出。未运行 NPU、未修改目标实现。
## F22 条件性整数累加：分 K 后是否必须保持原加法顺序

对已选第一、第二投影的逻辑候选 K 分别为 5120、2304，分析以下**限定的数学契约**：两个输入均为有符号 INT8，单个输出为这些整数乘积之和，累加为精确 INT32，零初始值；在 dot product 完成之前不插入缩放、零点补偿、量化或其他舍入，输出仅要求与该精确整数值相同。此契约是条件模型，不是历史运行的已验证 API 语义。昇腾 [CANN 8.5 Mmad 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0249.html)在产品及数据类型表中列出 A3 的 INT8 × INT8 → INT32 组合，但它不证明该历史任务采用何种 Mmad、参数及累加实现。

单项乘积绝对值不超过 `128×128=16384`。对任意分 K、任意不重复不遗漏的子集，每个部分和的绝对值不超过 `K×16384`：K=5120 时为 **83,886,080**，K=2304 时为 **37,748,736**，都小于 INT32 最大值 2,147,483,647。因而只要每个部分和及其合并都在上述整数契约内计算，任意分组和加法树都不会溢出，结果与原精确 dot product 相同；即使硬件整数计算在溢出时饱和，这一范围内也不会因加法顺序触发饱和。这个数学事实允许研究 K 维拆分和合并，不要求保留原整数加法树。

适用范围不能扩张为“F22 可任意融合”：实际输入可能有零点补偿或偏置、不同宽度的中间计算，以及中途反量化、动态逐行 scale、激活、BF16 舍入或其他消费者。特别是第一投影输出经 `DequantSwigluQuant`，最终量化依赖完整行统计的条件限制不因整数加法结合律而消失。历史 host 参数、权重与 scale 的具体绑定、kernel/指令和是否执行额外处理仍未知；必须先验证其满足上述契约才能把这一证明用于实例。

CPU 算术复算：`python3 -c 'print(5120*128*128, 2304*128*128, 2**31-1)'`。只读既存材料，不运行 NPU，也不修改推理实现。

**接口适用性补充：**[CANN 8.5 的 `aclnnQuantMatmulWeightNz` 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/context/ops-nn/aclnnQuantMatmulWeightNz.md)把 INT8×INT8、空 `x1Scale`、BF16/FP32 `x2Scale`、INT32 输出列为 A3 支持组合；同页一般公式的无 bias 情况还包含权重 scale 和可选 offset。文档的组合表和一般公式，不能直接替代该特定 INT32 输出分支的实现代码与舍入规则。已记录设备签名的 BF16 向量与 scale 形状相容，并不证明其实际张量身份或被忽略。即使先前已证明整数部分和安全，调用级重组也仍须确认 scale 是在完整点积前、之后还是被特殊处理，及其精度、舍入与输出语义；如果完整点积先形成后再处理相同 scale，那么保留该处理并重组整数求和树是条件上可研究的。该 8.5 文档不能单独绑定历史 9.2 的调用参数、行为或二进制。

例如**假设**该通用公式适用于某 INT32 输出分支，点积是 4、BF16 scale 是可精确表示的 0.5、无偏置和 offset，那么按公式得到 2，与原始点积 4 不同；结果本身可精确表示为 INT32，不需假设小数舍入规则。这仅是公式条件下的算术示例，并不证明该分支实际处理 scale 的方式，更不是历史输出值。

## 643/595 行差异：一组可排除的分片解释

针对前述 EP0/layer0 单个样本，复核已存 `shared_experts.py` SHA-256 `471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9`：`:229–246,363–388,425–429` 在 `SHARED_EXPERT_DATA_PARALLEL_ONLY` 路径先把 shared 输入行数 `T` 补齐到实际 TP group 大小 `P` 的倍数，再均分并将本 rank 行送给 dynamic quant；因此该路径的物理量化行数是 `ceil(T/P)`。下游 `:248–256,711–713` 还要将本 rank 结果按第 0 维 all-gather 并按原始行数裁剪，不能把各 rank 局部输出直接当成完整 shared 输出消费。实际前后流可见性和通信仍需运行路径绑定。

**条件性排除**：若这份源码确为样本运行版本、shared 动态量化的输入正由上述分支产生、该分支原始完整输入恰为 MC2 记录的 643 行，且实际 `P=8`，则量化记录应为 `ceil(643/8)=81` 行；这与样本中观察的 595 行不一致。这只排除了四个条件同时成立的解释，不证明其中哪一项不成立，也不证明 shared 和 MC2 的输入无别名。反过来，若同一路径确实输出 595 行且 `P=8`，其分片前 `T` 必在 **4753–4760**，不能单凭样本的 MC2 643 行确定 `T`。

对这个被排除的条件组合，若补齐行放在尾部且连续均分，则 643 行补齐到 648，8 片各有 81 个物理行；第 0–6 片各含 81 个原行，第 7 片含 76 个原行及 5 个补齐行。该分布仅用于检查上述假设下的行范围，不证明真实采用该分支、补齐数据的具体值或实际有效行。ZCode 的离线算术回答与主控逐片复算一致；其确切模型标识未独立绑定。

作为另一种纯源码条件，`SEQUENCE_PARALLEL_ONLY` 的已 gather 输入在 `:383–387` 按 `_EXTRA_CTX.num_tokens` 切片；若它的切片前 tensor 恰有 643 行且该值恰为 595，则这一分支与观察到的 595 行相容，但仍不能确定该模式、输入身份、行值或实际 48 行为何被排除。比较形状不能替代存储、事件、有效行和路径证据。该收紧用于审查跨 routed/shared 量化复用，并不是新的独立融合收益或实际瓶颈结论；本次只读既有代码及记录，不运行 NPU。

## 两档 scheduler 数值与 shared 行数的条件性关联

离线复核 `diagnostic-20261029/capture-summary.json`（SHA-256 `d3862f7c3ff92628aaefd4d39bda62352fc557a4f6c4fc2a48c24ca8318746e8`）和已存 target-layer 导出（SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`）。前者 scheduler pid 为 690；后者 EP0 worker pid 为 736。按匿名 digest 对照，两组资料给出：

| digest 前缀 | scheduler 记录总数 | EP0 第 0 层 scope | shared 调用栈已支持的 DynamicQuant 物理输入 | MC2 物理输入 | `ceil(调度总数/8)` |
|---|---:|---|---|---|---:|
| `7ddf7ae98a00` | 4760 | `736-51` | `[595,5120]` BF16 | `[643,5120]` BF16 | 595 |
| `db852bff987d` | 8180 | `736-6` | `[1023,5120]` BF16（角色依上文 shape/次序选取，尚未在该 scope 单独绑定调用栈） | `[1023,5120]` BF16 | 1023 |

每条 scheduler 记录中条目的 `scheduled_tokens` 之和分别等于表中总数。相同 digest 是已存采集的关联键，不是 tensor 地址或跨进程实际张量对象证明。两档都与“完整输入行数等于 scheduler 总数并按 8 路向上补齐后均分”的算术相容；但启动请求 TP8 不证明该实例实际 TP group、行数计算、resolved SP 和 shared DP、行顺序或实际有效行。SP+shared-DP 可直接使用预先分出的 shard；无 SP+shared-DP 则可在 shared 内部补齐并分片；两种路径不能仅由这个数字区分。`736-51` 同 scope 另有一个输入输出均为 `[4760,5120]` BF16 的 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove` 设备记录，`736-6` 则有 `[8184,5120]` 同类型记录；它们的 shape 分别与上述未补齐总数、8 路补齐后总数数值相容，但任务名称、相同 scope 和形状不能证明它们是 shared 的实际前序输入、也不能绑定 storage 或方向。

进一步只作**可检验的源码条件示例**：假设真实执行了 `prepare_finalize.py:274–284` 的 `replace_allreduce` 分支，routed 在补齐前输入与 shared 所用的 595 行为同一个 shard，真实 TP size 为 8，且 `_EXTRA_CTX.padded_num_tokens=5144`，则 `5144 // 8 - 595 = 48`，routed 输入会由 595 行尾部补到 643 行；该补齐分支还给追加的 mask 位写 `False`。`shared_experts.py:208–227,363–388` 的 SP+shared-DP 路径本身不会在该处对 shared 输入做此 routed 补齐。这说明已有源码提供一个可解释 595/643 数值关系的具体机制，并非证明它发生。现存另一 EP9 scheduler 摘要中的 5144 来自另一摘要及进程，没有证明与 EP0 的这个 scope 同时属于同一 DP 批次，也没有绑定 `_EXTRA_CTX.padded_num_tokens`；所以不能把 5144、48 行 padding、mask 值或上述条件路径当成该样本的实测事实。要确认必须获得目标实例的 resolved mode/TP group、目标时刻 forward context 的实际值和两支输入的 tensor/storage 关系。无新增设备采集。


## 对样本身份和 token 组成的限定复核

既存 `diagnostic-20261029/capture-summary.json` 的 profiler 路径标签分别标识 EP0 采样 worker 为 `dp0/tp0/ep0`、pid 736，另一个 EP9 为 `dp1/tp1/ep9`、pid 752。这些是采集文件的进程和 rank 标签，不能单凭标签证明相关记录在同一次跨 DP 同步中配对，或证明某个张量平行组的完整成员及运行时大小。**因此另一摘要的 5144 不能充作 EP0 scope 736-51 的实际 `padded_num_tokens`。**

对 scope 736-51 关联的摘要 `7ddf7ae98a00`，scheduler 汇总 4760 来自四项 **3200、1548、6、6**。前两项记录的已计算 token 分别为 29568 和 31220；后两项的 cached computed tokens 均为 32768，且两项的 `spec_token_count` 字段均记录为 5；前两项的该字段是 null，不能当作 0，也不能由字段数量确定设备有效行或实际阶段。这一摘要同时包含不同请求状态，不能把这 595 个量化物理行均当作单一 decode 或 prefill 的工作；也不能仅从全局条目顺序或 595 = 4760/8 判断每张卡上各类行的实际映射。比较的另一摘要 `db852bff987d` 的 8180 来自三项 **3200、3328、1652**。这些是保存的 scheduler 条目和两份不同摘要的数值，不是逐行设备 trace，也不证明调度条目与设备 tensor 的逐行身份。

本机 ZCode 只读复核了两份摘要的条目加和与向上整除结果；主控核验源文件 SHA、摘要唯一性并独立得到 4760→595、8180→1023。ZCode 的确切模型标识未独立确认；这项算术复核不能代替实际模式或张量绑定。

本次有限复核范围包括 F22 既有引用的启动与源码结论、`capture-summary.json` 的 10 条保存的 scheduler 样本和 16 个 profiler 身份标签，以及既有 EP0 单层调用栈和 shape 导出结论；上述资料中未获得目标 scope 的 `_EXTRA_CTX.padded_num_tokens` 值、resolved SP/shared-DP 与 replace_allreduce、实例实际 TP group、输入 storage 及有效行映射。它们继续分别为未知，不因没有找到就否定该 padding 机制。要闭合 595 → 643 解释，需同一 scope/进程的 forward context 值、resolved 分支与 group，以及两条输入的行映射或 storage 证据；若原始记录未保存，这些缺口不能通过重复比较相同 shape 补足。无新增设备采集。

## gate/up 输出的具体分块配对边界

本节细化前述“同一激活通道需要 gate 与 up 都就绪”的条件，不是新增已融合算子或可独立相加的收益。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py:104–146` 的 shared MLP 使用组合 gate/up 投影，模型配置 `moe_intermediate_size=2304`。已存 `agents/fusion/frozen-source/vllm_ascend/ops/activation.py:41–48`（SHA-256 `79185cd8bae9801aef50873cb082bb977ff8d2b3ddb27766e4f42ffbfa12b5c0`）的 Ascend `SiluAndMulWithClamp.forward_oot` 将末维前半作为 gate、后半作为 up：gate 只限制上界，up 同时限制上下界，然后组合并调用 `torch_npu.npu_swiglu`。`frozen-source/vllm_ascend/ops/fused_moe/shared_experts.py`（SHA-256 `471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9`）另包含融合与非融合路径，不可将一条 Python wrapper 的算术和运行时对象直接套用于所有 fused 路径。

既有 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl`（SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`）的 EP0 L0 scope `736-51`，有 INT32 `[595,4608]` 输入、INT8 `[595,2304]` 输出的 `DequantSwigluQuant`，以及相容的后续 INT8 `[595,2304]` 投影；完整 dtype、scale 输入、格式、相对时间与 host stack 的已有限定见本文件上文。这些签名与不分片或等效生成完整 gate/up 的路径相容，**不能仅凭 shape/同 scope 证明它们属于 shared MLP、矩阵输出与激活输入是同一张量、激活内部实现采用上述 wrapper、runtime 精确 clamp 或每个专家的实际 TP 分片**。本节只讨论实际调用经证明满足该 half-split 合同及各半宽为 2304 时的整数映射。

在该合同下，对于逻辑激活通道 `j ∈ [0,2304)`，必须得到完整投影结果 `gate[j]` 和 `up[j]`，它们分别位于组合输出的 `j` 和 `j+2304`。不同分块大小决定两部分能否按完整 tile 配对：

- 如将 4608 维组合输出按 256 通道、从 0 对齐分块，`2304=9×256`；每个 gate tile `r=0..8` 的对应 up 恰是 tile `r+9`。只在这两块对应 K 归约、各自反量化参数和中间精度就绪后，才可形成对应激活块。
- 如按 512 通道从 0 对齐，`2304=4×512+256`。gate 第一块 `[0,512)` 的 up 为 `[2304,2816)`，横跨组合输出 tile 4 的 `[2304,2560)` 和 tile 5 的 `[2560,2816)`。因此以一个固定**整数** tile 偏移将两个 512 块配成一对会漏掉部分通道或读取错误位置；需要处理跨 tile 拼接、调整分块或证明另一等价映射。

以下 CPU 代码复算对应输出位置，不查询 kernel 或设备：

```python
D = 2304
for r in range(D // 256):
    assert (r*256 + D)//256 == r + 9
    assert ((r+1)*256 - 1 + D)//256 == r + 9
assert D // 512 == 4
assert (D + 511) // 512 == 5
```

这只约束上述假定 tile 的逻辑输出通道；既存记录中的 `FRACTAL_NZ` 物理 weight shape 不单凭这些数就证明字节布局、logical N 的具体维度、设备实际 tile 是 256/512、任一 tile 可在 kernel 结束前对外发布或融合能够加速。每个 gate/up 完整投影通道仍依赖其完整 K 贡献，并须保留分别对应的 weight scale、量化反量化顺序、中间 dtype/舍入、activation clamp、不同分片/尾块和缓冲生命周期。若激活后量化确采用整 token 行的动态 scale，某块 gate/up 就绪也不足以发布最终 INT8 数据：还需要完整量化域的统计，或经证明等价的暂存、回写或重算；此约束与本文件此前的量化域反例是同一依赖，不能重复计收益。实际 fused kernel 归约域、scale 语义、已编译布局、ready 事件、数值验收及历史对象均未绑定。本次只读已有资料和 CPU 算术，不访问 NPU 或改目标实现。
