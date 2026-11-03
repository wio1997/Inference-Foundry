# F02：attention 输入的 HC 混合、RMSNorm 与系数支路

## 边界与新增范围

先前 `HOTSPOT-FUSION-REVIEW.md` 对 F02 记录了 `hc_pre → input_layernorm` 的单消费者结构与条件物化；`F03-HC-SOURCE-CONTRACT-AND-TILING.md` 已为 **FFN pre** 证明外部 pre_mix 下混合值不需要等待当前系数的数学依赖。本记录将该源码契约具体映射到另一条、先前未展开的 **attention pre → norm → attention / attention post** 链，并列出两个不同下游系数的最晚需要点。不是声称这是新发现的独立 HC 原语，也不能与 F03 的候选把同一个计算节省重复累计。

## 源码中的真实数据和最后等待点

`model.py:883–919` 以层入口 hidden 为 residual，执行 `hc_pre(hidden, hc_attn_fn, hc_attn_scale, hc_attn_base, pre_mix)`，得到混合值 y、`attn_post`、`attn_comb`、`attn_pre`。随后：

- y → `input_layernorm` → 如 SP=true 先 `sp_all_gather` → self-attention（包含 Q/KV 投影、缓存写与注意力）；attention 输出如 SP=true 先 reduce-scatter。
- `attn_post`、`attn_comb` 与原 residual 必须在 attention 的 `hc_post` 消费该 tile 前就绪；`attn_pre` 必须在**后续 FFN pre**使用它作外部 pre_mix 前就绪。不能因为 attention 前只用 y 就丢掉这三个结果，也不能回收仍将被 HC post 消费的 residual。
- 首层前 `model.py:1557–1559` 初始化 FP32 `pre_mix=[1,0,0,0]`；之后每层由上一层返回的 `ffn_pre` 传入，条件 Engram 会在层调用前原地更新 residual（`1570–1592`）。因此 y 依赖当层实际输入 residual **和已经就绪的前一 mix**；不能在 Engram 修改尚未完成时读旧 residual，或因首层 mix 特殊就将所有层视为相同。

冻结 A3 `hc_pre` 测试参考及实现摘录证明：**外部 pre_mix 存在时**，逐 token/feature 的 `y = cast_BF16(sum_h(pre_mix[h] * residual[h,d]))` 的数值输入不包括本次 `hc_attn_fn`、scale、base 运算结果；这些权重仍决定当前 `attn_post/comb/pre`。详细的源码路径及 25 文件双容器哈希证据在 `F03-HC-SOURCE-CONTRACT-AND-TILING.md`。无外部 pre_mix 时不适用此结论。相应参考测试只有数值容差，kernel 指令顺序与历史二进制/输入绑定尚未知。

## 有条件的拆分及必要同步

若该分支及前述依赖确实满足，可以研究让 **y 路**在 residual 与 prior pre_mix 块 ready 后单独计算，并随后做 norm、必要的 SP gather 和 attention；**系数路**同时从同一个 residual 完成当前投影、归约与 Sinkhorn，分别在 attention post 和 FFN pre 前 join。这个拆分可以把当前 pre 的部分计算放到 attention 工作期间，但它不是已实现行为：现有 A3 源码 pre 内部有 `SyncAll`，Python 接口返回全部输出后才调用 norm；没有暴露当前系数/混合值逐块 ready 的接口，可能需要新实现。两个路径同读 residual、可能竞争 Cube、Vector、片上存储及 HBM；数学无依赖不意味着可以无代价同时跑。

第二个候选是将 y 的混合与 input RMSNorm 合并：可按 hidden feature tile 产生 BF16 y 并累计整行平方统计，但最终规范化每个分量需要完整的该行 5120 维归约、gamma 和 epsilon。在同一 kernel 里要保留原 BF16 y 中间舍入后再规范化的数值边界，所有 y 数据直到 norm 消费完都要可得；不足以假设单个 feature tile 可立刻产生最终 norm，或该行一定能留在某级片上。SP 分支仍需正确 all-gather 和后续 reduce-scatter；cache update、query 为空但 KV 非空、图捕获与通信顺序也不能跳过。

若两条候选组合，y 流与系数流在 residual/pre_mix ready 后才可独立，attention post 必须等待 attention 输出、原 residual、`attn_post/comb` 和其所需的跨 rank 贡献，FFN pre 必须等待新的 residual 和 `attn_pre`。对应 stream/event、缓冲 alias、两个消费者的存活期、原 HC FP32 归约顺序/输出 BF16 舍入、norm 的实际算子精度与 eps、zero/NaN 情况及编译 lowering 均须证明。源代码未显示跨 kernel 的部分数据可见性，不能将硬件 barrier 直接当作数学必要的全 token barrier，也不能无条件删除现有 barrier。

## 已有样本能支持的范围

stage122 的 `f03-hc-shape-correlations.json` 对 16 rank × 3 轮 × 40 层按顺序归属的 attention pre 均记录 HcPre 输入含 FP32 `[3,4]` 第五输入、输出 BF16 `[3,5120]` 及三组 FP32 系数；与外部 pre_mix 的接口形状相容，但没有第五输入对象地址、值或当前 binary 的绑定。

具体 EP8 层0 run2：`f03-hc-task-metadata.json` 的 attention pre task26 在窗口起点后 `0–33.900 µs`，同流89 `f07-q-kv-streams.json` 的 RmsNorm task27 在 `35.080–39.860 µs`，输入 BF16 `[3,5120]`；后续 MatMulV2 task31 在 `62.859–80.739 µs`，输入 `[3,5120]`。名称、shape 和同流顺序与源码接口一致，但不能确定同一 tensor、实际选择的 pre_mix/接口、binary、请求阶段或已有 fused lowering；gap 也不等于可节省时间。对完整系统和 prefill 不能以该一窗推广。

未改实现、未启动 NPU。若没有现有 tensor ID/事件及实际实例证据，重组的正确性、可交叠程度和性能影响继续标 UNKNOWN。

## 重采样 prefill 中的跨模块边界：同层提交关联，不等于 tensor 边

离线读取已有 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl`（SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`），定向取 EP0、layer0、提交范围 `736-51`；其中两个不同 host flow 各有一条已唯一匹配设备任务，均在 stream 45。数字是设备 trace 的时间，不与 host 时间相减：

| 已匹配任务 | 相对第一任务开始 / 耗时 | 导出输入 → 输出，格式均为 ND |
|---|---|---|
| HcPre，flow `1793241333346342552` | 0 / 155.577 µs | BF16 `[595,4,5120]` 与 FP32 `[24,20480]`、`[3]`、`[24]`、`[595,4]` → BF16 `[595,5120]` 与 FP32 `[595,4]`、`[595,4,4]`、`[595,4]` |
| RmsNorm，flow `1793241333346498829` | 156.277 / 27.4595 µs | BF16 `[595,5120]`、`[5120]` → BF16 `[595,5120]`、FP32 `[595,1]` |

这是预填充重采集的**层提交范围、同流顺序和 shape/dtype 相容证据**，比只拿不同采集的同形记录拼接更有约束；它仍不是 tensor 地址或同一 storage 的证据，也不直接证明 HcPre 这个 task 对应 `model.py` 的 attention pre、对应特定 pre_mix 对象，或该 `RmsNorm` 消费了上条输出。所选 scope 的提取包含无法匹配 CSV 的 start event，不以任务 ID 连号宣称不存在未记录工作。两个导出时长之和 183.0365 µs，首开始至第二结束为 183.7365 µs、间隔 0.700 µs；不能将后者解释为实际 idle、排队开销、关键路径或融合收益。采集属于重 profiler 配置，不能等同未插桩线上时延。

假如之后绑定了同一数据边，候选输入 BF16 结果的逻辑大小为 **595 × 5120 × 2 = 6,092,800 字节**；三个系数输出逻辑合计 **57,120 字节**，但这些形状大小不是实际 HBM 流量，也不代表可全消除。源码链对照的关键限制是：结果 x 供 `input_layernorm`；post/comb 由 attention 结束后的 `hc_post` 使用；pre 由其后的 FFN `hc_pre` 使用；原 residual 跨 attention 仍需被 `hc_post` 消费。分拆 y 与系数支路时，必须保存这些输出及多消费者生命周期，不能把 `HcPre` 当作只产生归一化输入。若 `use_sequence_parallel` 实际为真，norm 后还有 gather，attention 输出后还有 reduce-scatter；当前样本并未单独证明 SP 最终生效或对应通信的张量绑定。

把此观察升格为真实边的最低证据是相同采集的图节点/CPU 调用关联和 output/input storage 或等价可追溯别名，外加实际 pre_mix 分支及其生命周期；除非已有工件提供，不能从 shape 和时间相邻补出这个关联。若未来确认为该边，feature 分块计算也必须保留 BF16 y 的原有舍入、整行 RMS 统计与三个系数后续消费者；可分行准备，不得把整 batch 一律当成不可避免屏障。

## 目标首层重复输入的 HC 系数投影：条件性结构重组

本节研究 embedding 输出与 **目标首层 attention HC 的系数投影**边界，区别于本文件已分析的 y 提前发布；不能把两者视作互不重叠的收益。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py` SHA-256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91` 的 `1519,1541–1553` 得到 embedding 或输入 `inputs_embeds`，可选 SP 分片，再将同一行 `repeat` 到四路，并设置首层 `pre_mix=[1,0,0,0]`；`:821–829,889–895` 使用 FP32 `hc_attn_fn`，形状 `[24,20480]`。`evidence/model-config.json` SHA-256 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7` 的 `text_config` 确定 `hidden_size=5120`、`hc_mult=4`、Engram 配置层为 1 和 14。只有当目标计算确实执行此源码、对应 target layer 0 是模型首个局部 decoder、该行在进入其 attention HC 前未被其他条件路径修改时，才能将初始四路视作相同。SP 分片后再复制不改变这个源码条件；实际 PP 分支、运行时对象与张量身份仍未证明。

在**有限实数**、不含舍入的参考模型中，将系数矩阵按 HC 维切为 `W[j,h,d]`，而输入 `x[h,d]=e[d]`，有 `sum_(h,d) W[j,h,d] e[d] = sum_d (sum_h W[j,h,d]) e[d]`。若在权重生命周期内另行预计算 `Wsum[j,d]=sum_h W[j,h,d]`，可以研究由一条 embedding 行直接计算所有 24 个系数投影；同样，在实数模型下原始 20480 维平方均值等于单路 5120 维平方均值。按照固定维度，标准 dense 逻辑投影从每 token `24×20480 = 491520` MAC 改为 `24×5120 = 122880` MAC；系数权重表示从 491520 元素到 122880 元素。这不计算权重预处理、版本更新与额外驻留，也不是实际物理 FLOP、HBM 流量或速度保证。除了 y，HC pre 的 post、comb、pre 都仍需正确生成；原 residual 的四路逻辑值仍供后续 HC post 消费。归一化 epsilon、sinkhorn、各输出 dtype、premix、collective、模型并行和 buffer/event 生命周期不能丢失。

有限实数恒等式不保证现有设备结果逐位相同。`F03-HC-SOURCE-CONTRACT-AND-TILING.md` 引用的测试参考将 matmul 转换为 HF32，A3 源码还沿 K 分块并同步，故必须核对实际编译 kernel 和误差合同；直接预先求和还改变了舍入和 HF32 转换的位置。以下 **CPU NumPy、逐步 FP32 舍入的构造性反例**只展示代数重排不自动 bit-exact，不模拟 Ascend kernel、不声称最终 gate 或 BF16 输出必然不同。令所有其余维度均为 0，而同一 embedding 元素取 BF16 可表示值 5.40625：

```python
import numpy as np
f = np.float32
e = f(5.40625)
w = list(map(f, [1.6406420469284058, -7.152994632720947,
                 -2.9253642559051514, -3.905107021331787]))
a = f(0)
s = f(0)
for x in w:
    a = f(a + f(e * x))
    s = f(s + x)
print(float(a), float(f(e * s)))
# -66.7283935546875 -66.72838592529297
```

既有采样 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`，在 16 EP 的 2 个 L0 scope 共 32 个已导出 scope 中，每个恰有两条 `HcPre`；输入类型和 `[T,4,5120];[24,20480]` 形状相容，T=595、643 的 scope 各 8 个，T=1023 的 16 个。首条的时间位次与 attention HC 相容，不能仅凭位置将它绑定到 `hc_attn_fn`，也不能证明真实输入四路的数值、stride、权重实例或新方案可行。提取器自身存在未匹配记录，因此 32/32 只覆盖保存的 scope，不证明完整原 trace。调用栈、storage、真实执行分支、weight 版本、数值接受范围及首层实际时间和可实现提速均为未知。本次只读既有材料、做 CPU 计算，未操作 NPU 或修改目标推理实现。

### 首层证据的进一步边界

已保存的 `diagnostic-20261029/sidecar-hierarchy-20261029.json`（SHA-256 `e4ffaf3cacaeebddb6beb249d9f48d76a3800dc5f02528cdc2ae46e6c3a8a023`）把 EP0 的 `736-51` 标为 layer 0 的 `conditional_target`。同 scope 的 `agents/diagnosis/f22-ep0-host-binding.json`（SHA-256 `3d8f0365efe5662c0e285558ffb03511a04216ffda313cc57c64621abc7218a9`）把一条 MoE 相关任务通过 flow 和 host callchain 关联到 `model.py(913):forward` 和 `model.py(1592):forward`，比单独根据设备形状更支持该 scope 属于 target decoder 运行。这份 host 查询覆盖的是 MoE 相关调用，而非本候选的首条 HcPre task 45397；因此不能将另一任务的栈作为 HcPre 调用栈，不能凭此证明其权重是 `hc_attn_fn` 或证明 repeat 输出流入该 task。

冻结配置表明模型文件中的 HC 维度与 Engram 层；`evidence/frozen-serve.sh`（SHA-256 `c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba`）是启动请求，不是诊断 worker 的 resolved 对象；同次 profiler 路径中的 `pp0` 是进程身份标签，不独立证明全部进程的 PP size、该对象实际 SP 属性或运行时未被替换。局部可核对证据仍缺 HcPre 自身唯一 flow 对应的 host 调用栈、权重 identity、repeat 输出到 HcPre 输入的 tensor/storage 与生命周期、四路具体值，以及历史进程实际加载源码和 kernel binary 的对应。已有的先后位置与形状只能支持条件假设，不能转成已命中或性能收益。无新增采集、远端访问或设备操作。
