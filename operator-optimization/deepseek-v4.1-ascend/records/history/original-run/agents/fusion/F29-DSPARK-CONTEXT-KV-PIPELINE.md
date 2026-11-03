# F29：DSpark context KV 投影、归一化、RoPE 与持久缓存

本次补充检查同一 context 输入被各 draft 层消费的真实源码、归约及状态边界；这是 F29 简表的深入说明，不涉及 F28 的 embedding 初始化。本记录只读冻结源码和已有 trace，不修改模型、不运行 NPU。**以下是可能重组的条件，未证明性能收益或实际执行绑定。**

## 生产、分组和后续消费者

- `spec_decode/llm_base_proposer.py:1063–1077,1480–1481`：目标侧隐状态先经 DSpark `combine_hidden_states`，随后 first-pass 输入准备，再在调用 draft 模型前调用 `build_model_inputs_first_pass`。`spec_decode/dspark_proposer.py:258–259,270–310` 将前 `num_context` 行保存到持久 buffer，并为各 attention cache group 调用已有的输入扩展 kernel，生成各 group context/query slot、共享 positions 等；再按 `_layer_group_idx` 从每 group context slot buffer 建列表。因此不同层若属于同一个 group，其列表项**可以是同一 buffer 的引用**，不能将列表不同位置误当独立存储。`_layer_group_idx` 根据排序后的 cache layer names 建立；`models/.../dspark.py:301` 根据 `self.layers.values()` 枚举消费，特定运行两种顺序一一对应仍需实例绑定，不应凭静态代码猜具体 gid 或实际组数。
- `spec_decode/dflash_proposer.py:301–319` 把 context 的 hidden、positions 和每层 slot 按实际 `num_context` 截切后传入；`dspark.py:284–309` 对每层使用其 `self_attn.wkv` → `kv_norm` → 拆出 nope 与 rope 子段 → 根据位置和该 `rotary_emb.layername` 取 cos/sin → 旋转 rope 子段 → cat/contiguous，写该层 `swa_cache_layer.kv_cache[0]`。尽管命名 `_project_shared_kv`，它是针对某层 attention 投影，不表示三层输出相同或可以共享参数/结果。
- `model.py:381–386,417–425` 中 wkv 是带 `quant_config` 的 `ReplicatedLinear(hidden_size,head_dim)`，之后完整 `RMSNorm(head_dim)`。冻结配置 `hidden_size=5120, head_dim=512, qk_rope_head_dim=64`，因此代码的逻辑切分为 nope 448、rope 64；运行权重 dtype、quant 算法及真实 shape 仍需绑定，不能仅据配置判断实际算子和物理流量。
- `attention/dsa_v41.py:227–242,613–680`：非 packed scatter 对 cache squeeze 后按 cache dtype 转换更新值并 `contiguous`，`npu_scatter_nd_update_sk` 按 `[-1,-1]` 跳过并保留 cache view 的实际 page stride；后续 `npu_sparse_flash_mla` 把对应 attention 的 SWA cache 作为 `ori_kv`，还读取其 block table、sequence lengths、sparse indices 等。相同模块的 `_write_swa_cache` 又按 query metadata 往该 cache 写入本轮 query 数据，必须维持 query/context 写与 attention 读取的正确顺序。packed 分支调用 `write_attention_cache`、其协议另算，不可套用非 packed 布局。
- 源码 `dspark.py:295–309` 逐层先投影后 `_store_standard_swa_kv`；当 context 本身空或 slot 列表为 None 时整体提前返回，每层 slot None/空则仅该层存储函数跳过。`dspark_proposer.py:359–363` dummy 场景把组 slot 填 -1；profile 路径也可能在未提供 slot 的调用中走空路径。因此仅看到某个 profile 内的类似算子名，不能证明它完成了生产请求的 context 写入。

## 可具体研究的组合及限制

1. **同源跨层投影打包**。若层间对同一个 context hidden 的读取、量化方式/权重布局和 rank 分布已证兼容，可将不同层 wkv 的输出通道打包为一次更宽的矩阵乘，之后分别交给各层自己的 512 维 norm、RoPE 和 cache。不能把 3 层的 KV 数值合为 1 份，也不能将各层的 norm 合为一个 1536 维归约。原三次量化/矩阵乘的 scale、zero-point、激活量化范围、accumulator 与输出 dtype、舍入、以及 packed 权重 shard 都需逐层验证；拼接后一次量化或另一 GEMM 的规约次序不自动等价。层数量、是否每层确实需要写与样本中的 context 行数未绑定；更宽矩阵能否提高利用率亦未知。
2. **行 tile 直接写 cache**。对某一个 layer 和 context row tile，只有先得到该行 512 维投影及完整 norm 的统计量，才能产出对应 nope 值以及旋转后的 rope 值；随后可研究把两段按该 layer 的有效 slot 和真实 page stride 直接写 cache，减少逻辑 cat / dtype 转换的中间物化，不能省投影输出舍入、norm 权重及 epsilon、位置与旋转的实际 dtype/计算顺序、cache dtype cast 和负 slot skip。单行内分 feature tile 可以并行算平方和部分值，但完整归约与原数值误差规则须保留；不能直接把 448/64 两段各算一个独立 RMSNorm。
3. **不能无条件交换 norm 与 RoPE**。即使视旋转为精确正交、它保持向量平方和，带逐特征权重的 norm 一般也不与旋转交换。例如忽略公共归一化比例，`γ=diag(1,2)`、`x=[1,0]`、90° 旋转 R，有 `Rγx=[0,1]` 而 `γRx=[0,2]`；该反例说明需要逐特征权重及精度条件，不是实际模型数值测量。
4. **按层提前写/跨层流水**。不同层在数学上各自读同一 context、写各自 cache，可能将后层投影或 cache 写与首层草稿计算部分交叠，但必须先证明各 cache 的真实地址/互不覆盖、per-group slot 和 positions 已准备完毕，确保后续各层在首次读取 cache 前观察到正确写入，且与同一层 query KV 写的相对顺序、所有流 event 及下一轮 buffer 覆写安全。某些行 slot 无效并不表示可以在当前调用语义中随意删整个投影：可能存在异常、算子/rotary side effect、其他可观察行为及重复 slot 的写冲突。多流还可能争抢 Cube/Vector/MTE，源码循环先后不能直接转换成收益。

原输入扩展 kernel、`npu_scatter_nd_update_sk`、`npu_sparse_flash_mla` 都是已存在的复合操作，不应把其内部逻辑再算作新融合。`cat` 和 `contiguous` 是源码表达式，是否各自真的产生设备任务或实际访存，要看运行时；无法据此报节省的时间或 HBM 字节。

## 固定窗口查询与证据限制

已有 stage172 的 gap 中观察到与 F27 三路 concat 及宽投影形状相容的任务，但没有 tensor 地址、调用或请求绑定，它本身不证明 F29 发生。准备了 `query_f29_dspark_gap.py` 与 `run_f29_dspark_gap.py`，在同一已冻结 EP8 gap 中只检索与 `[24,5120]×[512,5120]→[24,512]` 相容的矩阵乘和少量近邻，以探索是否有对应形状；校验 CSV SHA、限制输出、不运行设备。此处 24 只是选定窗口的探针值，**不是**本次 context 行数已被证明为 24。匹配同 shape、时间或流不能证明是该层的 context wkv 或确认数据依赖；匹配不到亦不否定其他行数、量化 kernel、融合、rank 或未导出任务。stage176 已由 ZCode 执行，结果及限制见下。


### stage176：有限窗口实际导出结果

独立核对 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-176-f29-dspark.exit-code` 为 `0`，`.verified.json` 的 `model_verified=true`，两次观测模型名均为 `deepseek-flash`。输出 `f29-dspark-gap-evidence.json` SHA256 为 `8b28ca4f17d235c41aa6667207ff5756038e65f040861190683f18f7243e6e8a`，与该运行报告一致；查询脚本 SHA256 `888cf8d908198e52e135eebe3015305ebc3942eb16a42383fd698cef0e2d9d84` 一致。CSV SHA256 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 与冻结 manifest 中对应 EP8 rank 一致；window 文件及 manifest 的 SHA 亦匹配输出。

范围是 EP8 指定 target layer39 run2 最后标记结束（`1793152960223546576 ns`）到下一 target layer0 起点（`1793152960261768012 ns`），约 38.221 ms。只筛查指定 `[24,5120]×[512,5120]→[24,512]` 形状的 MatMul 以及其 ±30 µs 内少数类型，**不是完整草稿运行 trace 或该调用的全部算子**。相对于窗口起点，记录显示：

| 记录 | 相对时间 µs | stream / task | 形状与 dtype |
|---|---:|---|---|
| MatMulV2 | 1169.937–1180.737 | 47 / 53843 | `[24,5120];[512,5120]` BF16 → `[24,512]` BF16 |
| RmsNorm | 1180.777–1190.776 | 47 / 53844 | `[24,512];[512]` BF16 → `[24,512]` BF16 和 `[24,1]` FP32 |
| MatMulV2 | 1325.494–1334.873 | 47 / 53873 | 同上 MatMul 形状和 dtype |
| RmsNorm | 1334.893–1344.873 | 47 / 53874 | 同上 norm 形状和 dtype |
| MatMulV2 | 1483.010–1492.450 | 47 / 53903 | 同上 MatMul 形状和 dtype |
| RmsNorm | 1492.470–1503.450 | 47 / 53904 | 同上 norm 形状和 dtype |

另有两个 `ScatterNdUpdateSk`，分别为 1312.234–1325.474 和 1472.251–1482.990 µs，在 stream 47 上、分别紧邻后两个 MatMul **之前**；输入形状 `[17307,128,512];[24,2];[24,512]`，dtype BF16/INT32/BF16。它们均被 ±30 µs 筛选捕捉；这不是三个 scatter 的完整统计，也没有说明其各自输入来自哪一条投影。由于 filter 只返回选定近邻，没列出 RoPE、cat 或第三次 scatter 不能说明其不存在。

三对同流投影/归一化及两次 cache 更新的形状、dtype、时间次序与三层 context KV 源码预期**相容**；但不知道 tensor 地址、实际层号、请求、参数和事件对象，也未证明 24 行是目标 context 或任何 scatter 属于前一对，因此不能说已绑定 F29 的真实执行，更不能把这些耗时相加当作串行关键路径或潜在加速收益。所有任务的 model ID 是 `4294967295`，不能据此识别具体模型。此次仅重读已导出 CSV，没有启动设备。

## context RoPE 位置索引的跨层复用：静态候选

在保存的 `dspark.py:55–68,284–309` 中，`precompute_and_store_context_kv` 对每个 draft layer 以同一个 `context_positions` 调用 `_project_shared_kv`，后者调用 `_apply_dsv4_rope`，每次都执行 `get_cos_and_sin_dsa(positions)` 再按该层的 `rotary_emb.layername` 取 cos/sin。这是源码调用链上的重复，不证明本次真实调用次数、执行耗时或设备访存。空 context 或 slot 列表为 None 时函数提前返回；具体层数、运行对象及实际路径仍需绑定。

补查 `sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/rope_dsv4.py:107–190,235–270`（SHA-256 `9cbcd2cb2a1123dcbf5afc3999025035d155dfd3b160f186c36aebb5aa7c706b`）：该调用未指定 `use_cache` 或 `layer_names`，分别使用默认 `False` 和 `None`；函数将 Tensor positions 包装成 `default` 位置组，遍历已注册且具有 `default` 的每个配置，从各配置自己的 full cos/sin 表按 positions 索引生成本次结果；`RopeDataProxy` 才按具体 layer 名解析其配置和需要的 group。因而，即便不同层有不同 RoPE 配置，该源码路径仍可能在同一次 context 预计算内重复索引多个配置；不能假设所有层共用同一个表，也不能把它与首层 query positions 的索引混为一谈。

可研究在一次 `precompute_and_store_context_kv` 中，对同一 positions、真实涉及的配置/group 生成一次对应 cos/sin，并供后续各层按层名解析；应检查配置注册与缓存初始化在调用期间不变、positions 未被异步覆盖、读取层名/group 均有对应项、索引结果 dtype/layout 和异常行为保持原语义、所有下游读取完成前结果不被释放或覆盖。若尝试通过 `layer_names` 过滤，需要证明实际 layer 映射与注册表完整，不改变异常行为或其他消费者。不同位置域不得共用，跨流必须保持输入 ready、配置表及输出生命周期，并检验 allocator/graph、并发请求和流同步；复用所增加的持有期与缓存压力也要计算。此候选与 F29 每层 RoPE/cache 写和前述跨分支流水共享工作，不能重复计收益。

stage176 只筛选特定 matmul 与少量邻近类型，不能据未出现 RoPE 索引认定它没运行；也没有 tensor、请求、配置或实际算子绑定。只有源码快照证据，未经 runtime 绑定、代价估计与正确性验证，不认为已实现、不计算提速。

## 投影合并与分层流水的冲突：反例和证据

`dspark.py:295–309` 的预计算逐层投影并写 cache；`dflash_proposer.py:301–319` 调用该预计算；Ascend `llm_base_proposer.py:1480–1494` 在其返回后才调用草稿模型。源码顺序描述调用和下发，不单独证明所有异步设备工作同步完成。提前启动首层需要改变生产者和消费者的连接、逐层 cache 可见性及调度，不能仅拼接权重。

F29 的跨层投影打包与逐层提前写流水不能独立计算收益。如果合并 GEMM 的接口仅在全部输出完成后通知消费者，首层 cache 也要等待全部层投影；逐层 GEMM 可条件研究在首层输出就绪后令后层准备与首层草稿计算并行。若设备实现提供正确的分块可见、跨流通知和生命周期管理，这一限制可能放宽，当前接口尚未证明。

精确的抽象反例：两层 cache 准备各需 3，草稿层计算分别需 7 和 4；准备与草稿计算资源独立，层 2 计算依赖层 1 的输出和自身 cache。逐层准备时，首层 3–10 计算，第二层准备 3–6 与之重叠，最终在 14 完成。若打包准备总共 5，小于原来总计 6，但仅在 5 时才发布首层 cache，则层计算在 5–12 和 12–16，最终反而在 16 完成。这是条件模型反例，不是该卡实测；只能排除投影总时间变短必然降低关键路径的无条件推论。

在 stage176 的单个 EP8 指定窗口，三条相容 MatMul 与紧随 RmsNorm 分别为 10.800+9.999、9.379+9.980、9.440+10.980 微秒；无张量、调用、跨层发布或首层启动绑定，不可将其代入反例计算真实收益。需要逐层 context cache 发布、首层计算启动和读取、事件对象等证据；缺失时实际重叠收益仍未知。
## noncausal 草稿行与 SWA cache 的跨模块就绪边界

F29 的 context KV 和逐层 query KV 是同一个 SWA cache 的两类写入，F31 已推导过 noncausal 索引长度，本节进一步检查**这些索引对 attention 能否按 query 行提前执行的依赖**。`dspark_swa_indices.py:20–43` 的 Triton helper 用 token_idx 找到请求后，只按该请求的 `query_start`、`query_end`、`seq_len` 和窗口生成索引；同一请求的每个有效 query 行因此得到相同的 `[start, start+visible)` 索引及长度。`attention/dsa_v41.py:1071–1096,1117,657–675` 在对应 noncausal 条件下使用这些 indices、lengths 和 `ori_mask_mode=0` 调用已有复合 `npu_sparse_flash_mla`。这只证明接口传入上述数据，**尚未证明历史采用该分支或所装 attention 的掩码及物理读行为**。

可复算例子：假设一个请求 `q=6`、`S=1000`、`W=128`，索引宽度 256、无溢出，helper 给每行 `start=866`、`visible=134`，有效逻辑位置为 `866..999`。第一 query 的位置若为 994，它对应的 indices 仍包含后续位置 999，不能仅按“第一个 query 行的前缀已写好”判断该行可提前计算。**若**运行 SMLA 按该索引集与 noncausal 模式确实读取位置 999，且该位置本轮值由尚未完成的 query KV 写入，且映射指向该物理 slot，第一行必须等待那次写入；忽略该写时，旧 cache 与新值不同即可构造输出不同的反例。上述条件缺任何一项都不能把索引出现直接认定成实际内存读取，block table、位置/slot、写入与 event 的绑定仍未知。此前 F08 的逐行 ready 是一般条件，不应错误升级为按 token 顺序天然满足。

可研究的重组是**按已证明不互相覆盖的请求或其完整读取集合**建立 cache ready 依赖，再尝试与其他请求的 KV 投影/写入交错，而不是对同一请求无条件按 causal 前缀切割。还需证明各请求页和 slot 不别名、不回收、所需 context KV 与 query KV 完成、Q 和 metadata 就绪、跨流可见性，及其他 cache 消费者保留；原完整 SMLA 接口没有在此源码中暴露每个请求独立完成事件或局部调用合同，按请求拆分会影响 batch、复合 attention 内部的布局和资源争用。**若**实际 kernel 使用全部所给非 causal indices，则其中更晚的索引可用于证伪只等待较早写入的设计；否则必须以实际 kernel 合同重算读取集。

同一请求每行生成重复索引还提示一种不同的条件候选：如果后端能按请求重用共享索引向量、却对每行提供与原 `[T,1,D]` 完全相同的值与有效长度，可避免重复物化；但原 metadata 和 SMLA 接口仍接收逐行张量，跨 rank、builder 的稳定地址及被其他消费者观察的 buffer 语义均须保留；不能把按请求共享当作当前已实现或无成本。它与 F31 的 global→local 长度复用是不同边界，却可能重叠，不能叠加估计收益。现有信息不足以判断其在工作负载上的耗时或是否值得改动；本节为冻结源码静态分析，不改实现、不运行 NPU。

### 按请求共享 SWA 索引时的空请求与 padding 行

对上节按请求共享 noncausal SWA 索引的条件候选，补充一个明确的索引生成边界；这不是额外独立的融合收益。已存 `agents/fusion/frozen-source/vllm_ascend/ops/triton/spec_decode/dspark_swa_indices.py:20–44`，SHA-256 `335be10476943d4ad429f4d52b7a0e069f3b65511d28265b0e9d9f7d2e94111c`：每个 token 行 `req_idx` 是满足 `token_idx >= query_start_loc[req+1]` 的有效请求结束位置数量；重复结束位置会跳过零 query 请求。`req_idx >= num_reqs` 时后续 load 的其他值为 0，该行的 `visible_len=0`，指定宽度的索引每列写 `-1`。`attention/dsa_v41.py` SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd` 的前述 noncausal 分支消费 indices、lengths 及 operator metadata；记录与源码不能代替实际 kernel ABI、有效长度、地址或 event 绑定。

具体整数反例：`query_start=[0,2,2,3]`，`seq_lens=[5,7,9]`，窗口宽 2，index width 4，num_tokens 5。按已存 helper 的算术，行 0、1 对应请求 0，长度为 4、索引 `[1,2,3,4]`；请求 1 的 query 为空，行 2 对应请求 2，长度为 3、索引 `[6,7,8,-1]`；行 3、4 超过最后结束位置，对应 invalid request，长度均为 0、索引均为 `[-1,-1,-1,-1]`。如果改成每请求只保存一个向量，却把未归属行的 request ID clamp 到最后真实请求，则行 3、4 会错误地得到长度 3 和后一个索引向量。CPU 可复算如下：

```python
starts = [0, 2, 2, 3]
seq = [5, 7, 9]
window, width, tokens = 2, 4, 5
out = []
for t in range(tokens):
    req = sum(t >= end for end in starts[1:])
    if req < len(seq):
        qlen = starts[req + 1] - starts[req]
        start = max(seq[req] - qlen - window, 0)
        visible = seq[req] - start
    else:
        start = visible = 0
    out.append((visible, [start + c if c < visible else -1 for c in range(width)]))
assert out == [(4, [1,2,3,4]), (4, [1,2,3,4]),
               (3, [6,7,8,-1]), (0, [-1,-1,-1,-1]), (0, [-1,-1,-1,-1])]
```

该例假定结束位置单调且所有有限数字在相关类型范围内，是源码数学映射的构造，不是声称历史负载有空请求、padding 或越界读取。任何实际按请求共享/广播设计仍要保留逐行长度和无效行、请求顺序、tensor dtype/stride/地址及 SMLA 的消费合同；是否接受广播或 zero-stride 尚未知，buffer 复用及 metadata 的 producer/event 也须另证。索引本身出现不证明 operator 在该位置执行物理读取。未启动设备或修改目标实现。

### context position 与 group slot：可观察语义依赖的条件

修订 G5 中的 position 与按组 slot 边描述了冻结源码的值流，不自动要求所有等价实现执行原来的输入扩展 kernel、复制 buffer 或全局同步。对于固定 cache 初态和精度合同，若存在两种允许的 context position，在其他相关条件相同下使同一输入的 RoPE 后值和最终**可观察** cache 状态不同，则正确实现必须能区分这两种情况；它可直接使用已有位置、预计算的 cos/sin 或等价结果，无须物化原 position buffer。若两种位置产生相同可观察结果、相应 slot 无效、消费者无法观察差异，或有关信息在初态已知，则该位置对该接口未必需要新增工作或传递。

slot 同理：若固定同一非零 KV 更新值与 cache 初态，两个允许的有效 slot 分别写入两个不同位置，且终态可被观察并因此不同，则实现必须区分目标落点；信息也可以来自已知映射表、位置推导、缓存或其他共享状态，不要求执行原 slot copy。此条件依赖 slot 有效性、结果 dtype 与取值、覆盖顺序以及终态可观察性；不能用源码中 slot 参数的存在自动证明某次运行满足它。

组间独立 buffer 不等于独立信息：如果两个 group 的 slot 始终由同一个变量决定，一份已知或传入的信息就可能同时决定两组写入，不能按 group 数将必要信息或资源时间相加。只有证明各组的允许取值可独立变化、联合 cache 终态都可观察，并核对初态与全部旁路，才可研究其**联合可区分输出集合**所需信息；即使如此，仍需独立证明同一边界的容量和信用，才能形成时间约束。

这一去重也适用于 position 与 slot 之间：若 slot 可由 position 和接收侧已有的固定 metadata 推出，两条逻辑边可能共享同一份信息，不能把各自的条件信息界相加；若 block table 和缓存状态允许它们独立变化，则须根据共同可观察结果重新确定联合像。真实取值关系、接收侧初态及历史运行条件均未知。

上述仅是条件性语义依赖及去重规则。真实运行的 RoPE 表、输入可达值、实际 slots、cache 观察边界、缓存初态和跨流事件均未绑定；这里不声称真实字节或时间下界，也不证明具体融合收益。

## 跨模板结构筛查：context buffer → 各层 KV 投影的数值和生命周期（离线）

`graph-revision-cross-template-v1/index.json` 的 25 个集合按三条跨模板条件连接分别为 hidden 6、positions 12、slots 7；它们是受限的 2–4 节点连通结构而非融合数。positions 和 slots 各自的**双节点**只描述两模板之间的 metadata/view 对应，不构成两个可合并算子；若组合中有 `B::input_expand`，它表示一个已把 context position、slot 复制及 query 扩展组合在一起的现有 kernel，`B::slot_outputs` 等是数据出口，不能再次把每个出口算成独立计算。`B::kv_precompute` 是 K 片段的外层 composite，已被本索引排除在连通枚举外；不能另把外层和细化的 `K::proj/norm/rope/scatter` 加算为两遍。positions → RoPE 的跨层表复用、slot 与 scatter 的已有条件分别按本记录前文和 `DSPARK-PER-GROUP-DEPENDENCIES.md` 审核，不能把同一处重复记收益。源码中的 alias、chunk ready 和非该图已知的其他消费者仍未闭合。

在 hidden 的六个结构中，`CROSS_STRUCTURE_6ee286b2c2e6e8b6` 含 `B::main_norm → B::context_buffer → K::context → K::proj`；`K::context` 是跨模板输入/view，**不是算子**。它提示一个与 F27 部分投影和 F29 各层投影重组重叠的条件候选：在前一阶段产生某些经原 dtype 语义处理的 context 行后，研究与某层 wkv 的分块计算更早对接，同时满足原持久 buffer 的所有可观察用途。不能仅因源码里复制一次便断定该复制是单独设备 kernel、该边发生 HBM 往返、或可以直接删除复制。其数值及状态依赖比简单的“输入相同”更具体：

- `spec_decode/llm_base_proposer.py:1049–1075` 的非零 K DSpark 路径先 `combine_hidden_states` 再 `set_inputs_first_pass`；`dspark.py:281–282` 的组合为 `main_norm(main_proj(aux_hidden_states))`。`dspark_proposer.py:48–59,257–259` 分配固定容量 `[max_num_tokens, draft_hidden_size]` 的 `_dflash_hidden_states`，dtype 是 `self.dtype`，把前 `_dflash_num_context` 行赋值写入该 buffer；继承来源及实例的实际 `self.dtype` 必须再绑定。所保存上游 `vllm/v1/spec_decode/llm_base_proposer.py:72–89` 确有 `self.dtype = vllm_config.model_config.dtype`，但本地快照缺 `eagle_proposer.py`，不足以独立闭合当前所有继承/覆盖路径。冻结配置文件的顶层 `dtype=bfloat16` 是配置线索，不是历史运行 resolved dtype 或 `main_norm` 输出 dtype 的证明。
- `dflash_proposer.py:301–319` 在 first-pass precompute 时只交 `[:num_context]` 切片、同长度 positions 和按层 slots 列表；`dspark.py:295–309` 在非空 context/slots 时，循环对各 draft 层使用该同一 context 行做该层自己的 `wkv → norm → RoPE → cache`。因此直接对接某一层不能丢掉其他层对相同版本行的读取，也不能在其他层仍读取时覆盖/回收。不同层投影权重及 cache 不可由同源输入推成同值；上层预分配地址与 graph capture/replay 和未查到的 alias/hook 也可能可见。源码顺序并不证明可逐块发布或任意跨 stream 可见。
- **dtype 反例仅用于约束不同类型输入的情况**：假设原 `main_norm` 输出 FP32，固定 buffer 为 BF16、赋值按最近偶数转换；某行两维为 `[1+1/256, 1]`，两个维度均可由 FP32 精确表示，BF16 buffer 变为 `[1,1]`。对权重 `[1,-1]` 用精确点积，读取 buffer 得 0，而跳过 buffer 直接读原 FP32 得 `1/256`。该反例不说明此次 main_norm 真为 FP32、这组值/权重可由模型产生、真实设备 GEMM 接受该混合 dtype 或具有同样舍入；仅说明不同 dtype 时不能把复制/类型转换当作无条件可删除。若 source 与 buffer dtype 相同且值/布局、存储生命周期和数值/异步合同均已证明，仍可研究按行/块对接或证明等价的暂存；需要保留其他读者所需内容及 graph 稳定地址，不能直接从数学模型推出可安全免 copy 或加速。即便原 source 已 BF16，其他中间归约、格式、异常、数据竞争和缓存成本仍需单独审查。

CPU 精确分数可复算这个抽象反例：

```bash
python3 - <<'PY'
from fractions import Fraction as F
x = F(257, 256)             # exactly halfway between BF16 1 and 1+1/128
q = (x-1)*128
n, rem = q.numerator//q.denominator, q - q.numerator//q.denominator
if rem > F(1,2) or (rem == F(1,2) and n % 2):
    n += 1
buffer_value = 1 + F(n,128)
print('source:', x, 'BF16 model:', buffer_value,
      'direct dot:', x-1, 'buffer dot:', buffer_value-1)
assert (buffer_value, x-1, buffer_value-1) == (1, F(1,256), 0)
PY
```

代码与图 SHA-256：`graph-revision-cross-template-v1/graph.json` `1950008ae0175e3a687b62087087ac240d62de957ec377acd26a413074ad9638`；index `dc3c9b670bedd392dff9d3c1a3e11a27155e380d2982d4584d0635c8fb863190`；manifest `0a540f5abf0752b70f9971645a679ba2d941738dfae3495a73728931910d084a`；`dspark.py` `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601`；`dspark_proposer.py` `70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c`；`dflash_proposer.py` `f3b736f44a59ec0831aa7f2849e2eae6d709951e955992b8016f49e772fce3ff`；Ascend `llm_base_proposer.py` `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`；上游 `llm_base_proposer.py` `5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab`；配置 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`。历史 profiler 对这些对象、tensor 地址、dtype、调用及 event 未绑定；无新增 NPU 工作、不修改实现或原图。
