# F28：DSpark 初始 embedding / HC repeat / first pre 的结构边界

本次只读现有冻结资料，不修改模型或运行 NPU。此处是对 F28 已有简表的补充，与 F02 对普通 attention HC pre 的分析有共用的 y/系数分支，但这里利用的是 **首个草稿层输入四个 HC 通道原始值相同、外部 premix 为 [1,0,0,0]** 的特殊结构；不能和 F02 的同一段收益重复计算。下述均是条件候选，不是已经实现的优化或实测耗时。

## 确定的数据与消费者

- `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py:159–176,248–276`：`VocabParallelEmbedding(input_ids)` 后 `unsqueeze(-2).repeat(1,hc_mult,1)`，有 SP 时先按条件调用 `sp_padding_mask`、再 `sp_shard(hidden_states)` 和 `sp_shard(input_ids)`；之后产生 FP32 的 `[T,4]` zeros，首列写 1，按顺序依次调用各 draft decoder layer，最后 `hc_collapse`，SP 时 gather 并截取。这里 T 是对应时刻的 token 行数，实际值和哪些请求/padding 行有效需运行绑定。
- `models/deepseek_v41/model.py:844–881,884–923`：该 decoder 层在入口把四通道 hidden 保存为 residual，调用已存在的 **复合** `npu_hc_pre_v3`，同时获得输入 `x`、`attn_post`、`attn_comb`、`attn_pre`。x 进入 input RMSNorm 和 attention；attention 完成后 `hc_post` 同时消费 residual、attention 输出、post、comb；后续 FFN 的 hc_pre 消费 attn_pre。后续层 HC 四通道会随 post 而分化，不能以初始四通道相同假设贯穿整个草稿模型。
- `agents/fusion/frozen-source/tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py:58–83` 的 CPU 测试参考描述：对四通道全部数据算 inv_rms、投影得到 pre/post/comb 和 Sinkhorn；若外部 pre_mix 非空，用它和每个通道相乘求和得到 y，输出转回输入 dtype。参考还说明投影按 HF32 round-toward-zero 模拟。该参考不是当前 169 已加载二进制的位级契约，已有 HC 自定义源码的编译绑定及对应 tiling 仍未知。

## 特定结构可研究的重组

1. **首层 y 旁路与系数分支并行**：假设初始每行四个通道都是 embedding `e`，premix 固定 `[1,0,0,0]`，按实数定义首层 `y=e`。因此可研究把 embedding 的适当分块尽早送往 first norm/attention，同时独立算 pre/post/comb；attention 后的 hc_post 仍必须等 post、comb 和 residual，FFN pre 必须等 attn_pre。还须先满足 embedding 实际并行通信完成、SP shard/padding、同 token 行、stream 的 happens-before，以及当前 `npu_hc_pre_v3` 输出边界；不应假设现有 kernel 已提前释放 y，也不能因没有数学依赖就忽略 Cube/Vector/访存争用。**有限数值与特殊值必须区分**：IEEE 运算中 `0 × ±∞` 为 NaN；符号零、NaN 传播及实际规约树均可能使直接 `y=e` 和原 multiply/sum 不完全一致。要么证明输入域和 bit 级行为满足契约，要么以原运算/验收精度为准，不能直接宣称恒等。
2. **四份输入的按块生成**：可研究令首层专用 hc_pre 的 load 映射在逻辑上把一个 embedding 块按四通道读取，或由生产者在消费者需要时构造块，减少显式四份副本；但首层 residual 的其他消费者 hc_post 仍须读取逻辑上四通道的数据。零 stride expand 是可观察的 alias，不能不核对 kernel 的 stride/连续性支持、写入/覆盖以及残差保活就直接替代 repeat。即使语义成立，是否消掉物化、每个 tensor 是否实际经过 HBM、是否会重复加载以及收益都未测。按配置 `hidden_size=5120`，若输入实际 BF16，四通道逻辑一行 40,960 字节，单通道 10,240 字节；这只是张量逻辑大小，不是节省字节或物理流量结论。
3. **系数投影按四通道折叠**：设四通道连接后的系数矩阵沿输入切为 `F0…F3`，实数有 `[F0 F1 F2 F3] [e;e;e;e]=(F0+F1+F2+F3)e`；也可探索每块产生原 4 个通道的部分乘积并保持原规约次序。前式理论上减少输入宽度，但先加权重再做乘法会改变浮点舍入、HF32 取整、累加顺序和可能的下溢/溢出；不能把实数恒等当实际算子等价。平方和/mean 的四份化简同样会更改规约次序；hc_scale、hc_base、epsilon、pre、post 及 Sinkhorn 的全部输出仍要保留。离线折叠权重还要证明模型权重不变、dtype 和 checkpoint 加载语义、精度需求以及重排结果可接受。
4. **SP 下先 shard 再 repeat**：理想情况下在 token 维切片与通道 repeat 可交换；但目前代码先把完整 embedding repeat、可能以其 shape 更新 padding mask，再 shard，input_ids 也要同样切分。想重排必须核对 `sp_shard` 真实 padding/分布逻辑、非整除尾部、mask shape、token 顺序和跨 rank embedding 的数据就绪，不能仅用数学切片恒等认定行为相同；也没有证据当前 SP 配置和本窗口命中此分支。

## 执行证据与缺口

冻结配置有 `hc_mult=4`；启动脚本请求 DSpark，但没有保存的证据将本窗口具体 embedding、repeat、first pre、对应 attention 输出和后续 hc_post 绑定到同一 draft 调用。F27 的 stage172 只看到一个历史 gap 内三路 concat 和相应宽度的矩阵乘形状，既不能证明这条 F28 路径命中，也不能估算其成本。现有资料未证明首层内的数据地址、运行 kernel 或各事件，故该候选实际优先级和收益未知；如找不到更具体已保存执行绑定，不将静态可能性当性能发现，不做额外无目的测试。

## 补充：target 模型首层存在相似结构，但 SP 顺序不同

本节是 F28/F02 的 **target 入口变体**，不登记为独立收益，也不宣称首层成本或其全模型重要性。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py` SHA256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`：`:1505–1550` 先选择 `inputs_embeds` 或 `embed_input_ids(input_ids)`，有 SP 时先 `sp_shard(hidden_states)` 与 `sp_shard(input_ids)`，再 `unsqueeze(1).repeat(1,hc_mult,1)`，构造首列为 1 的 FP32 `pre_mix`，随后进入层循环。`:879–899` 中首个普通 decoder 的 `hc_pre` 将初始 HC 状态及 `pre_mix` 作为输入，其输出接 input RMSNorm；`:902–905` 的后续 `hc_post` 仍需要初始 residual 与 `attn_post/attn_comb`。这与上文 DSpark 中 **repeat 后 SP shard** 的源码顺序不同，target 已经是 shard 后 repeat，不能把 draft 的“提前 shard”候选机械重复计入 target。

冻结 `evidence/model-config.json` SHA256 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7` 的 `hc_mult=4`、`engram_layer_ids=[1,14]`；根据 `model.py:832–839,1554–1585`，在该配置及普通从第 0 层开始的 target 路径下，第 0 层不插入 Engram 更新，因而初始四通道仍为 repeat 结果。该结论依赖历史实际归一化配置、模型分片及入口绑定；非首 pipeline 段、其他权重配置、前处理或有 Engram 的首个实际层不能一概套用。上游已有 `inputs_embeds` 分支，不得只针对 embedding lookup 构造新接口而遗漏该输入。

可条件研究：消费者接受同一行源和只读 HC 通道的专用表示，按需要生成四通道，或将首层 HC 混合分支及 norm 的必要块提前接入；之后的 HC post 仍要保留四通道逻辑 residual 和当前调用中算出的 post/comb，不能将共享地址视作可安全写入的四份独立存储。这个候选受 F28 上文相同的 FP32/BF16 中间舍入、规约树、NaN/Inf/符号零、特殊位置、padding/行域、stream 与 buffer 生命周期限制；即使数学上 premix 选择了首个通道，也不自动证明低级 HC 内核的特殊值行为或 bit 级输出相同。

现有材料未绑定历史 target 实际 repeat 的物理任务、graph lowering、张量地址、首层输入或内核读取情况，故无法确认 repeat 实际落地在何种存储层、是否已经被编译融合、是否影响关键路径；逻辑副本大小不是实际 HBM 流量，未做性能排序或节省估算。与 F02、F28 共享结构，任何后续测量必须避免重复累计。
