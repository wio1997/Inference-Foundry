# F14：Sparse attention 结果、inverse RoPE、head 交换与首级投影

## 审查范围

本次聚焦 F14 中 attention 归约的**最终值**如何分块送往 inverse RoPE 和第一输出投影，并研究 rotation 可否按 token/head 跨越 head all-to-all。F15 已经分析过 inverse RoPE + pack、交换映射、逐块通信条件；F16 已分析两级输出投影和 BF16 反例，重叠部分不能重复记收益。这里仅只读既有源码和样本，未改目标代码或运行设备。

## 源码与历史窗口

- `attention/dsa_v41.py:613–680,704–716` 中 `npu_sparse_flash_mla` 消费 query、SWA cache，条件地消费 compressed cache、稀疏索引、各类位置/序列 metadata、`sinks` 等；`return_softmax_lse=False`，caller 取 output。该算子本身已是 composite attention，不应将其内 attention 流程算成此前未融合。代码随后对 attention output **原地**执行 partial inverse RoPE，再交 `_project_output`，零 actual token 则构造空输出、仍走下游 padding/交换路径。
- `device/device_op.py:76–91` 通过 `inplace_partial_rotary_mul` 和 `partial_slice=[start,end]`、interleave 处理目标 tail，`inverse=True` 时按能力设置 negate-sin 或负 sin。`model.py:381–386` 和 `evidence/model-config.json` 的配置对应 `head_dim=512, qk_rope_head_dim=64`，即每个 head 的前 448 通道保留、最后 64 通道做旋转。实际 kernel 内部精度、分块和别名仍未知。
- 条件 legacy CP 分支 `attention/context_parallel/dsa_v41_cp.py:209–225`：先按 per-rank token 行补零，再 `restore_tp_heads`，不足 caller 全部 hidden 行还会补零，截取后送 `_forward_o_proj`。`dsa_cp.py:77–91` 的映射（F15 已用脚本复核）是 `recv[r][s*L+t][h][c]=input[s][t][r*(H/P)+h][c]`，适用于 P 可整除 heads 等该例条件；不同 CP/PCP 分支不能直接照搬。输出投影的 `npu_transpose_batchmatmul` 在已审条件下本身是转置与 GEMM 的现有复合调用，不能将它作为新融合。
- 历史 stage170 已保存 EP8、layer24 run2：stream 89 的 `SparseFlashMla` task1277 在 706.886–751.985 µs 输出 BF16 `[3,64,512]`；`InplacePartialRotaryMul` task1278 在 753.045–756.865 µs 的 BF16 输入 `[3,1,64,512]`、cos/sin FP32 `[3,1,1,64]`；之后 `TransposeBatchMatMul` task1282 在 790.585–841.144 µs 输入 BF16 `[24,1,4096]×[1,4096,1024]`。相对起点、具体 SHA 和所有限制见 `F16-TWO-PROJECTION-TILE-AND-ROUNDING.md` 与 `f16-layer24-run2-evidence.json`。同形、同流及相邻说明与源码链相容，**不绑定 output 地址、旋转方向、source call、通信依赖或各 tile 完成时刻**；这些数字不能相加计算融合收益。

## 具体分块与重组条件

1. **final attention → rotation → pack/投影**。冻结源码调用现有复合 SMLA，再对其输出执行 partial inverse RoPE 和投影；其普通注意力语义要求最终结果与所有实际有影响的 key/value、sink、mask 及归一化效果相符，但这不证明每个可融合实现都必须逐项计算所有有效 key/value 或等待所有分支。若某些值、权重和可达状态使结果与某些项无关，或已有足够的信息、缓存或可证明正确的重算方案，可在满足原输出、精度和副作用契约的前提下简化，见 F13 的条件反例。不同 token/head 在相应状态互不依赖的条件下可各自前进；旋转的 interleave 配对和 `wo_a` 所需信息须按实际权重及可达输入证明，不能从 shape 断言每个输出依赖对应组的全部输入通道。可研究在 attention 完成某块后保留原 attention 输出 dtype 的舍入效果，再做 inverse RoPE，并把该块直接 pack 或交投影；若所需数据、顺序及存储/事件合法，可以按组和 token 流水。当前公开的复合 SMLA 调用并未证明已把内部 tile 以可访问的稳定 buffer/event 发布；跨 rank 接收与 GEMM 需要另外的 ready 约束。保留图输出、in-place alias、其他消费者和 padding，不能直接断言消去实际 HBM 访问或现有 launch。
2. **尝试把 inverse RoPE 移到 head 交换之后**。对于上面的交换索引 E，若对每个来自 sender `s`、token `t` 的接收元素都使用**对应原 token** 的同一组 cos/sin 和相同 64 维配对，则数学上 `E(R_s,t(input_s,t)) = R_s,t(E(input_s,t))`；head 交换是元素重排，未引入归约。这让接收侧 rotation 和本地 `wo_a` prologue 合并成为可能，但要求接收侧能获得每个 source token 的位置对应 cos/sin，保证层、rank 和 dtype 精度相同，不能用接收 rank 本地 token 的位置代替。原先是对有效 token 旋转后**再零填充**：移动到后面必须跳过填充行或精确重现原零及符号，避免 padding 被非有限 cos/sin 污染；还须保留空 query rank 的 collective 参与、交换布局、所有读写 alias、stream/event 与 graph buffer 生命周期。以相同输入、系数、dtype 和位级运算为前提，此重排可望保持元素结果；不同形状实际 kernel 可能改变计算细节，仍须验证。现有实现没有证明此重排或分块事件已存在。
3. **不能随意吸收到固定权重**。相同 token 的实数下可写 `W R(p) x = (W R(p)) x`，但 `R(p)` 随 token 的位置、层 cos/sin 改变，故对所有位置使用一个静态预变换权重一般错误；对每个位置动态变换权重也增加依赖和成本。更重要的是源调用有 attention 输出 BF16、旋转结果 BF16、再输入投影的舍入边界；直接把旋转作用在 SMLA 的更高精度内部 accumulator，或代数改写为权重变换后合并点积，一般改变输出。可研究在投影 prologue 内**显式保留同样的中间舍入**，但精度、复杂指令序列与实际生成效率尚未证明。
4. **简化舍入反例**。只作为说明，假设一个 head 的未舍入 attention 坐标 `(x,0)`，`x=1+1/256`，逆旋转某一坐标乘 `c=1/√2`，BF16 round-to-nearest-even：先 BF16 输出再旋转，`bf16(bf16(x)×c)=0.70703125`；省去中间 BF16 舍入则 `bf16(x×c)=0.7109375`。这证明一般重排不保持位级等价，**不证明该样本真实 kernel 的 accumulator、旋转系数和舍入实现等于例子**。若验收允许近似，实际误差阈值仍须由任务确定。
5. **不把 rotation 放到 score/缓存一侧**。线性旋转在实数下可以分配到加权和的各项，但本处 rotation 使用**输出 query 的位置**，同一个缓存 value 可供不同 query 位置读取；预先用某个固定 key 或缓存位置旋转，会导致后续 query 的语义不同。对每个 query–value 项分别旋转会复制计算，并改变中间舍入；是否能在 SMLA epilogue 做一次末端旋转取决于真实 kernel 契约、内部计算精度和上述舍入，不可凭公式假定免费或等价。

若上述配置且 trace 所示 BF16 `[3,64,512]` 属于该调用，一个 3-token 张量的逻辑大小为 196,608 字节，其中旋转 tail 是 24,576 字节；这是元素大小计算，不是实际读写字节、最低流量、性能极限或可节省量。实际 tensor 归属和能否安全直接发布 tile 均未知；当前材料不足以排序这一候选或承诺任何加速。
