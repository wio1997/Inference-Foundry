# F15：DSA-CP 头部交换和输出投影的具体分块边界

本次针对 F15 补充冻结源码的数据映射和一份 CPU 可复算索引例子；它是结构证据，不是已验证的性能方案。`restore-tp-heads-witness.json` 记录源码 SHA；其中两个 CP 文件与已有 `evidence/actual-sku-and-cp.json` 记录的远端源码 SHA 一致。`python3 agents/fusion/reproduce_restore_tp_heads.py` 使用 Python 列表独立重建全元素映射，不运行 collective、设备或模型。

## 真实代码走向和分支

冻结启动脚本显式请求 `enable_dsa_cp=true`、TP=8；`dsa_v41_cp.py:34–45` 在 `enable_dsa_cp()` 为真且 PCP 为假时选择 `AscendDSAV41CPImpl`。这是配置和源码推导，历史进程最终解析值与选中实例尚未直接捕获。该 CP 实现会对未满 `per_rank` 的 attention 行补零、调用 `restore_tp_heads`，必要时再补隐藏状态的额外尾行，最后送 `_forward_o_proj`（`dsa_v41_cp.py:209–225`）。非 CP 的 `dsa_v41.py:432–438` 则没有这个 head all-to-all，不能将其费用无条件加到所有配置；PCP 是另一条实现。

上游 `dsa_v41.py:704–716` 先取得本 rank 的 attention 输出，再**原地**对每个 head 的末尾旋转维度执行 inverse RoPE，之后才进入 `_project_output`；零 token 有专门空张量分支。由 `device_op.py:75–91` 可见对应操作使用 interleave 和 `partial_slice`，A3 配置能力允许 `negate_sin` 参数，但具体 kernel 的乘加、内部精度、临时状态和与其他别名共享仍未证。不能把某个相邻 tensor 的 in-place 改写视为可随意删除的纯转置。

## TP 交换的准确索引

在 P>1、各 rank `tokens=L`、总 heads `H` 可被 P 整除的情形，`dsa_cp.py:77–91` 把 `[L,H,D]` 视作 `[L,P,H/P,D]`，排列为 `[P,L,H/P,D]`，`contiguous()` 后进行 `dist.all_to_all_single` 等分交换。按该调用的等分和源 rank 顺序拼接语义，输出满足：

`recv[r][s*L+t][h][c] = input[s][t][r*(H/P)+h][c]`。

这里 s 是发送 rank，r 是接收 rank。对冻结模型配置 H=64、TP=8，假如运行时确为该分支，每 rank 接受 8 个 head、P×L 个 token 行。脚本在 P=2 的小例子和 P=8、L=3、D=4 的简化例子检查所有元素；实际配置 `head_dim=512`，脚本只是缩短每个 head 的 channel 范围，不代表已执行硬件通信。per-rank padding、空行和结果后续截断必须保持。

## 可检查的重组，不是已证明的加速

- **inverse RoPE + pack**：生产者是 attention 输出，唯一显式后续代码路径经 inverse RoPE、补零、交换后送投影。对被处理的 token/head 块，可研究在同一写出过程中执行原 inverse RoPE 和按目的 rank 的 pack；保留原位修改可能被其他 alias 观察的语义、interleave 对应维对、cos/sin、`negate_sin`、中间舍入、零 padding 与空 query。是否实现了真正一次写入，alias/外部消费者和底层指令仍未知。
- **交换与投影按块流水**：上面的索引意味着 receiver r 的某一 `(s,t)` 行只依赖 sender s 的该行对应 head 组，因此在 collective 提供**已完成且不会覆盖的块**、各 rank 对分块与顺序一致、目标投影可按该组和 token 单独执行时，该块可在其他块仍传输时开始计算。原代码只显示完整 `all_to_all_single` 调用及后续消费；它没有证明运行时支持接收块提前可见或 per-tile 事件。head 排列、graph 固定 buffer、不同 rank 的空行与 collectives 次序必须保留。通信与 Cube/Vector/MTE 抢资源可能抵消重叠收益。
- **投影的双重 GEMM**：`dsa_cp.py:1754–1795` 非 A5 量化路径用 `npu_transpose_batchmatmul` 合并指定转置和第一组投影，再将 reshaped 结果交 `wo_b`。这个 transpose+batched matmul 是**已有复合调用**，不能重复记成待实现的新融合；第一投影输出仍是第二投影的输入。`model.py:434–445` 为 wo_a 要求 ND 以供该调用，wo_b 为 row-parallel。两个投影组装或 tile 化必须保持每组对应的权重、偏置/缩放、阶段输出 dtype、舍入和 TP 归约。两层矩阵相乘换成预先相乘权重并不自动与原浮点数值等价。
- 现有 `model.py:794–808,832–836` 与 `879–902` 显示，在 sequence-parallel 分支，wo_b 留 TP partial sums、外层再 reduce-scatter；其他分支的最终归约取决于 row-parallel 实例。把 projection 直接送下一阶段时必须保留实际通信和贡献，不可单凭两个函数相邻删 collective。是否可在各块上等价 reduce-scatter 要验证顺序、数学精度和所有 rank 的参加条件。

`dsa_v41.py:690–693` 注释称结果写入 graph-stable output；实际 `dsa_cp.py:1795–1799` 先得到 `projected_output = self._apply_wo_b(...)`，随后 `output[...] = projected_output`。所以目前源码只能确认调用约定和赋值，不能仅用注释断言无中间分配/拷贝，也不能仅凭赋值断言产生了实际 HBM copy。具体 alias、lowering 和设备轨迹未知。

## 与历史 trace 的关系

既有 16 rank layer attribution 里可见 `TransposeBatchMatMul` 和 `hcom_alltoall_` 类型，但同层也有其他投影/通信；没有同一次操作的 producer→consumer tensor identity 或每块 ready 事件，因此只能说明名称出现，不能确定特定两个算子归属 F15，也不能从时间重叠推出通信已隐藏。历史各层的实际分支、分片、可复用数据与收益仍 UNKNOWN。
