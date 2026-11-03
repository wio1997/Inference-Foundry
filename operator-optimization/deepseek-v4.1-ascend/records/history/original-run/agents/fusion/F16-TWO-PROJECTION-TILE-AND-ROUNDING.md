# F16：attention 两级输出投影到 rank 归约

## 与现有 F15 的区别和已有融合

F15 的 `F15-CP-PROJECTION-REVIEW.md` 已分析 inverse RoPE、head all-to-all 的数据映射，以及投影的整体边界。本记录只深化 **已接收 head 的第一组投影→第二投影→后续归约** 的粒度与精度；与 F15 的重叠部分不能累加收益。

冻结 `dsa_cp.py:1657–1685,1754–1800`：V4.1 CP 调用 `_forward_o_proj`，`full_gather_wo_a_enabled` 默认 false 时，取 `n_local_groups`，先调用已组合转置与 grouped GEMM 的 `npu_transpose_batchmatmul`，reshape，再 `_apply_wo_b`；局部权重走 `self.wo_b(...)`，full weight 分支走其 `quant_method.apply`。上游 `dsa_v41_cp.py:209–225` 先 padding、`restore_tp_heads` 并可能再次 padding，才调用该函数。它既不证明各 tile 能提前发布，也不能因 Python 源中 reshape/view 的名称认定额外设备复制。A5 的动态 MX quant/transpose-quant 分支与 A3 标准分支不同。

冻结配置 `num_heads=64`、`head_dim=512`、`o_groups=8`、`o_lora_rank=1024`、`hidden_size=5120`；启动请求 TP8。如果实际 CP 使用 local 权重、`n_local_groups=8/8=1`，每个 rank 每行的逻辑输入为该组 8×512=4096，第一投影 4096→1024，第二投影 local 1024→5120。不同 TP、full weight、padding 或实际量化实现必须重新推导。量化描述中 0/24 层 `wo_a.weight`、`wo_b.weight` 均为 FLOAT，是文件标签而非历史权重 tensor dtype/执行实例证明。

## 数据依赖与可条件并行

- 对单行输入，**直接稠密点积实现**按 4096 个输入 feature 形成第一投影 `z[j]`，再按 1024 个中间 feature 形成第二投影 `y[k]`；不同 j、不同 token 可独立分块。若保留这一实现及其数值契约，K 分块的结果需要以满足原精度与顺序的方式合并；当遗漏部分会改变可观察结果时，单个不完整部分和不能直接替代最终 y。允许特殊权重、受限可达输入、缓存、重算、代数重组或跨算子融合时，shape 本身**不能证明**每个 feature 对某个可观察输出都有影响，也不能证明必须物化或逐项计算所有中间值。下游一个输出 feature k 在该逐行线性接口的数学描述下不读取其它 k 的最终值；token 独立仍须以真实布局、索引、状态和消费者契约为条件。
- 已经完成某个 token/group 的 head 交换后，数学上可研究该行先完成投影，与其他未就绪行的通信重叠。当前 F15 的 `_restore_tp_heads` 是整 tensor collective，既无已绑定逐行 ready event，也未说明可重用的内部分片，不是声称现有通信已分块。空/不足 token 时的零 padding、接收顺序与输出后截取要保持；若 q 部分行其实不有效，则不能把 padding 当真数据计算出不同可见结果。
- `model.py:899–902` 的 `use_sequence_parallel` 路径会将 attention 输出 `sp_reduce_scatter` 再交 `hc_post`；`model.py:833–836` 相应设置 wo_b `reduce_results=False`。其他 `RowParallelLinear` 配置可能内部归约，需查 resolved 实例及 runner；不能对每一次额外重复 reduce。若被省略的 rank 信息会改变 HC 所需的可观察结果，则只提供未包含该信息的 partial value 不满足语义；对于恒零贡献、已经包含或可在接收侧等价重算的信息，不可单凭现有 collective 调用断言必须独立传输或等待该 rank。按输出 token/feature tile 流水归约需证明正确的 collective 接口、各 rank 次序、buffer 生命周期和 rounding；也可以研究其他保持输出和副作用的通信结构。现有代码及设备任务名称不证明按块通信或无资源竞争。
- 将第一个 GEMM 的结果直接供第二 GEMM 可研究减少中间物化；在需要保持原路径逐位语义、且没有证明其他等价变换的条件下，须保留中间舍入及第二 GEMM/collective 的可观察数值效果，不要求一定物化 z 或执行独立舍入算子。即使两级是线性运算，也不能未经证明把 BF16 中间舍入移走、预乘权重或交换 TP sum 与 BF16 round；若用户契约允许容差则按该容差重新评估。

**有限数值反例与适用范围（2026-10-30）：**抽象四维输入 `x=(1,0,0,0)` 和 `x'=(1,9,7,5)`，取第一投影某列权重 `(1,0,0,0)`，两者该输出都是 1；改后三项不会影响该输出。同理，若第二投影某列只有第一个中间权重为 1、其余为 0，则后 1023 项无需参与这个输出的数学计算。对跨 rank 求和，若部分贡献为 7、另一个 rank 恒为 0，则接收侧在已知条件下无须独立获得该零值就能得到 7；若另一个贡献为 3，则仅用 7 会漏掉信息。这里使用有限值、明确的无 bias 和普通精确加法例子，真实 BF16 的非有限值、signed zero、舍入与副作用需要另行处理；这不是声明 169 的固定权重有零、某 rank 恒零，或实际 collective 可以省略。复算 `python3 -c 'w=[1,0,0,0]; print([sum(a*b for a,b in zip(x,w)) for x in ([1,0,0,0],[1,9,7,5])]); print(7+0,7+3)'` 输出 `[1, 1]` 和 `7 10`。历史权重数值、输入可达集与该任务 tensor/collective 身份未绑定；必要信息须对固定权重、可达输入、目标精度与接收初态逐项证明。

## 预乘权重的反例与条件工作量

即使只考虑一维、BF16 round-to-nearest-even，并假设两条线性都按这一边界舍入，提前将两组权重相乘并以 BF16 保存也不普遍等价：取 BF16 `x=129/128`、`a=65/64`、`b=75/64`。原流程 `bf16(bf16(x*a)*b)=77/64`，预乘流程 `bf16(x*bf16(a*b))=153/128`，不同。中间 `bf16(x*a)=131/128`、`bf16(a*b)=19/16`；这是源码候选约束的数学反例，不声称真实 kernel 的具体累加与 rounding 就是此简化标量模型，或运行中出现这些值。保留不同权重/中间精度的其它实现要分别证明。

在上述 local 1-group、无特殊稀疏结构条件下，每 token 原两级逻辑点积含 `4096×1024+1024×5120=9437184` 乘加；预乘后 dense 4096→5120 为 `20971520` 乘加，即 `20/9` 倍。若两套权重及预乘权重都用 BF16 存储且无共享/压缩，原两矩阵是 18 MiB/本 rank，合并矩阵为 40 MiB/本 rank。这只是固定 shape 的算术与条件存储计数，非实际显存增量、HBM 流量、速度或不可超越性能界；改变参数表示、缓存与 tensor parallel 的方案须重新计算。

## 历史采样与后续

既有 `f07-q-kv-streams.json` 的 EP8 层24 run2，`MatMulV2` task1283 同流输出 BF16 `[24,5120]`，输入 BF16 `[24,1024]` 和 `[5120,1024]`，与 local wo_b 的逻辑形状相容，不能单靠 shape 证明是该模块或该轮权重实例。stage170 由 ZCode `deepseek-flash` 对同一历史窗口只读查询。独立核验 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-170-f16-layer24.exit-code` 为 0，`.verified.json` 的 `model_verified=true`；结果 `f16-layer24-run2-evidence.json` SHA256 `993d145ee0ed90284ebd76d2b33f4f1224d87f554c4cdd13bb6f4616bf591f97`，查询脚本 SHA256 `8e25a458493229ab1e6c2de61f2b8f30748430cd855feb3231ea4dabf5b95d80`，输入 EP8 CSV SHA256 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 与冻结清单相符。没有启动 NPU。

| 事件 | stream/task | 相对窗口起止 µs | 关键 shape / dtype |
|---|---|---:|---|
| SparseFlashMla | 89/1277 | 706.886–751.985 | 输出 `[3,64,512]` BF16 |
| InplacePartialRotaryMul | 89/1278 | 753.045–756.865 | `[3,1,64,512]` BF16 |
| TransposeBatchMatMul | 89/1282 | 790.585–841.144 | `[24,1,4096];[1,4096,1024]` BF16 → `[24,1,1024]` BF16 |
| MatMulV2 | 89/1283 | 841.164–855.943 | `[24,1024];[5120,1024]` BF16 → `[24,5120]` BF16 |
| hcom_reduceScatter_ | 88/301 和 stream/task N/A | 858.023–874.463 | 两条记录同一时间区间、shape N/A |
| HcPost | 89/1287 | 875.163–882.503 | `[1,3,5120]` BF16 等 → `[1,3,4,5120]` BF16 |

两投影任务的维度、dtype 和同流顺序与 local wo_a→wo_b 相容；以这些 shape 能收紧上述条件算术，却不能证明实际张量地址、权重、runtime 实例、真实 token 行或调用栈。两条 reduceScatter 记录的区间完全相同，分别为汇总样式和 `AivKernel`，不可将它们当作两个独立 collective 累加，也不单凭此断言二者必然是同一个 event；跨流事件、通信对端与该任务是否属于此输出都未绑定。HcPost 时间靠后与源码调用相容，不足以证明 collective 或投影输出就是它的输入。该任务时长、近邻空隙或理论算术均不能作为性能收益。


### 同卡相邻任务的条件稠密工作量对照

固定上述 EP8、第 24 层、run 2 的两条 BF16 记录，`TransposeBatchMatMul` 的导出时长为 **50.559 μs**，后面的 `MatMulV2` 为 **14.779 μs**，分别在同一个 model 49、stream 89，前者结束后 0.020 μs 后者开始。只按导出物理 shape 所描述的**标准稠密点积**计算：前者 `24×4096×1024 = 100,663,296 MAC`，后者 `24×1024×5120 = 125,829,120 MAC`；前者逻辑工作量是后者的 **80%**，导出时长却是 **3.42 倍**。若用这些逻辑 MAC 除以各自任务时长作规范化对照，分别约 **1.991、8.514 TMAC/s**。这只是一个已选窗口的两个不同算子；不是实测硬件指令数、芯片峰值、整机利用率、端到端收益或对其他 rank/轮次的推断，也不证明二者处理了 24 个有效 token 或直接传递了同一 tensor。

这一对记录与“两个任务具有相同单位逻辑 MAC 耗时”这一狭义模型不一致，却**无法区分**转置与布局、不同矩阵维度及 tiling、实际有效工作、数据来源与缓存、资源并发、编译路径等解释。需要绑定两个算子的实际权重和输入 buffer、kernel 实现/tiling、真实行数以及相应的数据搬运/缓存指标，才能继续区分。当前 F16 源码中的 `perm_x1=(1,0,2)` 还意味着不能仅凭物理输入 `[24,1,4096]` 就断定内部执行了 24 个 M=1 的 GEMV；实际历史二进制和参数尚未闭合。没有观察到足以单独解释这两个时长比的条件证据。

复算仅需读取已有 `f16-layer24-run2-evidence.json`（SHA-256 `993d145ee0ed90284ebd76d2b33f4f1224d87f554c4cdd13bb6f4616bf591f97`），按同 stream、model 和 task ID 1282/1283 选择两行，以保存的相对 ns 端点相减求各自时长；上述工作量是标准稠密假设下的算术。单卡内计算，没有跨卡时间戳相减，也没有新 NPU 负载或实现修改。

### 16 rank、三轮的第 24 层同形状对照（stage233）

ZCode stage233 只读查询 16 份冻结的 kernel CSV 和同 rank 的 3 个第 24 层窗口。**48/48 窗**分别唯一匹配到一条前述 `TransposeBatchMatMul` 和一条 `MatMulV2`，两者均为 model 49、同一主 stream、BF16 和上节列出的完整输入输出物理 shape；每对按该卡时间轴先后且完整位于窗口。表中耗时为每轮 16 卡的中位数与范围，倍数为**每卡先计算前者时长 / 后者时长，再取中位数和范围**；各列的中位数不必来自同一张卡，单位为 μs：

| 内部轮次 | `TransposeBatchMatMul` | `MatMulV2` | 逐卡时长比 |
|---:|---:|---:|---:|
| 1 | 51.549（50.379–53.758） | 14.530（13.840–15.300） | 3.560（3.316–3.733） |
| 2 | 51.499（49.779–54.959） | 14.750（13.960–15.780） | 3.507（3.241–3.731） |
| 3 | 51.910（50.379–53.439） | 14.870（14.040–16.060） | 3.488（3.220–3.642） |

每一对的导出先后间隔为 **0.020–0.040 μs**，但时间相邻不证明前者输出就是后者输入。两个任务的标准稠密逻辑工作量仍按上节固定物理 shape 分别为 100,663,296 与 125,829,120 MAC；这些只是条件算术，不是运行时已执行 MAC，也不证明 24 行都是有效 token。每一轮、每张卡都观察到前者更慢且同一组物理形状下其逻辑 MAC 较少，这排除了“先前单卡结果仅因该次采样偶然反向”在这 48 个窗口内的狭义解释；不能因此推断二者实际计算量、缓存状态、布局转换或调用路径相同，更不能给出原因或可兑现加速。不同轮或 rank 不保证同一请求、同一张量值或相同有效工作。

独立复核了 48 个 `(rank,run,layer)` 唯一性，16 个 CSV 哈希与冻结 manifest 一致；逐窗检验 model、主 stream、两个类型、完整 shape/dtype、窗口包含、时间顺序和时长与端点一致。EP8 run2 的两条端点与 stage170 历史记录逐条一致。结果 `f16-layer24-all-rank-run-pairs.json` SHA-256 `26d0ab676539995816c26f697d0702941e7284983f020ff9f221f86c3d81840f`；查询脚本 `query_f16_all_rank_run_pairs.py` SHA-256 `dc2b8a16a1e8eee94550135dc2625211b2d5b40bbeb894aae11feb9c031b5538`；wrapper `run_f16_all_rank_run_pairs.py` SHA-256 `39a925ce62aa9c60e58efc7f960d774eae71a7da6549a1445e48bb3c601df7c3`。本机验证 stage233 exit code 为 0、`model_verified=true`（ZCode DeepSeek）；查询仅只读已有导出，无新 NPU 负载或实现修改。

## 单组时的逻辑轴交换与二维投影候选

冻结 `dsa_cp.py:1754–1795` 中普通分支传给 `npu_transpose_batchmatmul` 的参数明确为 `bias=None`、`scale=None`、`perm_x1=(1,0,2)`、`perm_x2=(0,1,2)`、`perm_y=(1,0,2)`、`batch_split_factor=1`。该调用本身已把指定轴置换和 batch matmul 表达为一个接口，不能把这些参数直接理解为实际执行了三次内存转置；设备实现、图改写和内部工作区未知。

**若**实际进入此分支、`o_proj_groups=1`，对逻辑输入 `X[T,1,K]`、权重 `W[1,K,R]`，轴置换后的 batch matmul 有一个组、T 行；按下标其结果为 `Y[t,0,j]=Σₖ X[t,0,k] W[0,k,j]`。因此从数学关系可研究将该组直接作为二维 `X₂[T,K] @ W₂[K,R]`，再提供按原消费契约解释的输出。若原 X 是连续的，那么交换 T 轴和长度为 1 的组轴可通过视图索引表达；没有这一存储条件时也要按实际 stride 正确读取，不能假设连续或零拷贝。组数不为 1、full weight、其他布局和不同分支都需要另行推导，不能泛化。

冻结 `dsa_cp.py:1666–1682` 对已经为三维、首轴等于组数的 `wo_a.weight` 直接返回该权重；非三维权重或其它三维布局则走 reshape/transpose/permute 或报错等不同路径。另一个当前快照 `ops/linear.py:107–131` 在其 unquantized 权重加载后处理条件满足且权重为二维时，对 wo_a 进行 reshape、transpose 和 `.contiguous()` 形成三维权重；这是该加载分支的源码行为，不证明历史调用使用它、权重完整加载/切换发生的时间、每次推理有没有真实复制或 kernel 内部不会重排。所涉文件 SHA：`dsa_cp.py` `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305`；`ops/linear.py` `7654777d9252fc6a3cc2c95f92b67a42410a0de24a5912ef20e807cf64ab74e6`。

stage233 的 48 条 transpose 任务均导出物理 `[24,1,4096] × [1,4096,1024] → [24,1,1024]`，与这个单组模型相容；没有调用/存储地址和 runtime 语义绑定，不能认定 48 条全部执行上述 CP 分支。若探索二维 GEMM，也须验证动态 token/padding、权重内容和 strides、原接口与替代算子的累加顺序、BF16 输出边界、异常值行为以及后续 `wo_b` / rank collective；数学等价不保证逐位等价。在保留两级直接点积及其数值契约的候选中，下游必须获得对应输出所需的完整贡献；不能仅因单组就假定第二投影的所有依赖消失，但也不能仅凭 shape 认定所有中间特征都不可绕过。

这为 48 窗的高耗时提供了**可检验的布局/后端选择假说**，但现有物理 shape 不证明原 kernel 实际重排数据、每次调用转换权重、24 行全有效、二维方案可用或更快，更不能把约 3.5 倍时长差归因于转置。仅需优先寻找已有的匹配图节点、kernel 源/tiling、输入与权重 stride/layout、有效行数及同条件内存/硬件指标；没有则保持未知。本次只做源码与索引分析，没有 NPU 负载或实现修改。

### 两类任务的已导出核心字段（stage234）

stage234 按 stage233 保存的同卡同窗 model、stream、type、task ID 和精确起止时间，从 16 份已哈希校验的 CSV 中唯一提取 96 条任务；每条还核验物理 shape 与 dtype，全部所请求列均存在。本机独立核对 48 对与窗口和 stage233 的时间、type、shape/dtype 一致，并逐卡重算以下统计。每格为该轮 16 卡分别取的导出值中位数；比例字段已乘 100 转成百分数，不同列的中位数不必属于同一张卡：

| 轮次 | 前者 / 后者 AIC MAC ratio | 前者 / 后者 AIC scalar ratio | 前者 / 后者 AIC MTE2 ratio | 前者 / 后者 `cube_utilization(%)` |
|---:|---:|---:|---:|---:|
| 1 | 10.15% / 11.70% | 55.70% / 32.95% | 32.85% / 61.80% | 21.346% / 81.339% |
| 2 | 10.25% / 11.80% | 56.05% / 32.70% | 32.75% / 61.85% | 21.121% / 81.373% |
| 3 | 10.20% / 11.70% | 55.60% / 32.15% | 33.25% / 62.40% | 21.071% / 81.670% |

逐卡比较，48/48 对均为前者报告的 MAC、MTE2 和 `cube_utilization(%)` 小于后者、scalar ratio 大于后者；这比单张卡差异有更广的**导出字段一致性**。前者的 AIC 总周期三轮中位数为 472760.5、469515.5、472867.5，后者为 519604、516520、519755；这只是导出的数值，其每核/跨核聚合及与总任务时间的关系尚未证明，不能据它判定两任务总工作相同。两种任务 96 条的 `aiv_total_cycles` 均报告 0，**不是证明 AIV 未工作**；此次没有缺列，但零值可能与核心类型、采集或导出口径有关，不能当作缺失或直接推断实际活动。

静态导出字段在全部 48 对保持一致：两者 `Accelerator Core=AI_CORE`、`Mix Block Num=0`、输入格式 `ND;ND`、输出 `ND`，输入与输出均为 BF16；前者 `Block Num=24`、后者为 `23`，`OP State` 都是 `static`。前者 Name 为 `aclnnTransposeBatchMatMul_TransposeBatchMatMul_TransposeBatchMatMul`，后者为 `aclnnMatmul_MatMulCommon_MatMulV2`；它们是不同的**导出名称**，不能仅据名称证明具体编译二进制、实际微指令、有效 tile 或精度行为，也不能把一个类型的核心字段代替整芯片利用率。

既存的 [MindStudio Insight 字段说明快照](https://github.com/Ascend/msinsight/blob/efb28234772093b0cabc3c538e71375c957540b2/docs/zh/user_guide/system_tuning.md) 将相关 ratio 描述为对应单元时间与总时间的比例，但该文档与本次 exporter 的实现、cycle 分母、跨核聚合与重放映射尚未闭合。EP8 的 session 元数据记录 torch_npu `2.10.0.post7`、CANN `9.2.0`，这不等于已经验证此次每条 counter 的计算规则。先前针对另一 BF16 矩阵任务的 `cube_utilization(%)` 候选公式吻合只是另一任务的对照，不能直接证成本任务的具体公式。由于 scalar、MTE2、MAC 和 cube 指标统计口径不完整，差异只为后续查对应实现、布局、tiling、数据流量与指令路径提供优先线索；**不能将它们相加、将 scalar 高归因为指令瓶颈，或将 MTE2 较低归因为 HBM 带宽问题**。同卡相邻也未证明中间 tensor 身份或有效 token 数，无法给出可实现收益。

复算 `python3 agents/fusion/recheck_f16_fixed_counters.py`，该脚本验证四个输入 SHA、48 对唯一性、逐行与窗口及 stage233 锚点的 model/stream/时间/type/shape/dtype 一致，保留 missing 和 zero；输出 `f16-layer24-fixed-counters-summary.json` SHA-256 `7b975261e7bbe6f82b5ba09f80754cfdb18c78752876a3e0529dbe021a3da5ae`，脚本 SHA-256 `3dbd1d9103844c0d7aa1c367f24eddfe7fc48d294810b6e6b7cba0fa0c804eff`。stage234 原始 96 条 `f16-layer24-fixed-counters.json` SHA-256 `70f3d84b682e54766bc0b5e5ba8b04cd4fec8bee944029167e1f5e9e84f8b0b0`；本机核验 ZCode exit 0、`model_verified=true`、模型 `deepseek-flash`。只读已有数据，无新 NPU 负载。

### F16 的 Cube 字段与耗时并非独立证据

对 stage234 已保存的 96 条 F16 记录，复用各自 `aic_total_cycles`、整数纳秒起止和本卡 `DeviceInfo` 的 `aic_frequency=1800 MHz`、`ai_core_num=24`，按 [CANN 910beta2 官方 op_summary 字段说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/devaids/Profiling/atlasprofiling_16_0067.html) 所列候选关系复算：

`cube_utilization(%) = 100 × aic_total_cycles / (frequency_MHz × core_count × task_duration_μs)`。

使用同 rank、同 run、同任务的导出端点作为 duration，第一任务 48 条的最大绝对误差为 **0.0004831 个百分点**，第二任务 48 条为 **0.0004890 个百分点**，均小于导出字段最后一位的半个单位。示例 EP0/run1：第一任务 525156 cycles、53.519 μs，候选公式为 22.7141555%，导出为 22.714%；第二任务 521777 cycles、14.520 μs，候选公式为 83.1829979%，导出为 83.183%。两个公式分子接近时，分母的任务时长增长本身就会降低这个比值。

这对 96 条记录构成公式的强数值一致性证据：该字段并非独立于任务时长和 AIC 总周期的第二项证据，公式本身也没有 shape、逻辑 MAC 或 FLOP 输入；不能将其当作矩阵算力峰值效率、慢因或硬件不可超越上界。total cycles 是另一个导出计数字段，但**未对本次 F16 记录做 raw PMU 的同事件绑定**；公式吻合不单独证明实际 exporter 代码、各次计数来源或 replay 关系。且上述官方文档为 910beta2，EP8 这次 profile 信息自报 `torch_npu=2.10.0.post7`、`CANN=9.2.0`，同名字段的实际导出版本没有闭合；不能把该公式从数值一致提升为已确认的历史版本实现。频率元数据也没有证明是所有合法运行状态的硬上限。对每卡的起止只在本卡时钟内做减法，未跨卡比较绝对时间。

只读复算：`python3 agents/fusion/recheck_f16_cube_metrics.py`。脚本核对固定输入 SHA、16 张卡 metadata 和 96 条样本唯一性，只向标准输出写 JSON；记录结果保存在 `agents/fusion/f16-cube-formula-check.json`。输入 `f16-layer24-fixed-counters.json` SHA-256 `70f3d84b682e54766bc0b5e5ba8b04cd4fec8bee944029167e1f5e9e84f8b0b0`、`profile-device-capacity-info.json` SHA-256 `f0f1e014ce728cf1c81b4f6f76a4c869c5ca9ba77340f794c2fac2d436d345e6`；脚本 SHA-256 `bfccb528d328c414e6aa86009214498a29aa83e96e927db7d1dead47a26d9e13`，结果 SHA-256 `4ab8b84a4f4b73b784b8dc71e45da370c2b73a6203878feb80c1659cb4df0dcd`。本次未运行 NPU 或更改推理实现。

上述 96 条公式复算由 ZCode stage235 在本机执行，exit 0、model_verified=true，输出 SHA-256 `4ab8b84a4f4b73b784b8dc71e45da370c2b73a6203878feb80c1659cb4df0dcd` 与保存结果一致；无 NPU 操作。

### 单组候选的接口、stride 与权重方向核对

华为 [CANN 社区版 9.1 的 aclnnTransposeBatchMatMul 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnTransposeBatchMatMul.md)将 A2/A3 列为适用，说明 x1/x2 都是 **3 维 ND 输入且支持非连续 tensor**，x1 的 perm 可交换前两维，A2/A3 的 permX2 为 `[0,1,2]`。它支持的是有相应 stride 的接口输入，不保证内部不整理数据或无 workspace；这不是可直接接收二维输入的同一接口。若探索二维方案，需要选择并核查**另一**二维 matmul 接口及它对实际输入 stride、权重 stride、dtype、workspace、数值契约和图模式的支持。官方说明是 9.1 版本，历史采样自述 CANN 9.2，未证明历史库版本、PyTorch wrapper 与该文档完全匹配；不能把公开支持列表当作已观测 kernel 的行为。

在仅限 G=1、PyTorch strided tensor 语义且非 packed 分支的条件下，若 X 的 shape `[T,1,K]`、stride 为 `(sT,sG,sK)`，其索引 `X[t,0,k]` 的地址是 `base+t*sT+k*sK`；交换单元素组轴后的索引 `X′[0,t,k]` 同样是这个地址，无需假设输入本身连续。可通过对 singleton 轴的视图操作得到逻辑 `[T,K]`；是否能被目标二维接口高效接受、不被该接口内部重排，仍未知。同理，对标准 strided 输出 `[T,1,R]`，挤掉单元素轴有索引保持的 `[T,R]` 视图；`dsa_cp.py:1794` 使用 `.reshape(num_tokens,-1)`，**这里的代数可视图性不是历史底层实际零拷贝或输出 stride 的证据**。

该单组条件下，源码 `_get_batched_wo_a_weight` 返回逻辑 `W[1,K,R]`，`perm_x2=(0,1,2)`，故正确的二维乘法方向是 `X₂[T,K] @ W[0][K,R]`。不能将 `W[0]` 不做转换就当成 `torch.nn.functional.linear` 的 `[out_features,in_features]` 权重：用本条件的 K=4096、R=1024 时，`linear` 会把末轴 1024 当输入维，与 X 的 4096 不匹配；应选择接受 `[K,R]` 的 matmul，或以满足精度和 stride 约束的方式提供转置后的 `[R,K]`。这只是接口方向反例，不说明当前实现存在错误，也不说明转置视图被硬件接受时无复制。

`bias=None`、`scale=None` 和 perm 只在上述实际分支条件下成立；保存的 48 条设备任务输出是 BF16，而接口层 dtype 及底层累加算法仍需与当次 wrapper/实例绑定。公开 [Ascend TorchAir 固定版本 converter](https://gitee.com/ascend/torchair/blob/89b42adf4dd499bb1de7c6ab9ba91263d9ac0096/python/torchair/_ge_concrete_graph/ge_converter/custom/transpose_batchmatmul.py)将对应 op 映射到 GE TransposeBatchMatMul，并显式指定 `enable_hf32=False`；它是另一个具体版本的图转换实现，不能证明本历史图走了该 converter，尤其不能据此比较二维路径的 BF16 累加/舍入。二维替换仍需核对输出 BF16、特殊值和误差契约以及 `wo_b`/collective 的对应输入；本次没有修改实现或运行 NPU。

### 单次投影改变归约树的精度反例

上面的单组二维候选与三维调用在实数代数下使用相同点积，但这还不能保证逐位输出一致。以下反例针对**一个投影中的 K 维累加顺序**，不同于前面将两级权重预乘的反例：假设两个接口都以 BF16 输入、FP32 最近舍入执行点积并最终输出 BF16，某行前三项 `x=[1,1,1]`、某个输出列对应权重为 `[2^40,−2^40,1]`，其余 4093 项均为 0。这些输入值均可用 BF16 表示。

- 归约树一：`round32(round32(2^40 + (−2^40)) + 1) = 1`。
- 归约树二：`round32(2^40 + round32((−2^40) + 1)) = 0`，因为 FP32 在该量级无法保留加上的 1。

最终 0 和 1 均可精确表示为 BF16，故两种重组在该假设域中会输出不同的 BF16 值。这证明数学索引等价不足以保证不同归约实现逐位等价；并不证明历史三维内核和拟议二维内核分别采用这些归约树，或固定模型权重和历史输入出现该例。实际精度要求若允许误差，还需核对误差容限和可达输入范围；如果要求逐位等价，须证明两接口的输入/累加/舍入/非有限值与具体执行路径契约满足该要求。

此外，是否能使用二维路径仍取决于实际单组分支、X 和 W 的 storage offset/stride、padding 和下游消费者，且接口支持非连续输入并不证明无隐式复制。相同逻辑 MAC 数不证明相同执行时间或物理性能界。反例可离线复算 `python3 agents/fusion/recheck_f16_reduction_witness.py`，脚本 SHA-256 `011178030d24cb963e5c25474029d856a040896f6bd811b238e9ba3070c52061`；它用 CPU 上的 binary32 转换检查两种归约结果，不模拟 Ascend kernel，没有启动 NPU。

### EP8 两条固定任务的 DB 属性（stage236）

对 EP8、第 24 层 run2，按 model 49、stream 89、task ID 1282/1283 和各自精确纳秒端点在同一 profiler DB 的 `TASK` 中查询，前后任务各唯一匹配一行；其 `globalTaskId` 分别为 19471、19472，各对应一个 `COMPUTE_TASK_INFO`。独立核对两条时间、task ID、model、stream，以及计算信息的导出 name/type、ND 格式、BF16 和输入输出 shape，均与 stage234 对应 CSV 行相符。这是该 DB 内对固定事件的精确关联，不是对 48 对全部事件逐条核实。

第一条关联的属性字符串含 `perm_x1`、`perm_x2`、`perm_y`，限定白名单中的导出数值分别为 `[1,0,2]`、`[0,1,2]`、`[1,0,2]`，与本报告所引冻结 `dsa_cp.py:1784–1793` 的三个对应参数一致；属性键还包含 `batch_split_factor`，stage236 本身没有输出其值；后续 stage237 的单事件核查见下文。第二条属性字符串含 `transpose_x1`、`transpose_x2`，导出值分别为 0、1，与对 `[24,1024]` 和 `[5120,1024]` 形成 `[24,5120]` 的标准转置矩阵乘口径相容；具体 exporter、算子 ABI 和历史执行路径仍需核实。

每个 globalTaskId 在 `TASK` 表中各出现 **7 行**，表明该 ID 单独不能作为这次固定查询的唯一定位键；查询没有核实这 7 行是否均为不同 replay，因此不能据行数断定有 7 次重放。两个任务均使用 connectionId 184330，不能仅凭共同 ID 证明张量生产消费关系。两条 `COMPUTE_TASK_INFO` 均记录 `gridDim=5`、`blockDim=5`，编码语义未知，不把它们解释为实际 tile 大小；属性键中没有显式 stride 或 tiling key，指定 DB 的表名检索也没有匹配的 tiling/kernel/stride/binary/tensor 表名。这只限定本次两个任务的所选字段和表名，不排除其他表、原始采集或其他来源保存相关信息。

这些属性强化了 **位置、形状和轴参数与 F16 单组源码相容**的线索，仍没有证明该源码实例实际调用了这两个任务、两者共享 tensor、实际做了物理拷贝、历史二进制与源码一致，或某种布局解释了时长差。此查询将属性原文替换为哈希，只输出有限白名单的纯数字轴/转置值；未读取 tensor/权重内容、未运行 NPU。

结果 `f16-ep8-task-metadata.json` SHA-256 `4aecbfa9184d0836136b3797a53f3e5648c3160965d4b5fc8535758b47ef450f`；查询/包装脚本分别为 `ecdb84bac1753084cdbbe39ab06f4fb09ab966f51307e0b0d483a4a5cbe98c23` 和 `bc8582ad94d428667f8192ebea7d22112ff6694b80cfdf93502eb360ae9aa5af`；EP8 DB 冻结 SHA-256 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`。本机已核验 ZCode stage236 exit 0、`model_verified=true`、模型 `deepseek-flash`；固定源码 `dsa_cp.py` SHA-256 `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305`。

### EP8 单任务 `batch_split_factor`（stage237）

沿用 stage236 的固定 EP8 第 24 层 run2 第一条任务，stage237 以同一 DB SHA、model49、stream89、task ID1282 和精确起止时间查询到 1 条 TASK，且其 `globalTaskId=19471` 对应 1 条 COMPUTE_TASK_INFO；该信息的属性字符串 SHA 与 stage236 的 `319e40b4829a0df03cefc29dba88e2ed16da7e4e29c8de6a44416cad05d12eeb` 一致。属性中 `batch_split_factor` 恰好出现一次，按纯整数格式解析的导出值为 **1**，与冻结 `dsa_cp.py:1792` 的参数值一致。连同 stage236 的三个 perm 值，增加了此任务和该源码调用参数相容的证据。

结论仅适用于这个 EP8 单窗口的指定事件；19471 在 TASK 表中出现 7 行，不能以该 ID 单独定位某次执行，也不能据行数断定具体重放次数。相同导出属性不证明历史运行使用该源码实例、实际 tensor 或权重身份、所用编译二进制及 tiling、是否进行了物理数据拷贝或可实现的性能。未知的属性编码/运行契约也不应由数值吻合自动补成事实；其他 47 个窗口没有逐条验证此键。

独立检查 stage236 结果中 DB 哈希与属性哈希、任务行/关联数，以及 stage237 的唯一命中、整数值；用 Python AST 核对冻结源码调用的 `perm_x1=(1,0,2)`、`perm_x2=(0,1,2)`、`perm_y=(1,0,2)`、`batch_split_factor=1`。结果 `f16-ep8-batch-split-exact.json` SHA-256 `ba1875858cfdd44a5420d87d28ae039958c3b765752177cd28f4ed2c7802bb6b`，只读查询和 wrapper SHA 分别为 `abe6bd64583d4f8bb7f85d82de39516d26db2ea6e2bdb172dcb29376cb1fb4b8`、`e4a1760e13c2b0e97c57d66c75f2128e785a4dc1ad1c2f285d8f1b584095aab1`。本机核验 ZCode stage237 exit 0、`model_verified=true`、模型 `deepseek-flash`；无 NPU 工作或实现修改。


### GEMM + reduce-scatter 公开接口：跨 output 边界的条件候选

冻结的 `ops/linear_op.py:156–214` 含另一个模型 Kimi 的 `KimiOProjMMReduceScatterOp`，在特定检查通过时以 BF16 输入、转置权重和 `comm_mode=ai_cpu` 调用 `torch_npu.npu_quant_mm_reduce_scatter`；否则先 linear 再 SP reduce-scatter。该类的实例选择、支持的 shapes 与数值契约不能直接推广到 DeepSeek。DSV4 `wo_b` 在该快照的选择代码中指向不同的 DSV4 行并行实现，而 SP 分支的 reduce-scatter 在 self-attention 返回之后（见 `TENSORMOVE-O-PROJ-LINEAR-INTERFACE.md`）。因此要研究这个跨边界复合调用，必须重构输出接口，确认原未归约 output 没有其他观察者，同时保持通信组、所有 rank 归约顺序、输出分片与 padding、图生命周期和 BF16 舍入；不能仅把现有 GEMM 调用替换成 Kimi 调用。

昇腾 [CANN 9.1 aclnnMatmulReduceScatterV2 官方接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnMatmulReduceScatterV2.md) 列 Atlas A3 支持 BF16 matmul 加 reduce-scatter，并在 A2/A3 产品限制处描述通信模式 `aiv`。另一份 [CANN 8.5 官方文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/aclnnMatmulReduceScatterV2.md) 描述 `aicpu` 和 `aiv`；这些是不同文档版本的接口说明，不能推断历史 CANN 9.2 中实际支持的模式，也不能依据 Kimi 源码中 `ai_cpu` 的参数拼写推断其底层映射。版本对应、安装 API 可用性、返回精度和重叠实现均未证。

EP8 的选中投影记录形状为 BF16 `[8184,1024]×[1024,5120]`、TP 口径 8 时 8184 可被 8 整除、K=1024；这些仅是与二维及部分形状条件相容的静态算术。它不能证明该记录的实际 collective rank 数、输入 stride、所有 rank 均有一致 shape、output 可直接替换或算子能执行；样本范围和 tensor 绑定见 `ATTENTION-PROJECTION-EP8-MODEL.md` 与 `EP8-SP-REDUCE-SCATTER-FLOW.md`。通信与计算可能已经有重叠或争用；未测得收益，不将通信或 GEMM 时间直接相加为收益。

### GEMM + reduce-scatter 的进一步接口与资源约束（CANN 9.1 文档）

[昇腾 CANN 9.1 `aclnnMatmulReduceScatterV2` 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnMatmulReduceScatterV2.md) 的参数表规定 x1 为二维 ND、不支持非连续 tensor；x2 支持二维轴转置产生的非连续 tensor，但不支持任意其他非连续布局；output 不支持非连续 tensor。文档一般参数表还规定 `commTurn=0`、`streamMode=1`。其 A2/A3 专项说明对 BF16 指定 `commMode="aiv"`、bias 和 x1/x2 scale 均为 null，并说明通信任务使用 AI Vector 核。**这些是 9.1 公共接口的合同，并非历史 9.2 已加载接口或 torch wrapper 的实测行为。**

这收紧 F16 的具体设计条件：若当前 `wo_a` 输出 reshape 后不是目标接口接受的连续 x1，则需要在上游直接产生合格布局或另行整理；既有样本的 shape/格式字段没有 stride 和 storage，因此还无法判断是否满足，也不能假定增加 copy 或其成本。原未归约 output 若有消费者或不同形状的 output 契约，仍不能直接替换。9.1 的 `commTurn=0` 不能解释成没有内部切块或不可重叠，只能说明调用方不能通过该参数设置任意分块数；AIV 通信与可能同时运行的其他 AIV 工作存在资源竞争的**可能**，实际时间线和硬件活动尚不足以证明竞争或可隐藏比例。文档其他段落对 output 的描述出现 AllGather 字样，与开头 ReduceScatter 公式不一致，本记录采用函数名及公式，不用该处文字推导实际通信。

这不是新的独立融合项，也不证明该 API 能用于历史实例；历史接口版本、各 rank 的分片、实际 input/output stride、tensor 身份、同步、BF16 舍入和性能仍需证据。

### 投影 partial 的 BF16 边界与 reduce-scatter 融合反例（2026-10-30）

既存 EP8 单次诊断通过 host 栈与唯一 flow，分别关联 `vllm::unquantized_gemm` 的 BF16 `[8184,1024] × [1024,5120]` 矩阵任务、`aclnnInplaceCopy`，及之后 host 范围中的 BF16 `[8184,5120] → [1023,5120]` reduce-scatter 调用；详见 [host 绑定](../diagnosis/TENSORMOVE-EP8-PROJECTION-HOST-BINDING.md)。三者的 tensor/storage 和 collective 组尚未逐对象绑定，不能把相邻 shape 和 flow 升级为真实数据边。冻结源码在 SP 且相关调用路径成立时先产出 partial `output`、经 copy 写入该 attention 自定义接口的 caller output，再由后续分支 reduce-scatter；partial output 与 custom op 的可见性和其他消费者不能先假定不存在。

对候选 `MatmulReduceScatter`，仅以一个**可证伪的精度模型**说明为什么“把两个操作融合”还需检查中间舍入：令每个 rank 的局部点积由 BF16 输入和权重、精确实数乘加产生，原路径先把各 rank 的点积按 BF16 round-to-nearest-even 舍入，再按实数相加并以 BF16 存结果；候选路径假设先保留精确 partial、跨 rank 相加，最后才舍入 BF16。对某一输出元素，rank A 的两个非零乘积为 `1` 与 `1/256`，rank B 的一个非零乘积为 `-1`，其余 rank/列可补零；这些因子都可以由 BF16 表示，且可嵌入 K=1024 的假想二维输入。

- `1+1/256` 正好处于 BF16 `1` 与 `1+1/128` 的中点，最近偶数舍入为 `1`。原路径得 `bf16(1 + (−1)) = 0`。
- 假设的后舍入路径得 `bf16((1+1/256)+(−1)) = 1/256`，结果不同。

这表明在该明确模型中，越过原 per-rank BF16 中间边界改变跨 rank 归约位置，不能凭实数线性代数或相同输入输出 shape 声称逐位等价。它**不说明**上述历史样本的输入取过这些值、实际 GEMM 或通信的精确累加顺序，也不说明公开融合接口使用假设的后舍入语义或必然失败；如果用户允许有界误差，需按实际误差合同评估，不能由反例否决所有实现。中间部分结果的状态观察、scale/bias、通信组及其顺序、padding、各 rank 参与、输出布局及 event/graph 生命周期仍需独立核实；数值正确也不证明通信被隐藏或产生收益。

本机以 `Fraction` 独立验证中点与两个结果，未运行设备；源码与既有 trace 的引用及版本边界沿用本节和 host 绑定文档。这个反例是已有 F16 GEMM+reduce-scatter 条件候选的精度约束，不重复登记新的融合收益。

本机 ZCode 在 `zcode/f16-gemm-scatter-rounding.zcode.json` 独立核对 BF16 中点、偶数选择及两种结果，与本机 Fraction 验算一致；`1+1/256` 是精确点积中间值，不是可表示的 BF16 单个输入。该复核仅针对上述抽象数学，不验证 NPU 内核或真实样本。

### EP3 第 36 层：两级输出投影的主机栈与设备记录绑定（2026-10-30）

对已存 prefill 摘要 `db852bff…`、EP3 decoder scope `2128-42`，复用先前逐任务 scope→flow→设备与 CSV 的严格筛选结果，另以设备 stream、task 和精确起点只读查询同次 profiler SQLite，并沿 `TASK.connectionId → CONNECTION_IDS → PYTORCH_API → PYTORCH_CALLCHAINS` 查主机记录。两个 TASK 各唯一，globalTaskId 在该 DB 各只有一条 TASK；两条 host API 的开始时刻与此前 trace flow 起点分别一致，且都处于已绑定的 decoder host scope：

| 部分 | 设备 task / stream | 已导出的 ND 输入 → 输出（均 BF16） | 主机 API 与调用栈定位 |
| --- | --- | --- | --- |
| 第一投影 | 41440 / 47 | `[8184,1,4096] × [1,4096,1024] → [8184,1,1024]` | `aclnnTransposeBatchMatMul`，栈含 `dsa_cp.py:1784 _forward_o_proj`、`dsa_v41_cp.py:224 _project_output` |
| 第二投影 | 41441 / 47 | `[8184,1024] × [5120,1024] → [8184,5120]` | `aclnnMatmul`，栈含 `dsa_cp.py:1663 _apply_wo_b`、`:1795 _forward_o_proj`、`:224 _project_output` |

同一个设备 stream 的前者结束 `1793241329812341895 ns`，后者开始 `1793241329812341915 ns`，间隔 20 ns；这是导出时间次序，不足以证明中间对象身份、真实资源空闲或可省时间。冻结 `dsa_cp.py:1760–1799` 的非 A5 量化分支依次调用 transpose-batch-matmul、reshape 和 `_apply_wo_b`，与已记录调用栈和物理签名相容；`model.py:899–902` 在条件 SP 分支中把 attention 输出送至 reduce-scatter 后的 HC post。设备记录分别绑定到两个相应 host API 的证据，比此前单凭 EP8 graph 窗口中的相邻 shape 更强，但这仍是**两个分别绑定的调用**，不是中间 tensor storage 的直接证明；历史运行所加载源码哈希、weight 对象、真实 `full_gather_wo_a_enabled`、layout stride、reshape 是否复制和该次下游 collective 身份均未绑定。

若上述实际分支、单组物理布局及真实张量依赖得到证明，可以研究第一投影分块生成 BF16 中间值、再供第二投影分块消费；保留两级直接稠密实现时，需覆盖其 1024 个中间特征的点积项并满足原中间 BF16 舍入与后续 SP 归约、HC 的可观察效果。若可证明固定权重、输入域或等价重排使某些特征/归约贡献不影响最终契约，可以采用相应简化，不能将具体实现的全部点积项或其等待升级为全实现域的必要下界。该候选的公开 API、stride、归约树与 BF16 精度限制见前文；图中形状相容和 20 ns 间隔均不证明可跨 kernel 发布 tile、无需额外同步、物化可消除或会提速。

复现 `python3 agents/fusion/run_f16_ep3_projection_provenance.py`：在已有 stage 容器对固定 EP3 DB 使用 SQLite `mode=ro`，不发起设备任务。数据库 SHA-256 `084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4`；既有任务提取 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`；冻结 `dsa_cp.py` SHA-256 `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305`；查询脚本 SHA-256 `452ab2a87a683f5f8c36d112624e671f9204f240352e350a37f13157bbf53a07`，运行脚本 SHA-256 `820410a2a1f3043982b6e777c01532397302c6be62dab6f03798de411229c80c`，聚合结果 `f16-ep3-projection-provenance.json` SHA-256 `996eb37b8741739d6edcc412736159194eea168f7298f7a46dbd57e8887071e1`。该结果含内部记录时间与标识，只保留在任务目录；本次没有改目标实现。

### EP3 同次投影之间的 reshape/view 元数据（2026-10-30）

延续上节 EP3 第 36 层两个已分别连接 host API 与设备记录的样本，在同一 host thread 的有界 3.5 ms 范围查询选定算子名、输入 shape/dtype 及调用栈：第一投影 `aclnnTransposeBatchMatMul` 后、第二投影 `aclnnMatmul` 前，记录到 `aten::reshape` 输入 BF16 `[8184,1,1024]`，其时间区间包含 `aten::view`，后者输入也是 BF16 `[8184,1,1024]`；两者栈均含 `dsa_cp.py:1794 _forward_o_proj`。之后有 `aten::linear` 输入 BF16 `[8184,1024]` 和权重 `[5120,1024]`，栈含 `dsa_cp.py:1663 _apply_wo_b` 与 `:1795 _forward_o_proj`；内部 `aten::matmul` 输入形状为 `[8184,1024]` 与 `[1024,5120]`。另一个更后的 `aten::copy_` 两输入都是 BF16 `[8184,5120]`，栈含 `dsa_cp.py:1798`，与源码可选输出缓冲写入相容。

这加强了“第一投影→reshape→第二投影→可选输出缓冲”是所选代码路径的解释，且 reshape 内观测到 view；它**仍不证明** view 的返回值被 reshape 返回、第一设备输出和 reshape 输入相同、reshape 输出和 linear 输入相同，也不证明整个算子路径未发生隐式复制。查询只覆盖所选 host 名称、同一线程及窗口；`inputShapes` 不包含输出、stride、storage/offset，不能由未发现 `clone` 宣布不存在复制。`aten::view` 的实际对象、PyTorch 版本对应行为、输出生命周期及运行中加载的源码与冻结快照关系仍未独立确认。

进一步从同次 EP3 `FRAMEWORK/torch.op_range` 流式检查同线程 `reshape`、`view`、`linear` 三个目标，固定文件大小 355,292,880 B，完整扫描 165,722 条记录、只解码有限窗口的 op name 与类型 5 的 tensor 元数据，不读取 tensor 值。各目标在粗略时钟换算后起止均有一个 10 μs 内的同名候选，三个候选起止均相对 DB 记录偏 4.720 μs；但**均没有类型 5 字段**，因此没有可比较的 TensorImpl、storage pointer、stride 或 offset。该转换和原始文件的独立冻结哈希未验证，候选不应当作严格身份绑定；类型 5 缺失仅限这三条，不代表其他记录不存在。无论是否后来得到相同地址，还需匹配分配代际、范围、offset 及生命周期；同址或相似 shape 单独不能证明依赖。

可复现：`python3 agents/fusion/run_f16_ep3_host_tensor_ops.py` 和 `python3 agents/fusion/run_f16_ep3_oprange_tensor.py`，均只读 169 stage 容器中已有文件、CPU 执行。前者数据库哈希 `084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4`；后者使用既存 `profiler_info_3.json` 转换信息，核验扫描前后文件状态，并设 25 秒读取上限（本次 completed=true）。主机查询/运行脚本 SHA-256 分别为 `541d9fa17a0ee7afa0cd64da2ac69500241b9a888ce224cb3199bfc19a150d45`、`befcaef9d0c9286d14429bb33fb2ac2b5cad31912b90e58f022a1cd2d35e6d57`，结果 `f16-ep3-host-tensor-ops.json` 为 `e6887b562816481a5b1e534cabba6fe08c4d302dcdf0bc46df4eacaea3aff015`；原始 op-range 查询/运行脚本分别为 `fdd78970febc94539d4d8eec2804cb241d0ba66e408eb607e0cedb2ebf9c2a69`、`132db367ccad4298736511fc44b69fe4e3020ac739518e75853d1f85dc307f94`，结果 `f16-ep3-oprange-tensor.json` 为 `570091ce1c3ed869523dad72cae772e79730a61ebba765f7b754dbd0ab82d9ad`。要完成身份判断，仍需同次且可严格关联两端的 tensor/storage 元数据（含 stride、offset、分配代际）或可靠的运行时数据依赖；当前只读材料未发现这些字段，不新增 NPU 采集。


### 不同已存样本下的条件化逻辑工作率对照

将上文两种形状只作为描述性对照：假设导出物理维度全部参与标准稠密点积，两级的逻辑 MAC 之比均为 0.8。逐对以 `(0.8 × 第二任务时长 / 第一任务时长)` 计算第一/第二的逻辑 MAC 每秒之比：24 行的 48 个已选 layer-24 窗口最小/中位/最大为 **0.214 / 0.228 / 0.248**；独立的 EP3 layer-36、8184 行两个已绑定调用栈任务的同一比值为 **0.920**（时长 270.900 / 311.400 µs）。

这说明“第一投影的逻辑工作率恒为第二投影约 23%”不能同时描述这些已选记录；但两组来自不同采集、层、负载和 profiler 状态，24 行各对的 host 路径与中间 tensor 未绑定，不能归因于行数或 tiling，也不能称其为硬件利用率。有效行数、布局、实现、缓存及资源竞争均未证明相同。下一步若已有等口径细分 counter、stride 或不同 token 数同路径样本，可优先核对；否则记为条件性比较，不能推算收益或理论界。两份聚合输入与哈希见上文 stage233 和 EP3 主机栈节，计算只依赖各对记录时长及固定条件逻辑 MAC 比，无新增 NPU 采集。


### 同采集同卡同层：两档 prefill 物理行数的配对

同一已存重型 profiler 采集的 16 个 rank、40 个 target 层，每卡每层各有两个不同 scheduler digest 的已选 scope，共 **640 对 / 1280 scope**。对每个 scope 严格筛一条 BF16 `TransposeBatchMatMul`，输入 `[M,1,4096]×[1,4096,1024]`、输出 `[M,1,1024]`；以及一条 BF16 `MatMulV3`，输入 `[M,1024]×[5120,1024]`、输出 `[M,5120]`。1280 个 scope 均各唯一命中一对；每对同设备轨道与 stream、任务 ID 连续且第一结束不晚于第二开始，格式均报告 ND，算子 core 均 AI_CORE、block 24。相邻不证明输出存储是后续输入；除下述 EP3 大行样本之外，本次未做逐次 host 栈绑定。

| 范围 | M | 选定第一/第二任务时长中位数 µs | 条件逻辑速率比中位数 |
|---|---:|---:|---:|
| EP0–7，同层配对小档 | 4760 | 177.426 / 189.436 | 0.856 |
| EP0–7，同层配对大档 | 8184 | 275.405 / 311.297 | 0.904 |
| EP8–15，同层配对小档 | 5144 | 180.786 / 185.556 | 0.821 |
| EP8–15，同层配对大档 | 8184 | 277.134 / 312.724 | 0.901 |

每个单项中位数独立计算，未必来自同一条；速率比先在同一 scope 计算，再取中位数。它假设所有物理行均做标准稠密工作，取 `0.8×第二时长/第一时长`，**不是硬件利用率**。逐个同卡同层配对后，大档该比值高于小档分别为 285/320 和 320/320；这是描述性结果，不证明 M 变化造成差异。不同 digest 的输入、有效行、缓存和并发均可能变化，设备 binary 和 tensor 身份亦未闭合。EP3 L36 的小档两项为 181.660/188.160 µs，大档为 270.900/311.400 µs；先前 host 栈仅绑定该大档两项，不能推广给小档。

复算 `python3 agents/fusion/analyze_f16_prefill_rows.py`，脚本 SHA-256 `e70e96922c1368e6cb7ca602abfca1d3659ab0e9da844e7759ea1960425ae94c`；输入 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`。只读本机历史聚合，检查完整 shape、dtype、格式、core、block、设备轨、同流、task 和相邻顺序。按 rank/layer 匹配不同 digest 不涉及跨设备时间戳相减。


### 同卡同层计数字段与简单固定开销模型的检验

沿用上述 1280 个已选 scope、每 scope 两任务；所选 2560 条均有非零 `aic_mac_ratio` 和 `aic_total_cycles`。此覆盖不代表采样全图完整。以大档减小档，对 EP8–15 的 M5144→8184，第二投影的 `aic_mac_ratio` **320/320 下降**，两档中位数为 0.861 / 0.814，逐对差中位为 −0.049；与此同时，上节第一/第二逻辑速率比 320/320 上升。前者是 profiler 导出字段，后者是以标准稠密假设和两个任务时长构造的比值，口径不同且可能共享影响因素；不能把前者叫全芯片利用率、把二者联动认作因果或解释为资源争用。

为避免用两个点过度外推，条件假设同一投影在不同 M 和其它条件相同下满足 `t(M)=a+bM`、`a,b≥0`。逐对用两次时长求唯一的 `b=(t_large−t_small)/(M_large−M_small)` 和 `a=t_small−b M_small`：M5144→8184 的第二投影 **320/320 得到 a<0**，中位 −30.079 µs；第一投影 14/320 得到 a<0。M4760→8184 的第一/第二分别为 0/320、15/320。所有所选对的 b 均为正。这只说明带非负截距的**同一固定系数两点模型**无法描述那些选定记录，不能说真实启动开销为负或已证明行数导致非线性；不同 digest、实际有效工作与运行状态仍可能不同。对其他行数和无 profiler 性能没有预测力。复算脚本与输入哈希同上。

### 双投影分块的中间状态与空间权衡

以单 local group、单 token 为条件，第一投影的 BF16 中间向量 z 有 1024 项、逻辑大小 **2048 B**。若一个候选实现将第二投影 5120 个输出的未完成部分和全部以 FP32 保留，则需要 **20480 B**，为 z 的 10 倍；这只是明确的候选实现假设，历史 kernel 的累加类型及片上驻留均未证。对于明确选择按 z 做直接稠密第二投影、又按输出列分块且不能同时持有完整 partial 的候选，后续列块需要通过保留、共享、已有位置重读或按原精度重算取得其所用 z 信息。特殊权重、结果缓存和其他经证明正确的算法可改变这一要求；各方案的工作量、存储与同步分别核算，不能仅因接口合并就断言中间字节全部从 HBM 消失。

对 EP3 M8184 的形状，若完整 BF16 z 真的在某接口物化，其逻辑大小为 **16,760,832 B**，一次写加一次读共 **33,521,664 B**；不是实测流量、必经 HBM 或可省字节。每行第一投影的传统逻辑运算量为 `4096×1024=4,194,304 MAC`；如果极端方案将第二投影列分成 b 块并每块都独立重算整个 z，该第一投影逻辑量为原来的 b 倍，片上共享则可以避免此重算。该算术没有加入已有时间线，不证明最佳方案或可达到的提速。


### 条件性尾行重组：先区分两类 padding

冻结 `dsa_v41_cp.py:209–225` 先补齐本地 attention 行并进行 `restore_tp_heads`，然后在 `exchanged.shape[0] < hidden_states.shape[0]` 时再新建零尾行，送入两级 O 投影；这段后补尾行不能与交换前的 rank-major padding 混为一谈。冻结 `model.py:426–443` 的 wo_a/wo_b 构造均为 `bias=False`。可研究只对确有数据的行投影，再按原输出契约处理后补尾行，但仅在证明实际分支、尾行身份和消费者的条件下；假如要求保留整个 tensor，应精确重现这些行以及其布局，不能简单移除通信所需行或更改 collective 尺寸和顺序。

`bias=False` 不足以证明对任意浮点数都能直接以零代替原计算：非有限权重可能使 `0×∞` 为 NaN；实际 quant scale/offset、累加、signed zero 和 rank 归约也必须按原语义检查。还需确认 TP 各 rank 的有效行一致性、原 graph buffer 与 HC 残差侧的可见性、异常路径和尾行是否最终可观察。上述 640 组 scope 的 M 是物理行数，不证明尾行数量或该条件分支已触发；未计算收益。

本地已冻结的 CP 源 SHA-256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`，模型源 SHA-256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；历史 binary/实例仍未绑定。此方向与 F15 的 padding 和 F16 的分块候选重叠，不可叠加潜在收益。


### EP3 L36 对“交换后尾行”候选的条件性排除

同一已存 EP3、L36、scope `2128-42`、摘要 `db852bff…`：所选 `SparseFlashMla` task 41435 的第一输出记录为 BF16 `[1023,64,512]`；第一投影 task 41440 的输入记录为 BF16 `[8184,1,4096]`，后者的 host 栈先前已绑定到 `_project_output`。两项同 scope/stream 和形状仍不足以证明二者 tensor 身份或是否经过同一实例的生产消费路径。

若进一步假定：该 SMLA 输出确为这一次 `_project_output` 的 output，运行的是保存的 CP 源并成功完成本地补齐/等分交换，TP world_size=8，且被选第一投影确为该调用，则源代码要求 `per_rank ≥1023`，交换后的行数为 `8×per_rank ≥8184`；投影输入 8184 行对应该代码传入并成功处理的 hidden_states 行数。因此 `exchanged.shape[0] < hidden_states.shape[0]` 为假，**只排除该样本在这些条件下的“交换后再补零尾行”分支**。交换前局部 padding、其他阶段 padding 和其他请求/层仍可存在，不能由此称该样本没有补齐。

既存 scheduler 摘要给出 8180、准备物理行数 1023，数值与 `8×1023=8184` 相容；scheduler total 不等于已捕获 CP 的 per_rank，也不能仅凭 8184−8180=4 证明这四行属于上述尾行。最小补证需要同一匿名调用的代码版本与 rank/replay 关联，以及 output 行数、hidden 行数、CP range 的 per_rank、TP world_size、交换后行数和上述条件分支布尔值；均为元数据，不要求读取 token 值。可观察性或消除尾行的数值正确性仍需单独证实。现有 169 远端认证失败，此轮只读本地 JSONL，未新增采集。


## 大行数任务中按条件工作量归一化的 AIC 代理

沿用上文 1280 个选定 scope 的两个 BF16 任务，在每个 scope 分别计算 `aic_total_cycles × aic_mac_ratio`，再按标准稠密假设下的两任务逻辑 MAC 比 0.8 归一化，取第一/第二。M=4760（320 对）的 min/median/max 为 **1.002263 / 1.004649 / 1.006731**，M=5144（320 对）为 **0.999177 / 1.001394 / 1.003963**，M=8184（640 对）为 **0.999745 / 1.001701 / 1.003650**。这 1280 对第一任务的导出 `aic_mac_ratio` 均低于第二任务，但在这个代数代理及所设工作量口径下，归一化的逐对比例均落在约 0.9992–1.0068。它排除“较低的单一 MAC ratio 必然意味着单位逻辑 MAC 的该周期代理同幅较低”的解释。

指标分母、跨核汇总与字段依赖关系尚未证明，ratio×cycle 可能与工作量或导出算法存在数学耦合；物理有效行、真实执行指令、载入数据和张量身份未知。该比值不是独立硬件利用率，也不能用于判断 HBM 是否受限或推算融合收益。相关 AIV 值为零亦不证明 AIV 未工作。复算 `python3 agents/fusion/analyze_f16_prefill_rows.py`，脚本 SHA-256 `e70e96922c1368e6cb7ca602abfca1d3659ab0e9da844e7759ea1960425ae94c`；固定 JSONL hash 和配对筛选见上文。

## 两个独立投影的条件计算模型与观测

先用宽理想模型允许融合、并行和缓存；以下是为了和单条已存 task 对照而额外收紧的**独立算子窄域**。假设全部物理行有效、采用逐项传统稠密点积，第一和第二投影分别需要 F₁=M×4096×1024、F₂=M×1024×5120 次 MAC；全部该工作都在各自 task 时间窗由指定 Cube 池完成。再假设该池至多 24 核、每核不超过 1.8 GHz 和 4096 BF16 MAC/周期、且所有短窗突发/在途信用 β=0；理想容量 P=176.9472 TMAC/s，模型给出 F/P：

| 记录 | 第一：条件模型 / profile 时间 µs | 第二：条件模型 / profile 时间 µs |
|---|---:|---:|
| EP8 L24，物理 M=24 | 0.569 / 50.559 | 0.711 / 14.779 |
| EP3 L36，物理 M=8184 | 193.991 / 270.900 | 242.489 / 311.400 |

条件模型时间与相应 profile 记录之比分别约 **1.13% / 4.81%**、**71.61% / 77.87%**。它们只是模型条件下的算术敏感性，不是实测芯片利用率或可兑现加速；真实有效行、二进制、执行资源、严格最高容量和 β 均未证，两组也是不同层及采集，不能归因于行数。若允许 fusion、预乘、共享、其他算法或计算资源，未证明这些 MAC 必然存在，不能把本窄域结果当整个任务的理论上界。一般短窗容量式为 `(F−β)_+/P`，不能从标称 sustained 值自行令 β 为零。

4096 的单路径模型依据 Ascend 2201 BF16 Mmad 理论形状，不证明每周期真的可完成一次或它涵盖所有实现；核数和频率为现有元数据，不自动是硬上限。EP8 的 24 行记录只靠 shape/次序选取；EP3 的两条 8184 行任务各自有 host 栈绑定，但仍没有中间 tensor/权重身份。复算 `python3 agents/bounds/f16_conditional_compute_compare.py`，脚本 SHA-256 `c8587596d3e47601089f3becfd6d9bb1cf57b9b2310c2f9b12d3da51501db084`；来源与模型边界见 `agents/bounds/ATTENTION-PROJECTION-EP8-MODEL.md`。
