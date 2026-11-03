# 疑似 Engram 热点：已有融合与可研究的重组

`agents/diagnosis/ENGRAM-BOUNDARY-EVIDENCE.md` 给出 0→1、13→14 两个窗口中，大矩阵形状与 Engram wkv、其后常见算子与普通 gate 相容的证据。`ENGRAM-BOUNDARY-INDEPENDENT-REVIEW.md` 复核了形状与索引；**尚无目标矩阵 weight/storage identity 或逐调用栈**，所以以下是针对源码中的 Engram 路径做条件分析，不能把其中任何候选直接归因于那两条 trace 或计为已获得的性能。

## 源码中已有的数据关系

`engram/layer.py:23–55` 的 `wkv(rows)` 逻辑输入宽度 6144、输出宽度 25600；它将输出分成 key `[T,4,5120]` 和 value `[T,5120]`。key 供每个 HC 通道计算 gate，value 被四个 HC 通道复用。`common.py:61–72` 的普通路径：

1. hidden 转 FP32；若 rotation 非空，对每 32 维块进行反向基变换；key 也转 FP32。
2. 分别计算 hidden 与 key 的每行均方根倒数；用两者、`q_weight.float()*k_weight.float()` 和 key 做加权点积与缩放。
3. 按 dot 的符号执行开方、sigmoid 和 token mask；将 gate 乘同一 value，并与**原 hidden** 相加，转回原 dtype。

gate 是每 token、每 HC 通道一个标量，具有对该通道全 5120 维归约的依赖；在一般域内不能只处理某个 feature tile 就声称最终 gate 已就绪。hidden 的归一化部分不依赖 rows/wkv，key 的归一化与点积需要对应 key 和 hidden；value 投影是 key 投影不同输出通道，但最终结果要等对应 gate。

`model.py:1555–1592` 可能在本层 Engram 之前收集 DSpark aux hidden，然后在首次需要时等待 mask 和该层 lookup event、按需 SP shard lookup，写回 `hidden_states[:n]` 并进入 decoder。具体 aux 列表、SP、event 和分支以运行时为准；重排不能把更新随意提前到会改变这些消费者的位置，也不能让 wkv 在 lookup 未 ready 时读取。外部 hooks/alias、buffer 的其他未快照使用和 compiled graph 的 lowering 未闭合。

## 已有组合与不能重复计数

- 代码中的 `fused_engram_gate` 是已存在的复合实现，但调用仅在 rotation None、NPU 且 profile 支持 `ENGRAM_UNROTATED_GATE` 时成立。冻结 `_build_info` 与 A3 profile 的静态代码显示 A3 不带此 capability，正常加载且对象未替换时应走 ordinary `engram_gate`；不能把 A5 分支当作 A3 已命中的新收益。普通路径是否被编译器再融合仍未知。
- host-UVA 条件下，`engram/npu.py:196–247` 与 `embedding.py:394–401` 通过 `_engram_host_uva_gather_dequant_kernel` 完成已注册主机表行的查找及条件反量化；这已经是组合路径。host 表驻留不等于 NPU 查表是 CPU 运算；不同 device/CPU fallback 和表 dtype 仍各有条件。
- `engram/layer.py:53` 的 `q_weight.float()*k_weight.float()` 是在普通路径传给 gate 的一次预乘，不能笼统声称两者当前逐次使用前没有合并。该乘法是否每次在设备上执行、被编译/缓存消除或与其他算子合并，未证。

## 条件候选和必须保持的依赖

1. **gate 局部计算组合**：研究把 hidden/key 的平方和、加权点积的 partial reductions 与后续 rsqrt、符号开方、sigmoid、mask 合并成较少设备阶段。在 token/h 与不同 feature tiles 上分别形成部分和后，只有完整归约结果就绪才可形成 gate。须保留 `eps`、q/k 权重乘法和点积的 FP32 运算顺序、各次归约/舍入、`signbit`、clamp 和 NaN/Inf；相同实数公式或不同归约树不自动 bit-exact。已有 gate 是否 compiler fused、算子支持和实际收益均 UNKNOWN。
2. **projection 的 key/value 输出拆分与流水**：利用原 WKV 输出通道划分，让 key 的块进入 gate 归约，让 value 的块在计算 scalar gate 后供输出混合；亦可研究 value 与 key 部分并行。必须保持原 6144 输入完整、每个输出通道正确权重及原 matmul 的数值契约；不同 GEMM 形状、算法和累加次序可能改变结果。key 的 4 个通道各自与完整 hidden 归约，value 同时为四路消费，不能丢弃或在首个消费者后覆盖。拆成多个 GEMM 可能造成更多权重读取或 launch，不能由拆分本身推断更快。
3. **hidden-only 前处理与 lookup 重叠**：已就绪的 hidden 可作为条件候选，在 lookup 生产期间提前计算 rotation/hidden-side 归约；但要证明 hidden 和权重真实就绪、aux 消费与写回顺序不改变、此期间没有并发写或复用。mask event 不可在 `masked_fill` 前省略，lookup ready 也不可在读取 rows 前省略。多流相互争用资源、准备工作可能挡住主要 GEMM；是否能隐藏及多大收益未知。
4. **常量 q/k 权重预处理**：若在指定执行阶段权重不可变，且精度契约允许保留 `q.float()*k.float()` 得到的相同 FP32 值，可研究生命周期内复用它，避免重复生成；需要覆盖加载、图/缓存更新、权重替换、设备/进程和多层各自参数。当前源码有乘法调用，但没有证明设备上每次确实独立执行、可缓存或性能显著。

重要的数值反例：源码先把无效 token 的 gate 置 0，随后仍计算 `gate * value`。若 value 允许 NaN 或 Inf，IEEE 运算中 `0×NaN`、`0×Inf` 可产生 NaN；简单以“mask 为 false，所以跳过所有 value 计算并原样返回 hidden”并非在全部 IEEE 输入上自动等价。要采用该跳过路径，必须另证 value 的有限域和所需异常/NaN 语义，或明确更弱的精度契约。相同考虑适用于边界处理和被 mask 的写入。

## 实际证据缺口

0→1 和 13→14 窗口的主流大矩阵与普通 gate 名称相容，且每类窗口含 48 个样本；该关联仍缺 tensor/weight identity、精确分支、profiler 归属、事件以及设备 counter 解释。不要把 `[3,6144]` × `[25600,6144]` 全量权重字节直接当作一次 HBM 必要流量，也不能用窗口差直接算优化收益。仅在固定调用和输入输出契约明确、证据足以区分后，才能进一步计算可复现的上界和可行性。未新增设备测试或修改实现。

## 完整 lookup 组合重复时的投影结果复用：条件候选

此前的 key/value 分块仍每次按 rows 运行 `wkv`。另一个结构是**跨 token 或跨调用复用完整投影结果**：冻结 `engram/layer.py:34–55` 先执行无 bias 的 `self.wkv(rows)`，之后 gate 才读取当前 `hidden_states`、`token_mask` 和 rotation。`model.py:1190–1246` 从完整 hash 经表 lookup/通信得到各层 rows；`engram/embedding.py:333–358,378–408` 展示本地查表及条件 DP exchange、TP gather。因此如果同一层再次出现**完整 lookup 输入和对应行完全一致**，且表、scale、投影权重及其量化/计算语义未变，可研究复用先前的完整 `kv`；仍须对新的 hidden、mask 执行 gate/residual。此结构不同于 F25 在单次投影内部做分块，也不是把一个 head 的投影独立相加：后者改变加法和舍入，不能自动保证等价。

约束比“相同 n-gram 文本”严格：必须包含该层完整 hash 列、head 顺序和表/scale 版本、sharding/设备与投影参数版本；不同的上下文即使当前 token 相同也可能生成不同 hash。缓存键需能确认完整 tuple，不能只依靠不经核实的短哈希碰撞；还需处理并发、释放、重载和图重放的 buffer 生命周期。精度上需证明相同行在不同 batch/矩阵形状下选用的 GEMM 算法与计算契约允许复用（未证明 bit-exact）；不能把不同后端/权重更新前的结果当成相同值。DP/TP 是 collective 时，单个 rank 命中不能自行跳过 collective，必须有全参与者一致的执行序列并保留其他 token、padding 和空 batch 的正确性；在无共同协议前，即便命中也不能据此宣称可以绕过 lookup 通信。

按当前候选样本 BF16 宽度，存储单个结果 `[25600]` 为 **51,200 字节（50 KiB）**，完整 `[6144]` 输入为 **12,288 字节（12 KiB）**；只是张量逻辑大小，不含键、元数据、缓存组织及流量，输出约为输入 4⅙ 倍。额外显存、缓存查询/写入、命中率、分布式同步及潜在新资源争用都可能抵消收益。**历史 trace 没有完整 tuple 身份、重复率或缓存访问模式；权重与该矩阵任务身份亦未绑定，因此该候选没有实测覆盖率、节省时间或端到端收益。**不得用 90% 前缀缓存命中率代替 Engram tuple 命中率，也不能用缓存假设收紧无条件性能上界。进一步取证需在约定允许的历史材料中寻找逐 token hash/表版本/rows 绑定及可用重复率；当前未启动设备或修改实现。

本段源文件 SHA-256：`engram/layer.py` `dc9457c88aa3c90133115bc608447e037b0693f8db71bb8da37f1ead44021f80`；`engram/embedding.py` `a7741ed16522be4fe607e062e1c988c4e2435af7d258fe7db99f672d4508c4c9`；`model.py` `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`。这是静态逻辑与条件结构分析，不证明本次历史运行命中这些分支。

## F25 投影接 gate：点积重排不能消除 key 的完整范数

冻结 `engram/layer.py:23–29,41–55` 是一次 `wkv(rows)`，随后拆出四路 key 与一份共享 value；普通路径 `engram/common.py:61–72` 将 hidden 按需反旋转、转 FP32，再计算每路 hidden 和 key 的均方根倒数及加权点积。现有 `wkv` 已经把五组输出通道合在一次线性调用里；它和 `fused_engram_gate` 可用条件如上，不可将这五组计算当成五次已证实独立 launch。

对单行、单 HC 路，只作**理想实数、无量化/舍入的代数模型**：设 lookup 行为 `r`，key 权重矩阵为 `W`，`k=W r`，旋转后的 hidden 与 `channel_weight` 逐项乘得到 `a`。gate 的 dot 有项 `aᵀ W r`，可按 `aᵀ(Wr)` 或 `(Wᵀa)ᵀr` 计算；但代码还需要 `mean(k²)` 用于 key 的 rstd。仅保留 `aᵀk` 不能确定这个范数：取 `a=(1,0)`、`k₁=(1,0)`、`k₂=(1,1)`，两者点积均为 1，但平方和分别为 1、2，归一化 gate 一般不同。不同的 lookup rows 或权重实现这两个 key 的存在并不是本次运行的事实；这是说明不能普遍删除 key 范数的数学反例。

因此可以研究两种**相互重叠的条件结构**，但均不代表当前运行一定物化了 key，不能叠加预计收益：

1. **key 投影 tile → 原语义转换 → 双归约**：某些输出通道的点积完整就绪后，按实际 linear 实例的 dtype 和舍入产生对应 key tile，随即对该 tile 同时累计 `original*channel_weight*key` 与 `key²` 的 partial，保留其他 HC 路与 value 投影输出。只有该 HC 路 5120 维所需 tile 全部完成，且归约顺序、rsqrt、eps、signbit、sqrt、sigmoid、mask 均符合原精度契约，才能生成其 gate；输出混合还要等共享 value 和原 hidden，保留 mask 对 NaN/Inf 的语义。hidden 的 rstd 可独立于 lookup/key 提前计算，但原旋转、hidden 的其他消费者和 buffer 生命周期仍需满足，且与 projection 可能争抢 Cube/Vector/MTE。所需 tile 可见接口、片上缓冲容量、编译图已做的融合和调用次序均未证明。
2. **代数消元或预计算的成本与局限**：理想实数下 key 平方和可写为 `rᵀ(WᵀW)r`；但为了维持四个 HC 路各自的 key 范数，一般需要每路相应的 Gram 信息，而不仅是上面的点积。对于本配置中每路 `W` 的逻辑形状 5120×6144，一个稠密 FP32 `WᵀW` 是 6144²×4 字节 = **144 MiB**；四路共 **576 MiB**，仅为假设的张量逻辑尺寸，不表示必须实际分配、真实访问量或最快实现。它还要求权重版本固定；如果做对称压缩、隐式表示或低秩替代，必须分别证明输出与精度契约。把原矩阵乘后的实际输出舍入、权重量化、FP32 归约以及非线性替换成此实数恒等式，通常不能保证原结果，除非另证等价；`Wᵀa` 也依赖每次当前 hidden，不能当作对所有请求预先固定的常量。该表达式只是用来识别隐藏的计算和存储代价，不证明适合实现。

这些结论仅基于所存源码、配置维度及上述显式假设。历史大矩阵任务仍没有权重/tensor 身份绑定，且两路同时执行会出现真实资源竞争；不能从该代数重组推算延迟改善、实际吞吐或硬件理论上界。若要进一步收紧，需找到历史 `wkv` 实例与设备任务的调用/权重绑定、量化输出 dtype、原 gate kernel/图及其数值误差契约和 tile/缓冲接口；缺失时保持 UNKNOWN，不新增 NPU 负载。
## lookup / 通信 → WKV 的 K 维流式连接

前文分别研究了 lookup 准备、WKV **输出** tile 与 gate，也研究过完整 rows 重复时缓存投影；这里检查不同的跨模块边界：能否把一次 lookup 产生的**输入 K 维块**直接交给本次 WKV 点积，而不要求先得到完整的 `[T,6144]` rows。冻结 `engram/npu.py:196–247` 的 host UVA 条件路径已将按 ID 查表与反量化放在同一个 Triton 调用，输出逻辑为 BF16 `[tokens * pad_heads, width]`；不应重复将“查表+反量化”作为新融合。`engram/embedding.py:333–356` 的非 direct 路还可能对 rows 做 DP exchange、TP head all-gather，再按 checkpoint 顺序组合；`model.py:1199–1236,1565–1592` 的 direct 分支要求已有 `output_buffers` 且 `dp_size=tp_size=1`，生产每层独立 lookup buffer 并记录完成 event，消费端按需等待、SP shard、截取，然后调用 layer。`engram/layer.py:23–29,34–41` 用一个已有 `ReplicatedLinear` 对完整 rows 做 WKV，key 和 value 已合在同一投影里。

**条件重组**：对确定的 token 与该层，每段 hash-head/特征块已完成正确 lookup/反量化及所需跨 rank 交换、按原排列映射到 WKV K 维，且其数值已达到原 BF16 输入语义时，可研究将该块递交 WKV 的部分点积；一条输出通道仍必须累计**整个 6144 维**才算完成，之后继续遵循既有 key/value、gate、mask 和 residual 的消费关系。这与前述 WKV 输出通道分块及完整结果缓存不同，但可能与它们重叠，不能重复累计收益。当前源码的外部 event 是整层 rows ready，当前 WKV 接口没有证明支持逐 lookup 块就绪或读 host table 的方式；该方案可能需要新 kernel 或通信接口，不能声称已有路径可直接打开。

一个明确的非等价反例：把完整 K 维求和拆分为先独立归约各 head、再合并，不能仅凭实数加法结合律声称与原 GEMM 逐位相同。假设 FP32 累加和 BF16 精确可表示的三个乘积按原次序为 `[2^80, -2^80, 1]`，原次序求和为 1；若先合并后两项，再与第一项相加，得到 0。该例不代表原设备真实归约顺序或本模型实际输入，只否定未经数值契约证明的任意 K 维重排。输入的 BF16 生成、无效 ID/零填充、table scale/dtype、padding 和实际算子舍入都必须保持，不能以未经 BF16 语义处理的 FP32 反量化值直接替代。

direct 条件即便成立也只说明 lookup 不需该 `embed_gathered` 内的 DP/TP rows 交换，**不排除**上游 hashes gather、SP token shard、图持久输出及每层 ready event、其他消费者或跨轮 buffer 生命周期。非 direct 模式须保留必要 collective、rank 一致顺序、head 重排和到达；不能把未经交换的本地表块当作完整 rows。若某层的 lookup 缓冲对 graph replay、alias 或其他消费者可见，移除物化前须证明这些观测被保留。host UVA 访问、Cube 权重读取和通信可能竞争，碎片化 K 维也可能重复加载权重；不能由消除一个逻辑缓冲推断实测 HBM 流量或提速。

可证伪条件是：已绑定某次历史任务使用该层 lookup、WKV 权重和同一 token 行，且能取到对应 head→K 的排列、各块 ready/collective 契约与原输入 dtype；若这些不成立或必须保留整行可见副本，则无额外接口的直接连接不可行。当前所存 48 个层边界矩阵形状及 checkpoint header 与 WKV 相容，但尚未绑定 tensor/weight、实际模式或时序，不能称该候选为历史瓶颈或估算收益。本次仅读源码和现有证据，无 NPU 操作。

### TP8 与 DP shared 对 direct 路径的具体限制

`engram/embedding.py:74–110,135–138` 将 resolved `dp_shared_memory=true` 时 table 的 `dp_size` 设为 1，但 **`tp_size` 仍从 TP group 取得**，并按 TP rank 分配 hash heads。`engram/parallel.py:24–26,58–73` 的 shared 条件让 `gather_engram_hashes` 返回本 rank hash，而 `embedding.py:349–355` 对 `tp_size>1` 仍执行 head 维 `tensor_model_parallel_all_gather`。因此，若诊断实际表实例的 shared 生效且其 TP group 实为启动请求的 8，`model.py:1207` 的 `dp_size==1 and tp_size==1` 为假，**不会进入该 direct 分支**；这不证明历史 resolved group、source/对象或真实 collective 已绑定。

按冻结配置 `(max_ngram−1)×n_heads=3×8=24` 个 hash columns、每列宽 256，以及 shared 且 TP8 的条件，可得每个 TP rank 的 table 分到 3 列、768 个 BF16 输入特征；完整 WKV 的 K 仍是 6144。用单 rank 的 local 768 列直接算完整输出，在一般权重域不成立：只保留另一个 rank 某列的非零权重，该列值取 0 或 1 时，本 rank 本地列完全相同而 WKV 输出不同。想提前按本地 K 块做部分点积，必须取得其他 rank 的相应 head 块或设计另一种正确的跨 rank 部分和归约，并保持原 BF16 中间量、归约顺序及 group/collective 次序。即使 shared 开启使 DP hash gather 无操作，它也不能消去 TP head gather 或跨 rank 信息依赖。

若探索“每 rank 本地 768 列做 WKV 部分和再跨 rank 合并”，完整 rows 一行 BF16 是 `6144×2=12,288 B`，WKV 逻辑输出一行 BF16 为 `25600×2=51,200 B`；后者约为前者 4.17 倍，仅是张量形状会计，不是实际 all-gather/all-reduce 通信字节、内核时间或必然的传输方案。该变换还会改变 GEMM 累加顺序，并可能需要 FP32 partial；当前没有精度与通信契约证成。实际若 SP 切分 token，需要同时按真实 token owner 与 padding 建立相同的行映射，不能以同名 TP rank 自动对应。

当前既有输出不是抽象临时值：`model.py:1332–1352,1418–1456` 为各层创建跨运行复用的 BF16 容量缓冲和 mask；正常非 direct 分支会将已交换/收集的 rows copy 到该缓冲并清零尾部，再记录整层 event（`:1217–1226`），消费时必要地 wait、SP shard 和截取（`:1565–1589`）；准备流还等主流后才复用此前 buffer（`:1462–1468`）。因此在这一路径中直接消除 `copy_` 或将事件提前至只完成 local lookup，会改变图与消费者可见的缓存内容或发布时点，除非有新的逐块数据/事件协议并证明所有消费者和跨轮复用。

host-UVA 的源码契约也有具体地址边界：`engram/npu.py:23–27,135–162,196–247` 将 host 注册后取得的 device pointer 每 `2^22` 行建立一个设备地址表，传给 Triton kernel，另外传 head 范围、ID stride、scale 与输出等；`close:173–185` 在注销注册内存前同步设备。新的 Cube/混合实现必须保留同等有效地址、ID 范围/stride、缩放及最后访问生命周期；现有 Python 接口证明的是 **Triton lookup** 可使用注册地址，不能单据 UVA 名称证明矩阵乘后端可直接从该 host 地址取数。若 runtime 查询实际 TP group 非 8、表实例非 shared，或不存在相关 rows/copy 消费路径，上述特定分支可被否证；当前没有逐调用绑定、逐块事件或直接 host→Cube 的已验证接口，不据此宣称实际瓶颈或性能差值。

## F24：`gathered.contiguous()` 的逐元素 stride 证据（2026-10-29）

本轮 EP8 两个采样窗口的队列调用栈中，共四条 `aten::contiguous → clone → copy_` 指向 `embedding.py:342 embed_gathered`；导出 copy 输入 shape 分别为 `[8184,24]`、`[4888,24]`，但没有输入 strides、storage offset 或 tensor 身份。队列证据、源码镜像一致性、采集扰动与关联限制见 `diagnostic-20261029/binary-binding-draft/second-binding-run/EP8-ENGRAM-COPY-QUEUE-PROVENANCE.md`。源码中的 `.contiguous()` 对某些输入可以原样返回；这里已观察到的调用栈比单看源码更强，但不能把 API 排队时长当可节省的设备或端到端时间。

**数据流和形状。** `engram/hash_state.py:95–103` 的正常 hashing 分支创建 `[tokens,num_layers,(max_ngram−1)*num_heads]` 的 int32 输出；冻结 `evidence/model-config.json:121–132` 有 Engram 两层 1/14、max_ngram 4、8 heads，所以**当实际归一化配置、该正常 hash 分支及实例与快照一致时**为 `[T,2,24]`。`model.py:1182–1204` 可能先按 token 索引，然后对完整 hashes 调用 `gather_engram_hashes`；`model.py:1205–1220` 在非 direct 分支把 `gathered[:,slot]` 交给 `embed_gathered`。shared/group-none 时 `parallel.py:58–67` 返回原 hash，其他分支先做 padding/DP gather；dummy hashes、token 索引、collective 返回布局与实际历史分支未绑定。Python 的通用 `embedding.forward` 也可从其他 2D indices 调用它，因此这不是对所有调用点都成立的假定。

对一个**标准连续**的三维张量 `A[T,2,24]`，`A[:,slot]` 有 shape `[T,24]`，列 stride 为 1、token stride 为 48；slot=1 的 storage offset 比 slot=0 高 24 个元素。这个数学结论不能从历史 shape 单独倒推出原张量的实际 stride 或 offset。`T>1` 时它通常不满足完整连续要求，`.contiguous()` 将按逻辑顺序形成每行 stride 24 的新张量。若错误地直接删除 copy 但把行 stride 硬写成 24，那么对于 `[2,2,24]` 的 slot 0，第二行将读取第一个 token 的 slot 1 而非第二个 token 的 slot 0；这说明不能仅凭 24 列判断紧排。

**两条 NPU lookup 的实际源码契约。** `embedding.lookup:385–405` 根据 `_codes_uva` 等条件分别调用 host-UVA Triton 或 device-table Triton；`engram/npu.py:90–111,224–245` 均向 kernel 传 `ids.stride(0)`。只读 stage 源 `engram_lookup.py:48,90` 显示两种 kernel 的索引分别是 `ids_ptr + token*ids_stride_t + HEAD_START + local` 与 `ids + token*ids_stride_t + HEAD_START + head_local`。因此若输入为上述 `[T,24]` view，且**列 stride=1**、行 stride 与真实 layout 一致，每种 kernel 都能按照逻辑 token/head 得到与连续副本相同的 ID；它们既没有要求所有 token 行紧邻，也不会通过这条公式读取其他 layer 的列。假如列 stride 不是 1，两个 kernel 没有接收 `stride(1)`，此推论不成立；其他 layout 仍须复制或另行设计接口。保留 global vocab owned 判定、invalid-ID 零值、scale/MXFP8 解释及输出 BF16 舍入。`_torch_lookup` 是另一个 CPU fallback，不能将两条 NPU kernel 的条件结论自动扩展成所有后端的保证。

**offset 与提交。** `diagnostic-20261029/engram-ascend-pointer-detail.json` 保存的 stage 已安装 Ascend Triton driver `driver.py:590–615,956–958` 显示生成的指针参数 glue 使用 `getPointer` 并调用 tensor 的 `data_ptr()`；[PyTorch 文档](https://docs.pytorch.org/docs/stable/tensors.html) 定义 `Tensor.data_ptr()` 是该 tensor 第一个元素的地址。因此在**该 glue 确实用于本次 kernel、参数确为该 tensor view**的条件下，slot=1 的 storage offset 已由 view 的首元素指针承担，不能再手工加一次 offset。stage `engram_lookup.py` SHA-256 `c2b1b46ea4eeb70513cb4a53a5fc584bd024c81269fff9f997431a3ff172899b`，Ascend driver SHA-256 `ab4d1031efdfb07492ca95d9405089e3b7166ccee0555dbc756f191abe53b1e7`；源码与本次实际 runtime/编译 kernel 身份、stride 参数、地址及后端行为仍未逐调用绑定，这不是设备数值验证。

**结构候选与必要约束。** 针对经核实的 unit-column-stride view，让现有两类 stride-aware lookup 直接读此 view，继续传实际 token stride；其他输入、后端或未核实 layout 保留已有连续化。DP hash gather 仍按原协议先完成、padding 与 DEAD_ID 保持；lookup 的 DP rows exchange、TP head gather、输出 buffer 和对应 ready event 仍按当前语义执行。消除 clone 会改变临时存储的生命周期：必须证明同一逻辑 hash 已写完、所有参与流有必要依赖，view 底层 storage 在 lookup 最后一次读取前不被其他消费者或下一轮重放改写/复用，且 CUDA/NPU graph、allocator、capture 对指针和元数据的要求均满足。若原 clone 本来提供隔离快照而源随后可被改写，直接读取会改变结果；此时必须保留快照或另建等价同步/缓冲。空张量、非连续列、token 索引、其他 graph/runtime 和 CPU fallback 均须单独处理。已有 host-UVA gather+反量化已组合，该候选只针对上游 ID 物化，不重复计 F24 其他融合收益。

以上是**源码级条件候选**：还缺历史四条 copy 各自真实 stride、storage offset、producer/consumer alias 与流事件、使用的后端和编译对象，以及数值契约；没有 NPU 正确性或性能结果，也不说明它就是低利用率根因。定向只读查询由本机 ZCode stage252/253/254 完成，均 exit 0、`model_verified=true`、模型 `deepseek-flash`；无 NPU 操作、无目标实现修改。

### F24：复制的可证明工作与物理带宽边界（2026-10-29）

已保存的 EP8 队列证据见 `diagnostic-20261029/binary-binding-draft/second-binding-run/EP8-ENGRAM-COPY-QUEUE-PROVENANCE.md`。其中 `aten::copy_` 导出两种形状 `[8184,24]`、`[4888,24]`，显示 dtype 为 `int`，没有具体位宽、tensor 地址、实测 HBM/L2 字节、输入/输出 cache 初态或 clone 的 device task 时长。冻结 `engram/hash_state.py:95–103` 正常 hash 路径创建 int32，但还没将该具体 tensor 与历史两种 copy 输入绑定。

仅在假设历史输入确为 int32、shape 精确且没有压缩/重编码下，各个输入的逻辑大小分别是 **785,664 B** 和 **469,248 B**；一次完整读输入加一次完整写相同大小输出的逻辑读写账分别为 **1,571,328 B** 和 **938,496 B**。这是条件下的元素访问量，不是 HBM 流量、实际字节、性能收益或任何通用硬下界。队列中的每种 shape 有两条样本，并不证明两个 copy 是同一请求或可以合并执行。

**可证但严格受限的写入界。** 对单次 copy，人为固定 cut 为“发送到预先存在的、互不重叠的平铺目标存储位置”，假设所有 32 位输入均需逐字节精确复制，输入各 bit 可自由取值，原目标内容在计时起点全部为对应输入逐位取反、不得以别处已有副本替代该目标、不得压缩或通过旁路更新，并要求在计时窗内将完整结果写入该目标。此时输出中每个位都必须改变，所以指定目标 cut 至少须接收逻辑大小的不同位，即分别至少 785,664 B 或 469,248 B 的原始写入。该构造证明某个受限初态的必要写入量，而非任意输入/目标初态、真实 allocator 状态或 HBM cut 的界；实际源和目标可以在片上缓存，并可与其他阶段重叠。删 copy 后也不能直接说节省这个字节数：直接读取的内存事务、消费者复用和缓存状态可能改变。

**可复算的反例：stride 访问可能比连续化更耗 cut 字节。** 仅构造抽象 64 B 缓存行的 cut 模型：int32 `A[T,2,24]` 起始按 64 B 对齐、`T` 为偶数；slot 0 的每个 token 行起始间隔 192 B，每行需求 96 B，因而逐行访问需要 2 个不同 cache line，即 128T B。假设每次访问前缓存均为空、完整 cache line 必须跨 cut 读取、copy 允许不读旧行的整行写、没有额外开销，且同一份 ID 由 R 次 lookup 消费。直接 view 总读取 `128TR` B；copy 方案读取 `128T`、连续目标写入 `96T`、R 次连续读取 `96TR`，合计 `(224+96R)T` B。R=8 时 copy 为 `992T` B、直接 view 为 `1024T` B；R>7 时该模型下 copy 的总 cut 字节更少。比如 T=8184 分别为 **8,118,528 B** 和 **8,380,416 B**。这只是反例，不表明 169 的 cache line、次数、cache 冷态或事务模式如此；实际 F24 是否多次读取、哪一个 cut 是否承担这些字节尚未查明。算子合并、相邻 slot 同时访问和分块复用也会改变上述账。

如果将来证明某一指定 cut 的必要字节数 D，以及覆盖整个执行窗口所有并行路径的硬服务包络 `Q(t)≤b+B*t`，该部分才可得 `T≥max(0,D-b)/B`；其中预取和在途字节计入初态/突发 b。读写是否共享资源需另证；若分别竞争同一 cut，才能在同一窗口内合计其不可避免字节，否则不能把读、写和计算时间直接相加。当前没有能将必要 D 和硬容量 B 同时绑定到该历史 copy 或直接 view 的证据，因此无法给出本次可复现的数值硬时间界或提速承诺。官方 `GetCoreMemBw` 的 Byte/cycle 单位不建立具体 ini 字段的硬容量映射，见 `agents/bounds/hardware_sources_audit.md`。

CPU 算术复算：`python3 -c 'for t in (8184,4888): print(t, t*24*4, 2*t*24*4, 8*128*t, (128+96+8*96)*t)'`。没有启动 NPU、修改推理实现或读取用户 token。

### 四条历史 ID copy 的生产者追溯与边界（2026-10-29）

对上一节 EP8 四条 exact `aten::copy_` 锚点，限定同一线程、各锚点前 5 ms，查询 `select/slice/index_select/cat/contiguous/clone/copy_`，再按同目录保存的调用栈过滤。四条均找到先于本次 `embedding.py:342` 的 `contiguous → clone → copy_` 的最近一个 `aten::select`，栈顶均为 `model.py:1217 prepare_engram`：两条 select 输入记录 `[8184,2,24]`，另两条 `[4888,2,24]`；对应 copy 分别是 `[8184,24]` 和 `[4888,24]`。四条 copy 的栈还包含 `model_runner_v1.py:3234 → vl_model.py:239 → model.py:1664 → model.py:1459 → model.py:1217`（从外向内列举）。这提供了同线程、源码调用位点与前后 shape 相容的具体证据，支持前述三维选择结构；采集并不提供 select 的实际 index/dim 标量、输入/输出 storage ID、stride 或 offset，最近时间和相同调用栈也不直接证明 tensor 地址一一相同。不能把 `[T,2,24]` 的 shape 提升为实际 `(48,1)` 的 stride 观测。

反例：一个常规连续 `[T,2,24,2]` 张量沿最后维度取第 0 列可产生 shape 同样为 `[T,2,24]`、stride `(96,48,2)` 的合法 view；再按中间轴选择，形状仍是 `[T,24]`，但 stride 是 `(96,2)`。这个 view 需要列 stride 支持，不能由现有仅传 `ids.stride(0)` 的 kernel 正确读；该例只证明**shape 不足以判定**，并不声称历史路径真的创建了这一 view。

从已存 runner 的 `model_runner_v1.py:3169–3190,3216–3240` 和 `vl_model.py:225–249`、`model.py:1650–1674,1392–1404` 看，已记录栈对应的 `_model_forward` 向 wrapper 传 input_ids、positions、padded_tokens、lookback 和仅包含 `query_start_loc/slot_mapping/block_table` 的设备参数，并没有传 `token_indices`；两个 wrapper 对该参数默认 None、原样透传。如果本次调用确使用这些未被其他代码替换的文件与此调用路径，那么 `model.py:1192` 的 `hashes.index_select(0, token_indices)` 在这四条对应的准备过程中不执行。stage 只读核验 runner 和 VL 源文件 SHA 分别为 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`、`bb98a2e2a0e24c18aafe17229eb593937aaed2e37c1fd7cc2003c45ce872d282`，与已有快照一致；诊断镜像/rootfs 关联见前述来源报告。运行时 monkeypatch、wrapper 改写和参数实际值没有独立快照，因此这仍是**条件源码推断**，不是直接观测 None，也不能证明 hashing 而非 dummy、DP shared 是否生效或物理 buffer stride。

源码中，无 DP group 或 shared 时 `gather_engram_hashes` 返回原 hash；否则先处理等长 slot，再由 group 做 dim=0 的 gather。已存 `parallel_state.py:750–769` 将 group gather 委托实际 device communicator；stage 的 `base_device_communicator.py:247–270` 显示该实现分配新 output、`all_gather_into_tensor` 后 reshape/movedim/reshape。若**该次 DP hash gather 实际使用这个 base 实现、输出分配为标准连续、且接口操作符合源码**，dim=0 的移动不换轴，所得 `[T,2,24]` 可按标准连续布局推导 slice 的 `(48,1)`；没有证据证明历史 DP hash gather 走了这个 communicator、调用时有无替代路径或其输出实际 stride，所以不能据此断言四次拷贝可移除。

同两次采样中，两条出现在两次 ID copy 之间的 `[T,8,3,256]` BF16 clone 的调用栈另含 `base_device_communicator.py:265 → tensor_model_parallel_all_gather → embedding.py:354`；结合源码，这是 **rows 的 TP all-gather 及其 reshape/copy 路径的强线索**，与上游 ID 的 `embedding.py:342` copy 是不同操作。trace 未直接记录通信组身份、跨卡流或 tensor 地址；不能把这两类 copy 合并计时、推断通信字节，或消除 ID copy 后省略 TP gather。

**生命周期仍未闭合。** `model.py:1462–1483` 的 overlap 分支先令准备流等主流，并对外来 NPU input 和持久 output buffer `record_stream`，之后把 `prepare()` 放到该流；非 overlap 分支直接在当前流调用。上述源码表明本路径有明确的 producer-stream 顺序设计，但未记录历史是否启用 overlap、hash/gather 的真实 allocation stream、view storage 是否由其他流复写或底层 allocator 如何追踪被 Triton 异步读取的引用。用原 view 取代副本必须继续证明写完成、每个 lookup 读取完成前原 storage 不变、已记录的跨流依赖与图 replay 生命周期；现有数据不能给出结论，也不能将其宣称为诊断根因或实现收益。

复核数据 `diagnostic-20261029/binary-binding-draft/second-binding-run/engram-copy-producers.json` SHA-256 `38d5bd58b662e5f0910a558abf75a8710b63aa339aed4cc9182798a57b5d9613`、`engram-copy-producers-recheck.json` SHA-256 `62c019b8d0f8b8a7a5d9f0fd6fd6095e72968a6a533157cb5fc9d8ac304d13bb`。前者记录 DB 原先保存的预期 SHA，查询本轮没有重算大 DB hash；相关 DB 的离线导出异常与完整性限制仍适用。复查脚本 `python3 diagnostic-20261029/binary-binding-draft/second-binding-run/recheck_engram_copy_producers.py` 只用上述保存数据，查四锚、scope 嵌套、最近 select 与 shape；它不能补证 tensor identity。ZCode stage270/271/272 均 exit 0、`model_verified=true`、`deepseek-flash`，仅只读历史 DB 和 stage 源码，未触 NPU、未改目标实现。

### 四条历史 copy 的原始 torch profiler 元数据核验（2026-10-29）

**结果：未取到这四次的真实 stride。** 安装版 torch_npu 的 `TorchOpBean` 将输入 tensor 元数据放在原始 `torch.op_range` TLV 的 type 5，`_event_tree_parser.py:124–146` 所述结构可包含 tensor impl、storage 指针、dtype、size、stride、device；数据库导出的 `PYTORCH_API` 则只有输入 shape/dtype 等列，没有这些 tensor 字段。注意这是另一套 FRAMEWORK 文件，与此前发现的 CANN `host/data/*additional.tensor_info*` 不是已证明的同一格式。

定向只读本次第二份 EP8 profile 的 `FRAMEWORK/torch.op_range`（355,405,698 字节，读前后 size/mtime 一致，SHA-256 `4e55ae162e1e20edbe22c7692f672b8dd7e04914e323c0354f515373e8ae132a`），依据已安装同版本 `TLVDecoder` 的 `<HI>` 外/内 TLV 与 58 字节 `TorchOpBean` 固定头，流式核对 165,794 条记录。采用该 profile 的 100 MHz `host/info.json`、`host_start.log` 和 CANN `host/start_info`，按已安装 `ProfilerConfig` 规则把 raw counter 转为本地时间；每个目标还要求 pid=746、tid=746、事件名、**起止时间均精确相等**、shape 和调用栈均匹配。四次各自的 `select / contiguous / copy_` 共 12 个目标，**12/12 唯一匹配，全部没有 type 5 字段**。因此不能从这些 raw 事件得到其 tensor strides、storage 身份或 offset，之前的 `(48,1)` 保持条件推导，不是实测。该结果只覆盖这 12 条事件，并不证明整个 trace 的所有记录均没有该字段。

本次 `profiler_info_8.json` 记录 `record_shapes=true`、`record_op_args=true`、`profile_memory=false`；这些配置事实不能单独证明 type 5 缺失的确切原因，也不能承诺打开某个开关就一定有正确的 stride 或零性能扰动。版本解析源码、raw 匹配和时间配置见 `diagnostic-20261029/tensorinfo-parser-sources.json`、`tlv-decode-sources.json`、`raw-profiler-clock-audit.json`、`ep8-clock-details.json` 与 `engram-raw-stride-metadata.json`；结果文件 SHA-256 `1e3dd29694eb701330a909bcdc47437582089e8b52ca62115dd36cae8eacd8be`。脚本 `run_extract_engram_raw_stride.py` 和 `extract_engram_raw_stride_inside.py` 从已保存四锚生成目标，只有精确匹配后才尝试解析 shape/stride，并只输出本次内部的匿名身份标签，不输出原地址、tensor 内容、请求或权重。离线 DB 的原记录 SHA 是状态文件中保存的预期值，本轮没有重算 DB hash；导出曾有 SQLite I/O 错误这一限制仍有效。12 条匹配是 host 事件身份，不证明 NPU 完成、tensor 生命周期或潜在性能。

同一 EP8 profile 中还找到六个 CANN host `additional.tensor_info` 数据切片，已在 `cann-tensorinfo-decoder-audit.json` 保存大小和 SHA，读前后 size/mtime 均一致。只查已安装 `/usr/local/Ascend/cann-9.2.0/tools/profiler` 目录的 Python 源码中明确文件名关键词，没有找到解析器；**不能由此排除二进制实现、其他目录的工具、其他字段或这些切片包含所需信息**。这批 CANN raw 只做指纹，没有解码，因此它对四个目标的 stride/offset/alias 仍是 UNKNOWN。对应 stage 使用 ZCode、exit 0、`model_verified=true`，只读离线资料；无 NPU 或目标服务操作。

最小后续证据：优先找到与这版采样一致的 CANN tensor-info 格式/解析器，确认它是否有按确切 host 事件身份可绑定的 stride、storage identity 和 offset，并限定仅提取目标调用。若以后允许新隔离诊断，则须在同一次对应 view 上记录匿名 storage/impl 对应、shape、stride、offset、dtype、rank、调用序号及相关流/graph-replay 标识，且单独建立 producer 写入、lookup 最后读取、跨流复用和完成之间的事件顺序；这些条件无法仅凭另一轮输入相同推回本历史样本。当前任务限制是**不启动新 NPU 负载**，本节没有启动新的采集。

### CANN `additional.tensor_info` 的版本匹配解码器与单切片核验（2026-10-29）

**更正上节的检索缺口：**此前只按完整文件名字符串匹配 Python 源，未找到解码器；这不代表不存在。现按安装文件与 dispatch 关系核查，CANN 9.2 安装包中的 `file_name_manager.py:234` 明确将 `aging/unaging.additional.tensor_info.slice_*` 分配给该数据类别，`msparser/add_info/tensor_add_info_parser.py:85–97` 调用 `TensorAddInfoBean`。选定的 `msprof_analysis.so` 中也有 `additional.tensor_info` 字面量，但字面量本身不证明哪条任务用了该二进制。

安装版 `StructFmt.TENSOR_ADD_INFO_FMT` 为 `=HHIIIQQI55I`，每条 256 字节；`TensorAddInfoBean` 将头部映射到 level、type、thread、timestamp、node、tensor count，剩余至多 5 组、每组 11 个整数。`GeTensorBaseBean:129–141` 把每组解释为 input/output 标志、format、dtype 和 8 个 shape 位置。安装版 `TensorInfoV2Map` 的导出字段也只列 thread、timestamp、op 名、tensor 数和输入输出 format/dtype/shape，**没有 stride、storage offset、tensor impl 或 storage identity 字段**。这是对版本匹配的该数据类别及其公开解析路径的源码判断，不是对所有 CANN 二进制或其他 profiler 数据格式的断言；未另行建立同一次记录和四条目标 copy 的精确 identity。

对第二次 EP8 保存的 `aging.additional.tensor_info.slice_0`（2,097,664 字节，SHA-256 `a12df64fb3934da72c2d5ded8b7c17a52ec252bdeb5f95b35b2dfe76a1a3dd88`），禁用 Python 字节码写入，使用**该安装包的 `TensorAddInfoBean.decode`**直接解码，不调用会保存数据库的 parser。以安装包的 magic/level 和 256 字节边界检查：原始切片共 8,194 条，magic、level、tensor count 和 bean 解码错误均为 0；这些原始记录全为 node level，thread 值均为 6857。该线程值与候选 dequeue 工作线程相同，但单凭线程相同不能将任何一条记录认定为四条目标调用，也不能作为 stride/alias 证据。这里的计数是**一个 slice 的原始记录数**，未应用工具的 aging 文件跨片 offset/去重逻辑，不是全采样调用数量；没有读取或输出单条 tensor shape、地址、内容、node ID 或时间。其余五个切片未解码，不影响对已核实安装版固定结构的字段说明，但不能据此排除其他存储或私有扩展。

源代码与结果分别为 `diagnostic-20261029/cann-tensor-parser-source.json`、`cann-tensor-bean-source.json`、`cann-tensor-base-bean.json` 和 `cann-tensor-slice-aggregate.json`（结果 SHA-256 `9e4410751d898b18fa361e83ee0455b2bb3106bedd3b119ce0d9f2adccdf6e8d`）；复现入口 `python3 diagnostic-20261029/run_audit_cann_tensor_slice.py`。静态文件 inventory `cann-profiler-binaries-audit.json` 含各已检查安装文件 SHA，`tensor_add_info_parser.py` SHA `a89f63b1207d759a6ad4fe04e024a780f1f269c7e09b2a6d7b5dd9c8167ea07c`，`tensor_add_info_bean.py` SHA `2ccdd4f838cfada6c1eade75b75d5965055201939035ae8b3fd2055018df6d68`。ZCode 阶段均 exit 0、`model_verified=true`；没有调用设备或修改目标服务。

因此，**这份 CANN 公共解析结构不能补回四次 view 的实际 stride、offset 或存储别名**。进一步找到精确 `(thread,timestamp,node)` 关联可能增加 shape 和调用归属证据，却无法凭该结构产生不在字段里的 stride；要判断原 view 的异步读取是否安全，仍需要另有 stride/offset/匿名 storage identity 和读取/覆盖的同次流与生命周期证据。当前不因其他无关记录的时间邻近猜测这些值，也不启动新 NPU 采集。

### 后续无 profiler 诊断：对 gathered 直接读取候选的观测与独立审计

2026-10-29 后续独立运行 `001d14496b2146bba25dae0f27c780b5` 的测量窗口 `139b95f4c22540c2bca8b398b57edd65`，已保存的 `audit_live_metadata.json` 报告 1,513 条唯一 begin 且 `tensor` 为字典的保留记录，全部显示 NPU `torch.int32`、非连续、stride `[48,1]`，storage offset 0 或 24；其中 shape `[24,24]` 的是 1,462 条。shape 不止一种，不能将这些条目都称作某个 prefill 阶段或某一层。分布来自日志中保留下来的观察，不是整次调用的概率分布或覆盖证明。

该 observer 在 `embed_gathered` 包装调用的 **原方法执行前**读取输入 view 元数据，记录对象是进入 `gathered.contiguous()` 之前的张量。冻结源码的 `embedding.py:333–355` 在此进行 contiguous，再交 `self.lookup`；`npu.py:96–104,229–237` 的两个 Triton 入口均接收 `ids.stride(0)`。这些新观测增强了“对部分真实运行输入，保留 unit inner stride 并让 lookup 接受 row stride 48，存在省去中间 contiguous 的结构候选”的证据；不意味着真实路径一定使用哪一个 lookup 分支、改变后输出一致、能节省多少、或该候选已实现。

**聚合审计边界。** `audit_live_metadata.py:8–20` 用 progress 的 window ID 筛选 begin/end，以 `(process_instance, call)` 分组；1,513 是 begin 分组中唯一一条且 metadata 为字典的数量。826 是其中 end 恰有一条、且结果为 `returned` 的键数；返回只表示 Python 原方法返回，**不证明异步 NPU lookup 完成**。输出报告这批记录来自 16 个不同 `process_instance`，并不提供到真实 rank、设备序号或所有 worker 的经过验证映射。日志共 1,048,084 bytes，并有 `trace_cap`；总记录中 1,513 begin、1,490 end 并不能推导本窗口丢失数、配对成功率或实际总调用数。window 记录丢弃、graph replay 绕过 Python、并发锁丢失均可能存在。

`audit_live_metadata.py` 没有检查 `progress.run_id` 与硬编码 run 相同，也未将每个 begin 的 process-instance 与状态文件的 run ID、安装证明逐个关联；`21–24` 读取所有能解析的 status，`25` 的不完整计数将缺字段按 false 处理，且未记录解析失败文件。因此 19 status、17 个 incomplete 是该聚合所见的**可解析文件摘要**，不能作为全 worker 完整性或 16 rank 覆盖证明。独立的同次 `capture-summary` 显示 progress 的 run/window 与上述一致、19 个 status 文件都通过 run ID 检查，但也明确标记 rank、graph、窗口 ack 和日志完整性 UNKNOWN；这增强同次关联，不补上逐 process 精确身份和生命周期。

**工作负载关联。** 同次 launch 记录中 observer SHA `b87308a636201eebd97b48b605382409a05d8e661eea471bd811d819ab710d48`、客户端 SHA `b0dd42ee840c2a66d1dea13fcefa42056622bd931f9d3b760947e3795333fb9a` 与保存源码相符；launch 标记本 adapter 未启 profiler。该客户端 `phase:120–151` 核对冻结 seed、requests、references 文件 SHA，结果记录测量请求 8/8、输入 262144 token、输出 8192 token、前缀查询 262144 / 命中 235904、0 抢占，8 个输出与冻结 reference 字节匹配。这是 **当前原实现的诊断一致性检查，不是删除 copy 后的正确性结果或性能基准**；进程/时间记录完整性和其他隐性 profiler 不由这条标志独立保证。

**仍需验证的条件：**（1）实际 lookup 路径及两种 Triton 后端对输入 view 指针、storage offset、非连续 stride 的寻址是否正确；（2）producer 写入完成、异步 lookup 最后读取与 buffer 复用/释放之间的跨流依赖与生命周期；（3）graph capture/replay 中输入与 tensor 地址、动态 shape 的约束；（4）源张量与其他写者或消费者的实际 alias，包括 buffer 复用、DP/TP 数据交换及顺序。匿名 Python storage wrapper 身份不是物理存储或设备生命周期证明。须在相同数值契约、零/边界/无效索引及并发场景核对输出；不得把此新运行反填为上节第二次历史 profile 四个 copy 锚的实测 stride，也不得将本次未记录数据推定为同一布局。

复核只读本地已保存聚合、capture-summary、launch 与冻结源码，没有再次采集或操作 NPU。`audit_live_metadata.py` SHA256 `db86b5d4b606925a1004f651e3622d161a4777bc70359c1f3b8fd8cbb3b94083`；`audit_live_metadata.json` SHA256 `1865b172050fe6e66807e707a84d45971e8788ff77a0242ddc553ea544e50a13`，其中原始 JSONL 的 SHA256 为 `b32f03b3aca61939fbf76e56a6c7faaa0d1d8a9c4d225515f03e72082c44cfbf`；`capture-summary-001d14496b2146bba25dae0f27c780b5.json` SHA256 `cf46e2898bee6478091532cc592156063e6a8c5fb86ab520aedee0549a8927d8`；`launch-001d14496b2146bba25dae0f27c780b5.json` SHA256 `12de36efbeca0a707e3f8e2549aa7bfe6497aa4b3e3a6f2bdc07c18ceb4f1b06`。聚合逻辑已审，原始日志未在本次独立重新解析，故计数沿用已有输出并保留上述缺口。

## F25 非线性重组的特殊值与小值证据

上述 F25 已指出必须保持 `signbit`、`clamp`、`sqrt` 和 sigmoid；这里补充可复算反例，防止把不同公式误写成等价融合。冻结 `engram/common.py:68–72` 先得 dot，再计算 `m=sqrt(max(abs(dot),1e-6))`，以 **signbit** 选择 `-m` 或 `m` 后 sigmoid，然后才 mask、乘 value 和加原 hidden。源码 SHA-256 `4797b8962264d137cdb142a9988825288283092fb797ddbd334c6ab2223eb58d`，上游 `engram/layer.py` SHA-256 `dc9457c88aa3c90133115bc608447e037b0693f8db71bb8da37f1ead44021f80`。这属于普通 gate 路径的条件源码分析，不证明历史设备分支、每次 dot 值或设备对特殊值的具体执行模式。

- 把 `signbit(dot)` 换成 `dot < 0` 在 `dot=-0.0` 时不同：IEEE 符号位为负，而 `-0.0 < 0` 为 false。以通常实数 sigmoid 近似计算，原 gate 为 `sigmoid(-0.001)≈0.4997500000`，比较法为 `sigmoid(+0.001)≈0.50024999998`。如果选择同一完整非线性后继续处理，也不能单凭「两个零相等」认定原表达式可替换。
- 把条件分支替换成 `sign(dot) * sqrt(max(abs(dot),1e-6))`，即使普通 `sign(±0)=0`，它在 `+0.0` 和 `-0.0` 都给 sigmoid(0)=0.5，不等于上面的两个原 gate 值。把 clamp 去掉也不是一般等价：`dot=1e-8` 时原 sigmoid 输入为 `0.001`，无 clamp 则为 `0.0001`。这些差异可能经过后续 value、加法及 BF16 舍入被放大或消除，**不能据此声称最终输出必然不同或实际精度受损**。

CPU 复算方式，使用 Python 双精度 `math`，只用于示意符号与分支，不代表 PyTorch/NPU 的 sigmoid 位级值或设备是否产生/保留负零：

```sh
python3 - <<'PY'
import math
for x in (-0.0, 0.0, -1e-8, 1e-8):
    m = math.sqrt(max(abs(x), 1e-6))
    sigmoid = lambda z: 1 / (1 + math.exp(-z))
    original = sigmoid(-m if math.copysign(1.0, x) < 0 else m)
    comparison = sigmoid(-m if x < 0 else m)
    signed = sigmoid(0 if x == 0 else math.copysign(m, x))
    print(repr(x), original, comparison, signed)
PY
```

可研究的是保持同一符号位判定、最小幅值、精度和 mask 次序，把上述 pointwise 串接到 gate 归约或 residual epilogue；已有归约、数据依赖、NaN/Inf、rotation 和前面列出的数值及事件约束仍全部适用。若实际目标只要求误差范围而非逐位一致，应明确验收契约，并单独验证真实 dtype、特殊值行为和输出；不能仅用此 CPU 示例断定某种 NPU 实现失败。该补充没有新增设备工作或推理实现修改，也不能估算收益。


## 跨层 ID 视图联合读取：必须保留逐层 ready 的候选

冻结 `model.py:1199–1226` 先为所有 Engram 层调用一次 `gather_engram_hashes`，再按配置中的 layer 顺序依次读取 `gathered[:,slot]`、处理该层 table、必要时完成各自 DP exchange/TP gather、写回输出，最后记录该 layer 自己的 ready event。模型消费者 `model.py:1565–1589` 在首次使用 mask 和该层 lookup 前分别等待对应 event，而不是等待整个 lookup 集合。因此，在 source 和角色绑定成立时，两个 slot 的 ID 已有共同 producer，可以研究在一个 lookup 设备阶段联合读取多个 slot；每个 slot 仍需独立选择对应 table、head 起止、vocab 范围、scale、输出、mask 与通信结果。

这与前面的“单层 view 不复制”不是可以独立相加的收益：两者可能消除同一 ID 中间结果或共享同一次读取。近期观测的 `[T,24]`、stride `[48,1]`、offset 0/24 与连续 `[T,2,24]` 的两个切片**相容**，但不同 observer 记录未证明同一次调用、共用 storage、两个 slot 均就绪或表指针相同。不同层有独立 table 对象引用，不能合并 IDs 后用一张表假定相同的查表结果；仅当实际后端能力、同 dtype/参数和 table descriptor 证明兼容时才可研究联合。

**逐层就绪是反例边界。** 假定 slot 0 的 lookup、必要通信完成的时刻为 `t0`，slot 1 的 lookup 尚未完成。如果将两层工作简单装进一个只在所有工作结束后才能发布的单 kernel，并且只在此后记录 slot 0 的 ready，则 slot 0 消费者必须等待整个联合 kernel；它可能错过原本 `t0` 后与模型前续计算的重叠。减少 launch 数本身不证明端到端更快。要保持原可观察数据依赖及不无依据牺牲已有 overlap，需明确逐 slot 的完成可见性、event 可以在哪个粒度发布、下游 collective 的组和跨 rank 顺序、output 生命周期与 graph replay；或者证明等待整组对该工作负载不延长关键路径。不能在不支持设备级子任务完成事件的情况下假定一个 kernel 中途可向另一个 stream 发布安全的部分结果。

直接 output-buffer 分支和 `embed_gathered` 分支各自的通信、padding 与表实现不同；0 token、无有效 head、dummy hashes、两个 layer 的实际先后、共享表/独立表及异步源缓冲复用需分别检查。重排会改变调度和资源竞争，甚至影响两层 ready 时刻，不证明数值相同或性能有收益。此为同一 F24 重组边界，无新增 NPU 采集或实现修改。

## F24：UVA 的非本分片 ID 与无效输出发布

在已有 F24 的 head 分片、stride、buffer 与逐层 ready 约束之上，再检查当前 stage 保存的 `engram_lookup.py`。源文件文字保存在 `diagnostic-20261029/engram-lookup-source.json`，JSON SHA-256 `09411c44c991b765a2e64e27d99c339ab3fede5e11b28b6931e1258d0234bc01`，其中源码 SHA-256 `c2b1b46ea4eeb70513cb4a53a5fc584bd024c81269fff9f997431a3ff172899b`。这说明所保存源码的具体实现，**没有把它绑定到历史运行的 kernel binary、分支或数据**。

host-UVA 路径对一个 `(token, local_head)` 读取该 head 的 global ID；若 ID 不在 `[vocab_start, vocab_end)`，源码先将 `local_row` 选为 0，然后按 `CHUNK=2^22` 行计算 pointer 表索引和局部偏移，表达对 row 0 的 `tl.load`，在 `QUANTIZED` 分支再表达 scale 读取，最后用 `tl.where(owned, result, 0)` 对本输出行写 BF16 结果。device-table 的相应 kernel 同样先将非 owned ID 映射为 0、表达读取、再写零。若 owned，逻辑行号为 `ID−vocab_start`；按源码分别读取该 shard 的行与对应 scale。仅凭这些源语句不能判断实际编译器是否消除、合并或重排读取，也不能判断 runtime 中有多少非 owned ID、任何外设事务、缓存命中或等待。

量化语义需随数据一起保留：普通 INT8 分支使用 FP32 group scale，MXFP8 分支读取 UINT8 E8M0 scale 并按源码特别处理编码 0 和 255；最后均转换为 BF16。以 `WIDTH=256`、`GROUP=32`、完整一行以及输入格式确实对应这些分支为**条件**，分别有 256 个 code 元素和 8 个不同 scale 元素；不同源码分支的 scale 加载表达式不同，同一 scale 地址被多列引用不代表多少物理读取事务。BF16 非量化分支不读取 scale。上述大小与 `[3,6144]` WKV 输入的逻辑维度相容，但不能仅凭维度证明运行类型、参与分片、具体内存流量或 `wkv` tensor 身份。

可研究在判定 `owned=false` 后屏蔽无用的代码和 scale 读取，但仍必须向指定目标输出 **写入正确的零**：caller 可以传持久 output buffer，不能假设它在每轮都是零或跳过写入就等价。owned 行的 ID/stride、chunk 映射、scale、转换精度与所有输出顺序必须不变；输出完成、TP head gather、逐层 ready 及最后消费者的依赖也须保持。若使用掩码或分支，需要核对真实后端对指针、异常值和 masked load 的语义。该候选仅针对源码的非 owned 路径，不是已实现融合、真实无效行占比、必省 HBM/UVA 字节、性能收益或理论性能界；现有已存观测中的 lookup task 名称也不能证明具体参数或该源码分支。没有新增 NPU 负载或修改推理实现。

本机 ZCode `zcode/engram-invalid-index-read.zcode.json` 对保存的源码独立复核了上述两条路径的范围判断、row 0、条件 scale 和零值写出；本机重新计算 source hash 并核对语句。它只验证源码文字的限定逻辑，不验证历史运行或优化结果。


## 将 head all-gather 改为投影局部和归约的条件通信示例

冻结配置 max_ngram=4、8 个 head、head_dim 256、hidden 5120、hc_mult 4，故完整输入为 24×256=6144 特征，WKV 逻辑输出为 (4+1)×5120=25600。在实际 TP=8、Engram num_shards=8（例如 DP shared 或其他使 DP size=1 的分支）且均匀分配成立的条件下，每 rank 持有 3 个 head、768 个 BF16 元素；embedding.py:333–355 在 TP>1 时有 head all-gather，layer.py:23–29,41 再进行完整 WKV 投影。历史采样未证明最终 resolved group、token 行映射或具体通信算法。

构造两种完整输出都在每 rank 可用的条件模型：A 为标准 ring all-gather 各 rank 的 768 个输入元素，再做投影；B 为各 rank 对本地 768 特征做 25600 个输出的部分点积，再用标准 ring all-reduce 合并完整 partial。忽略协议、padding 和额外元数据，按每 rank **单方向发送量**计，A 为 7×768=5376 元素/token；B 的 reduce-scatter 和 all-gather 各发送 7×(25600/8)，共 44800 元素/token，是 A 的 25/3≈8.333 倍。三个 token、BF16 partial 的单方向逻辑发送分别为 **32256 / 268800 B**；FP32 partial 的 B 字节数再乘 2。若只是比较张量总形状，25600/6144=25/6，与上述具体 ring 两阶段通信比不同。

这是比较两种**明确假定的算法**，不是历史链路流量、网络性能或所有可行方案的下界；输出只保留在部分 rank、reduce-scatter 结合后续消费者、压缩、分块或不同通信算法均可能改变比较。部分和归约顺序及 BF16 舍入也须符合原 WKV/gate 精度约定，不能单靠代数等式断言与现有实现等价。此例反驳“先局部投影就必然减少通信”的笼统判断，不断言原方案更快或无法优化。

复算 `python3 agents/fusion/engram_comm_order_witness.py`，SHA-256 `e5009611633b3f78f87eeb96c025b6d1d887d259ee2f3b9e2defd8b629e08f62`。冻结 embedding.py SHA-256 `a7741ed16522be4fe607e062e1c988c4e2435af7d258fe7db99f672d4508c4c9`，layer.py `dc9457c88aa3c90133115bc608447e037b0693f8db71bb8da37f1ead44021f80`，配置 evidence/model-config.json `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`。本机只做整数算术，不启动 NPU。

## F24：host-UVA chunk 指针表与宽地址计算的条件重组

冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/npu.py` SHA-256 `2b2594a2126ef50ee6d77f508ab8e0d2f5b360d15ea700eea0d39923af8bc64f`，`HostUvaBuffer:135–162` 和 `SharedUvaBuffer:251–312` 分别注册一段 host allocation 或 shared backing 的本 rank 映射，各自调用 `aclrtHostGetDevicePointer`，再以该 rank 的返回地址 `base` 建立设备侧指针表 `ptrs[k] = base + k * CHUNK_ROWS * row_bytes`，其中 `CHUNK_ROWS = 2^22`。共享 host backing 不意味着多个 rank 的 device pointer 地址相同：每个 rank 独立注册并取得地址。分别保存的 `diagnostic-20261029/engram-lookup-source.json` SHA-256 `09411c44c991b765a2e64e27d99c339ab3fede5e11b28b6931e1258d0234bc01`，所嵌 kernel 源码 SHA-256 `c2b1b46ea4eeb70513cb4a53a5fc584bd024c81269fff9f997431a3ff172899b`。其 90–118 行先把 owned global ID 换为 local row，再选择 chunk、读 codes 的地址表和对应代码值，量化时还读取 scales 地址表和尺度。非 owned ID 的零输出边界仍按上一节处理。

一个有限的结构研究对象：假定 **CANN 返回的 device 地址对整个注册分配均支持连续字节寻址**，且地址和整数运算无溢出，对合法行 `r` 和合法列 `d`，整数算术中 `ptrs[r//C] + (r%C)*row_bytes + d*element_bytes = base + r*row_bytes + d*element_bytes`。据此可研究向 kernel 直接传每 rank 的注册基址，避免按 chunk 动态索引指针表。此等式是数学条件，不是该平台的映射、指针运算或性能保证；代码逐项构造地址表不能单独证明设备映射整体连续、Triton 后端支持所需宽偏移、编译输出消掉指针读取或 actual loads 发生外存访问。

codes 和 scales 必须分别维持各自注册基址、行宽、元素 dtype、group/尾块、拥有区间、非法 ID 输出零、原 BF16 结果与特殊 E8M0 编码，以及 graph/stream、buffer 最后消费者和注销先后。若直接将基址存为新增字段，还需验证创建、重映射、失败清理、rank 生命周期和 graph 重放是否一致。地址表可能驻留缓存；省一次源码级间接访问不等于省一次 HBM/UVA 事务，亦不证明更快。实际历史是否使用 host-UVA、各 rank pointer、可访问范围及真实 shape 未绑定，不能据此断言历史命中或估计收益。配置的 `num_embeddings` 是容量，而 `embedding.py:93–111` 仅要求 `sum(head_sizes) <= num_embeddings`；在没绑定运行时 `engram_layout.primes` 和 shard 信息时，不可据总容量除 rank 数断言每个 shard 需要多少 chunk。

**偏移宽度反例（CPU 整数模型）：**假定每行宽 `WIDTH=256`，对 valid local row `2^23` 先用 signed int32 保存 `row * WIDTH` 的元素偏移，则数学结果是 `2^31`，二进制 32 位有符号重解释后是 `-2^31`；以 `CHUNK=2^22` 分解，该行位于 chunk 2、局部行 0，原结构的 chunk 内元素偏移为 0。下列代码不读设备或内存：

```python
import ctypes
C = 1 << 22
width = 256
r = 1 << 23
assert (r // C, r % C) == (2, 0)
assert r * width == 1 << 31
assert ctypes.c_int32(r * width).value == -(1 << 31)
base, element_bytes = 1 << 40, 2
row_bytes = width * element_bytes
for row in (0, C - 1, C, 2*C - 1, 2*C):
    chunk, local = divmod(row, C)
    for col in (0, width - 1):
        table = base + chunk*C*row_bytes + local*row_bytes + col*element_bytes
        direct = base + row*row_bytes + col*element_bytes
        assert table == direct
```

上述反例明确假设错误地把**元素偏移**收窄为 signed int32，不说明任何选中历史 ID、该行存在于实际 shard，或当前实现发生整数溢出。保存的 kernel 已显式将 ID 转为 `tl.int64`；须进一步确认后端 lowering 和地址 ABI，不得因为此构造称现有 kernel 有 bug。本次只读源码和本机 CPU 算术，未访问 NPU 或修改目标实现。
