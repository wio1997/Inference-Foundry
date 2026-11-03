# F21：复合 MC2 内的 dispatch → 第一矩阵乘 → 激活/量化流水线边界

这是对已有 F20/F21 的条件性结构重组分析，不是新增设备测量，也不把当前复合调用内部阶段重复记成未融合算子。此前 `LAYER0-MC2-DP-RECOVERY.md` 已指出内部 flag 和动态专家数是低 AIC 活动的可能机制；这里补充一个具体的程序顺序及可探索的改造边界。**冻结源码与历史草稿 `DispatchFFNCombine` 的确切编译对象、tiling 和执行路径未绑定**，因而以下仅描述所保存的实现，不证明某个历史草稿调用实际执行该分支，也不证明这段顺序是性能瓶颈。

## 代码中已有的流水线

`dispatch_ffn_combine_kernel.hpp`（SHA-256 `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`）中，AIC 依次执行 GMM1、等待 AIV、GMM2（218–222）；AIV 执行 `DispatchAndCombine`（228–230）。AIV 在路由与跨 rank 计数、累计后，按本地专家组从各 peer 搬送该专家的数据和 scale，组末 `SyncAll` 后向 AIC 发送该组 ready flag（801–866）；AIC 的 GMM1 在其被分配 tile 计算前补足对应组 flag 等待，最后还排空剩余组等待（450–529），具体粒度见文末。因此源码**已经提供按专家组的 dispatch 与 GMM1 重叠机制**，不能再把二者重叠作为全新、完全缺失的机会。

GMM1 在 `epilogueGranularity` 处（如果该组不是最后一组）发送第一次完成通知，全部专家组完成后发送第二次（508–535）；AIV 先处理 `dequantSum1` 再处理可选的 `dequantSum2`，分别通知 GMM2（920–954）。`dispatch_ffn_combine.h`（SHA-256 `d2b5712ad97f7505233f340e18660627424e5e1fe4c33be68f8e34b51a479ab1`，266–270）对每卡专家数大于 4 使用 `expertPerRank - 3`，否则 `expertPerRank - 1`；这是保存源码的静态赋值，不等同于历史调用参数已知。GMM2 的启动还受 AIC 顶层程序顺序限制：**单个 AIC 的 GMM1 结束并等待 AIV 首个通知后才进入 GMM2**，不是收到第一个 GMM1 完成通知就能立即开始 GMM2。

## 尚可研究的重组边界

在现有 AIV 程序中，`dequantSum1/2` 在整个专家组搬运循环内逐组累计，但 AIV 要退出该循环、等待两个 copy 事件后，才会等待首个 GMM1 完成并运行第一批 `blockEpilogue1`（842–890、920–931）。即使 AIC 对前一批 GMM1 已发出完成通知，**当前源码顺序不允许该 AIV 先于整个 dispatch 搬运循环结束执行第一批激活/量化**。这不意味着两者实际相差多少，或 AIV 此时空闲。

可研究把 AIV dispatch 与激活/量化划分为能独立推进的任务组：当一批专家对应的跨 rank 输入已就绪、GMM1 对该批写入完成，并且其 scale 和行范围都完整时，开始这批 activation/quant；同时另一任务组继续搬运后续专家。类似地，可探索更细批次使首批量化数据更早可供 GMM2；但 AIC 顺序本身也必须研究，单改 AIV 并不保证 GMM2 更早运行。这是对已经融合的 MC2 **内部重排**，不是宣称当前没有融合或保证提速。按照现有代码，整 AIV 团队多处执行 `SyncAll<true>()` 和 cross-core flag，不能简单把原循环拆成两组后让一组跳过同步：须设计并证明新的参与者集合、事件计数、内存可见性和无死锁协议，且确保 GMM1/GMM2 各 tile 的 producer 完成后才可消费。

正确性边界：保留每个专家收到的全部 rank 行和原排列、capacity 截断与零行行为，保持 `expandedRowIdx`、每 token 量化 scale、权重 scale、bias、SwiGLU limit、两个 GMM 输出的 dtype/舍入和最终概率加权 unpermute；每 rank 不得改变 peer 交换/收敛的必要次序，也不能提前覆写待消费的 workspace。跨 expert 分批可能改变资源争用、buffer 生命周期和数值顺序；是否允许不同浮点舍入取决于明确的精度契约。这里没有证明任一候选数值等价。

## 对照 SGLang：跨 microbatch 隐藏通信的预填充候选

SGLang [固定版本的 DeepSeek V4 策略](https://github.com/sgl-project/sglang/blob/3b6c28b0b9aff784115bafeecc67f37e1fe43454/python/sglang/srt/batch_overlap/operations_strategy.py#L164-L228) 在 prefill 的 EP 路径把 dispatch 和 combine 拆成前后阶段，并在两处让出执行权，以便交错另一 microbatch；同文件明确该 V4 策略的 decode/target-verify 尚未实现，且 TBO 模式关闭跨层 mHC 融合。这与本任务 MC2 内部的按专家组流水是不同层次：可研究让一个 microbatch 的 MoE 通信与另一个 microbatch 的 attention 或专家计算交错，而不先假定单个复合内核内部发生了什么。

这不是 169 的现成功能或加速证据：当前 vLLM 的 `DispatchFFNCombine` 是复合调用，须先核实 910C 后端是否能安全异步拆分或并行完整调用，以及是否存在两个实际可并行的 prefill microbatch。90% 前缀命中会影响可供并行的剩余预填充工作量；不能把另一框架的吞吐提升直接套入。还须保证所有 rank 的 collective 顺序一致、不同 microbatch 的 buffer/event/cache 互不覆盖、mHC 和量化语义不变，并考虑额外显存、调度与共享资源争用。**最低成本下一步**先离线检查本机保存版本的 dispatcher 接口与相关调度记录；只有确认接口和 batch 条件后，才考虑在空闲隔离资源中做一次带控制组的预填充对照。Decode、target verify 及历史运行适用性均为未知。

## 如何判断是否值得推进

需先从已有捕获查能否绑定具体历史 MC2 调用到该已编译对象、专家组行数和参数；若无法绑定，仅保留源码条件候选。要区别 dispatch 搬运、GMM1、激活及 GMM2 的真等待时长，还需要同一次调用的内部阶段时间、ready/flag 或可对应的硬件观测；现有整体 `DispatchFFNCombine` task 与 AIC MAC 周期代理不能拆出这些时长。分组之间可能争抢 AIV、搬运引擎、缓存和带宽，调度/同步可能反而增加；没有这些数据不能给收益或理论上界，也不能判断此次低活动是否由该边界导致。

## 内部重排的 workspace 别名与 scale 发布条件

上述拆分候选还有两个具体存储冲突，不能只补细粒度 ready flag。`dispatch_ffn_combine_kernel.hpp:1105–1133` 对同一 `workspaceOffset` 依次赋值 `ptrC` 与 `ptrC2`，之后对另一相同 offset 赋值 `ptrA` 与 `ptrPermutedToken`。`initBuffer:240–246` 分别把它们绑定到 `gmC/gmC2` 和 `gmA/gmPermutedToken`；GMM1 读取 `gmA` 并写 `gmC`（497–503），activation 从 `gmC` 读并向 `gmPermutedToken` 写（923–931），GMM2 从 `gmPermutedToken` 读并写 `gmC2`（611–617）。这证明**所存实现的这两对 GM 起始地址相同**，不需假设 allocator 是否巧合地复用；具体区间重叠仍要按实际 shape、dtype、layout、偏移和 tile 计算。

`dispatch_ffn_combine.h:190–200,216–219` 的该源码模板将 A、D1 设为 INT8，C 设为 FP16 row-major；`kernel.hpp:544–545` 设 `n2=K`、`k2=N/2`。给一个明确的地址反例：假设连续 row-major、当前行宽 K=5120、N/2=2304 且第一路 GMM1 还未完成读取第 0 行，那么 activation 的第 0 行 INT8 输出从相同起点写 `[0,2304)`，与 GMM1 输入第 0 行 `[0,5120)` 重叠；提前写会覆盖仍需读取的输入。类似地，GMM2 向 `gmC2` 写入的字节范围如与 activation 尚未读取的 `gmC` 范围相交，也会覆盖第一矩阵乘结果。该反例只说明不受控的提前写不安全，不表示现有顺序有 bug 或历史调用遇到冲突。一个具体的新流水协议至少须逐区间确认 GMM1 对相关 `gmA` 的**最后读取已完成**、GMM1 已写出的对应 `gmC` 已被 activation **最后读取**，然后才能复用对应范围；消费侧和生产侧跨核可见性、异步 block 完成与空专家/容量截断也需证明。对于不同组，必须算真实行起点及字节区间，不能仅根据“专家组不同”认为地址不交叉。

此前仅列举了 dispatch/activation 各自在 AIV 上的步骤；现有同版本文件 `evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_epilogue_pertoken_swiglu.hpp` 的 SHA-256 为 `baaacd84dc4c29d5efe72b7c7bd3b0cac64db233bc9431bd6fb678390cf2a0ad`。dispatch 的 `CopyGMToGMPerToken:323–353` 使用当前 `resource.ubBuf` 的字节起点 0 与 96 KiB 的 ping-pong 区，和 `MTE3_MTE2`、`MTE2_MTE3` 事件；该 epilogue 构造函数的 `ubOffset` 初值为 0（97–121、312），从同一传入 resource 的 UB 起点分配 C、D 和 FP32 工作区，并使用自己的流水事件。如果同一 AIV 核上交错这两段，起点为 0 的局部区域会互相覆盖；必须重分配不重叠缓冲并核实事件复用及内存次序。不同 AIV 核是否能各自承担不同任务还受 `SyncAll<true>()` 参与者和启动协议限制，不能仅因为每个核有局部 UB 就假定现有代码可直接分组。

相同 epilogue 的 `158–179,187–204,248–294` 进一步说明**数据和 scale 分开发布**：每个参与核遍历被分配的行，每行对完整 `ChunkTileLen=N/2` activation 结果取绝对值最大值、计算该行的 scale，并写 INT8 行；scale 被临时保存，直到这个核所有分配行的循环结束后才批量写出 `gmPerTokenScale2`。因此行的 INT8 数据写出本身不是完整 GMM2 输入就绪的证明：相应行 scale 也必须写完并按实际设备内存语义对消费者可见。现有 kernel 在 activation 批次后 `SyncAll<true>()` 并以 `PIPE_MTE3` 发送通知（933–935、952–954），其确切跨核/流水语义及历史二进制仍需绑定；不能仅从源码顺序估计等待耗时。若设计更细行发布，须为每行提供对应 scale 的明确发布及 buffer 最后使用协议，保留全 2304 特征的最大值归约和原 INT8 舍入，不能以部分 tile 的最大值代替整行 scale。

这些发现是该源码所示的约束，与现有 F21 的同一流水候选重叠，不计为独立收益。若历史 MC2 的编译对象、tiling 和动态专家行未绑定，则它们不能解释历史低活动或计算速度；更不能把样本中的物理维度直接当作有效专家行数。本次只检查已存文件，不启动 NPU、不修改实现。

**跨行别名的具体反例：**在第一组至少有两行、两次输出均为连续无填充 FP16 row-major、复用 `ptrC/ptrC2` 共同基址且不存在另设块缓冲的假设下，取本节已有示例 `K=5120, N/2=2304`。GMM1 输出宽为 4608，前两行字节区间分别为 `[0,9216)` 和 `[9216,18432)`；GMM2 输出宽为 5120，若完整写入第 0 行，其区间为 `[0,10240)`，会覆盖 GMM1 第 1 行的前 1024 字节，即 512 个 FP16 元素。因此，即便 activation 已读完 GMM1 第 0 行，且 GMM2 第 0 行输入与 scale 就绪，也不足以允许其全部输出覆盖共享缓冲：还须确保被覆盖的第 1 行相应数据已完成最后读取。列 4608–5119 与所存 256 列 tile 大小相当于最后两个 tile 的范围，但实际块写序和可见性仍需核实。

这一反例是对现有别名约束的细化，不证明历史运行发生冲突或原程序有问题。第一组的两个 group offset 从零开始；后续组的偏移随不同输出宽度累积，必须分别计算。冻结源码使用 row-major 类型、相同 workspace 基址和按组构造的 layout，与这个假设相容，但 `LayoutC` 的具体 offset 实现、实际实例的有效行数、填充和二进制尚未绑定；不能把配置中的 5120/2304 自动当成该历史 routed MC2 调用的已证参数。安全重排须按实际物理字节区间和各消费者最后读取建立依赖，并考虑独立缓冲可能改变此约束。

## GMM2 → Combine 的现有等待粒度与前移条件

F23 已分析 combine 的输出通信、逐 token 合流与跨 rank 可见性，本节只核对上游 GMM2 交付给 combine 的具体源码控制流及前述 `gmC/gmC2` 别名；不是另一个可独立记收益的融合候选。对这份源码，`problemShape.m()*topK <= 4096` 选择 V2，否则 V1（`kernel.hpp:253–256`）；该条件仅使用 tiling 参数，未证明历史二进制或实际调用参数。

- AIV 在任何 combine 分支前，先执行条件性第二批 activation、`SyncAll<true>()`、第二次 `SYNCFLAGV2C` 通知，再执行 `blockEpilogue1.Finalize()`（937–963）。因此，在这份 AIV 控制流上，即使某一早期 GMM2 专家组已完成，也要先走完上述流程才能进入 combine；源码允许 AIC GMM2 在另一核上推进，但不证明实际时间重叠。
- V1 的循环先为每个专家组等待 cross-core flag，并执行 `SyncAll<true>()`，再读取该组对应 GMM2 与 scale、按 peer 路由行发出 epilogue（981–1021）。这只是代码结构，不推断跨核可见性和等待耗时。
- V2 按与 GMM2 对应的 scheduler 枚举 block，当前模板 `L1TileShape={128,256,512}`（`dispatch_ffn_combine.h:200`）；每个 block 的 M 维按两个 AIV 子核再分为 16 行小段，最后尾块行数按源码修正（`kernel.hpp:1025–1083`）。但 `syncLoopIdx` 在进入专家组循环前初始化为 0，**跨 tile 保留**，而 `CrossCoreWaitFlag` 仅在 `syncLoopIdx <= groupIdx` 的循环中执行（1027、1064–1067）。因此对于同一 AIV 上同一组连续两个待消费 tile：首 tile 使计数推进到 `groupIdx+1` 后，第二 tile 不再执行这条按组循环的 cross-core wait。这是否与其他隐式底层生产者同步相配合，不能只看本文件判断；不能把它描述成“每个 tile 都有独立的 cross-core ready 等待”。V2 epilogue 的 MTE2/V/MTE3 标志是局部拷贝、计算和双缓冲事件，不能自动等同于针对 GMM2 每个 tile 的生产完成标志。

GMM2 在各专家组的每个 AIC loop 中，仅在条件 `loopIdx + coreNum >= coreLoops` 时把 `syncLoopIdx` 置为组号，并将该值传给 `blockMmad(..., syncLoopIdx, 0)`；最终 `Finalize(expertPerRank - 1,0)` 仅出现在 V1（538–635）。所存 wrapper 指定 `enableUnitFlag=false` 并选择 `MmadAtlasA2PreloadAsyncFixpipe`（`dispatch_ffn_combine.h:175–176,209–213`），但当前本地材料尚未绑定 `blockMmad` 模板内部如何解释该参数、何时发布每组 flag、等待覆盖哪些写入。故目前只能确定源码中的消费侧等待次数及参数传递，不能单据这些片段认定它是整组完成、每 tile 完成或某固定时刻可见，也不能据此判断时长和吞吐。

V2 已把 **GMM2 输出读取、逐行 scale、转换及按目的 rank 的块写回**组织在同一个 epilogue 路径（`block_epilogue_pertoken_v2.hpp:123–228`），不能把这些当作当前缺失的独立 fusion。若探索另一 AIV 组在早期 GMM2 区间就绪后提前进行 combine，仍须逐个实际块证明：对应 GMM2 写已完成、`gmPerTokenScale2` 对应行已发布且可见、`gmC2` 范围没有被别名 `gmC` 的晚到 GMM1 写入覆盖、也不会覆盖 activation 尚未读取的 `gmC`；再保留由 tokenPerExpert / preSumBeforeRank 决定的目的 peer 与偏移及末尾 unpermute 的全部贡献。前述 UB 和事件复用、本地同步参与者及跨 rank payload 可见性也不可跳过。将分组划得更细不会自动证明上述约束成立。

F23 原“按专家组和 tile 等待并执行”已改成“按专家组等待并按 tile 消费”，避免误读为每 tile 都存在独立生产者 ready 协议。以上仅基于快照的控制流和条件，不说明历史二进制、执行分支、真实 overlap 或性能收益；未访问 NPU，也未修改目标实现。

## 已存 BlockMmad 模板：条件性生产者信号说明

补充前节的模板缺口：本地既存 `evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp` SHA-256 `a417d5287e9e6b8f693b8cdce522de73f7eb0e844669f76a448a66011224e30c`。已存 `dispatch_ffn_combine.h:43` 包含 kernel，kernel 在 `HCCL_COMM` 分支通过 `utils/` 包含此模板（kernel:41）；两处保存的 kernel 文件哈希同为 `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`。模板:34–77 对 `BlockMmad<MmadAtlasA2PreloadAsyncFixpipe<...>>` 提供特化，与 `dispatch_ffn_combine.h:209–227` 的 policy 和 BlockMmad 选型相符。`!HCCL_COMM` 分支采用仅文件名的 include，实际 include 搜索路径尚未闭合；历史编译宏、源文件、二进制均未绑定，不能断言采样运行此模板。

在 **未定义 `__TILE_SYNC__`，并且该次特化实际使用所存文件**的条件下，模板 `L1TileMmad` 在当前 block 最后 K 段先安排 `copyL0CToGm`，再调用 `Finalize(params.syncLoopIdx, params.flag)`（470–508）；GMM2 的 `BlockMmad` 以默认 `flagPtr=nullptr` 构造，因而 `Finalize` 对本对象的 `syncGroupIdx` 到 `target`，每个未发布的组号使用 `CrossCoreSetFlag<0x2, PIPE_FIX>(group/15+flag)`（155–160、275–293）。GMM2 调用传 `flag=0`，并在**本 AIC 分配的该组最后一个 tile**满足 `loopIdx+coreNum>=coreLoops` 时将 target 设成该组号（kernel:593–617）。结合 V2 消费侧按组推进的等待，这比前节“模板内含义未知”更具体：在上述编译条件下，该 producer 以本核已处理的组进度驱动发标志，而不是显式对每个 tile 发布唯一的组内序号。不同核无该组 tile、空专家组、preload 缓冲与提前发送多个欠缺组号时的配对仍须完整验证；代码行先后也不单独证明对另一核的数据可见或整组所有生产者已经完成。

源码还存在另一编译分支：若定义 `__TILE_SYNC__`，模板最后 K 段只在 `params.flag > 0` 时直接发送 `PIPE_FIX` 标志（501–505）；GMM2 显式传入的 flag 是 0。该条件下不能直接套用上述非宏分支的生产者解释；是否还有其他同步路径、该宏在实际构建中取值为何及分支正确性均未知，不得仅凭这一片段断言错误或该路径发生。此前前节“当前本地材料尚未绑定模板内部”的说法以本节补证为准，但构建与历史二进制绑定仍缺。

当前 wrapper 的 `enableUnitFlag=false`（`dispatch_ffn_combine.h:175–176,209–213`）在这一模板选中时使 `L1TileMmad` 使用显式 `M_FIX` SetFlag/WaitFlag，再调用 `copyL0CToGm` 和 `FIX_M` 事件（484–492），而非启用 unit flag 的另一分支。该源码顺序、预加载完成点和 `PIPE_FIX` 发标记同样必须结合匹配版本的管线与 cross-core 语义才能构成可见性证明；不能直接据此删除屏障、断言早发 tile 可被另一核安全消费或推导加速。

## 官方跨核同步资料：同一 AI Core 内的 AIC → AIV

[华为 CANN 商用版 8.5 的 NPU 220x 架构文档](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)列出 Atlas A2/A3，说明其 AIC、AIV 通过 GM 交换数据；mode 2 的同步域是**同一个 AI Core 内的 AIC 和两个 AIV**。该文档专门以 AIC 将 L0C 结果搬到 GM、AIV 再从 GM 搬到 UB 为例，说明配对的 CrossCoreSetFlag/WaitFlag 可确保该搬运顺序。[CANN 社区版 9.0 CrossCoreSetFlag 接口](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html)同样列出 A2/A3、mode 2 范围，并给出 AIC `CrossCoreSetFlag<0x2, PIPE_FIX>`、AIV 等待后处理的例子。文档还要求标志匹配并注意计数上限、kernel 类型和其他接口的标志冲突。

据此，**若**历史代码实际使用前述模板与非 `__TILE_SYNC__` 路径、所选 `copyL0CToGm` 确实按对应版本 FIX 流水写 GM，且 producer/consumer 的核映射、flag 和计数正确配对，那么官方文档的 L0C→GM→UB 场景可作为同一 AI Core 内 AIC 发出 `PIPE_FIX` 标志后、相应 AIV 消费该前序写出的条件性依据。这比前文仅依据代码先后推断更强；它不是对跨 AI Core 全体 AIC、跨 rank 远端 payload 或完整 token 的 barrier 保证，也不证明 V2 每个 tile 有独立 flag。代码中 V2 按 `get_block_idx()` 对应 AIC 并分配两个 AIV 子核是相容线索，但实际 kernel 配置、scheduler 映射、每组尾 tile、空组及多次 flag 的配对仍未在历史二进制验证；若生产者写入的是其他 AI Core 的数据，则不能仅靠本条 mode 2 证明对该数据的完整就绪。跨卡部分仍以 F23 的单独证据和未知为准。

资料版本分别为商用 CANN 8.5 与社区版 9.0；历史 profiler 自述 CANN 9.2.0，未验证所装头文件、编译宏、实际硬件 profile 和二进制与这两份文档一致。因此以上是**带版本、条件和参与域的接口证据**，并非历史路径的运行事实或可量化性能收益；本轮没有启动 NPU。

配对粒度再明确：按[CANN 社区版 9.1.X CrossCoreWaitFlag 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6/%E6%A0%B8%E9%97%B4%E5%90%8C%E6%AD%A5/CrossCoreWaitFlag_ISASI.md)的 mode 2 说明，在一对多方向上，单个 AIC 的一次 Set 对应同一 AI Core 内每个参与 AIV 各一次 Wait；反向则要求参与的各 AIV 都 Set，AIC 才可解除一次 Wait。flag 对应计数器为 0 时阻塞后续指令下发，非零时消费计数并继续；已下发的指令不能一概视为被回收或阻塞。所配置的 `PIPE_FIX` 与此处配对，只为对应核在该前序流水和接口契约范围内建立顺序，不证明所有其他 AIC 的 FIX 写、其他流水、未来写、跨卡 payload 已完成。实际路径必须验证参与核数、flag 号及计数的一一匹配，特别是同组多 tile、空组和跨组欠发场景；不能直接将 AIC 本核最后 tile 的标志扩大为全部核的组完成。该 9.1.X 文档也不是已绑定的 9.2.0 运行时规格；无法仅凭文档确认当前采样实际启用的宏或编译路径。
## GMM1 无 tile 的核仍需排空专家组信号

已有 F21 和 `LAYER0-MC2-DP-RECOVERY.md` 已覆盖动态 token per expert 与 AIC/AIV 分组通知；这里补一个**每核等待数量与 tile 分配不等价**的静态边界，不新增独立融合项。`dispatch_ffn_combine_kernel.hpp:450–452,454–484,517–529` 在每个 AIC 的 GMM1 中先等待一次 cumsum 信号；对于当前核有循环 tile 的组，第一次 tile 前会从 `syncGroupIdx` 到该组逐一补等之前未消费的组信号，组内后续 tile 不重复这一递增循环。全部专家循环结束后，不论该核曾经获得几个 tile，尾部都把剩余 `syncGroupIdx` 一直等到 `expertPerRank`。在循环正常到达尾部、无异常/提前退出的源码控制流下，这个核执行 **1 次 cumsum Wait 加 `expertPerRank` 次组 Wait**，其中等待可能立即满足，不能把次数等同阻塞次数或延迟。

与之相对，AIV `:809–833,842–867` 在 routing、跨 rank 计数和 cumsum 后发一次 cumsum flag，然后按 `expertPerRank` 遍历每组，无论该组复制行数是否为零，代码都经过本组 `SyncAll<true>()` 和一次组 flag 发布。示例：假设 `expertPerRank=3`，某个 AIC 在三组的 `startLoopIdx` 均不小于对应 `coreLoops`，该核的 GMM1 没有执行任何 `blockMmad` tile；它仍在循环后的排空段执行三次组 Wait，另有前置 cumsum Wait。这是根据源码的条件性控制流推导，不假定 `currentM=0` 必然映射到某个 `coreLoops`，也不证明 flags 当时尚未就绪、产生阻塞或该核空闲。一个核无 tile 不代表其他核、其他流水或整卡无工作。

这具体说明：低 AIC 活动可以与“少量/不均匀的 tile 分配”和“完成既有组通知协议”同时存在，但不能从 AIC MAC ratio 较低推出哪一个占主导；相同 Wait 调用数也不意味着相同等待时间。能否将空组的协议省去或合并，是同一 F21 内部重排问题：须首先证明所有参与 AIC/AIV 组、flag 编号和计数、当前与后续轮次配对、跨 rank 数据/scale 可见性、空组/容量截断、workspace 以及最终输出语义，不能只因 `currentM=0` 删除 Set/Wait/SyncAll。源码组总数、实际每核 tile 和时间可由绑定的运行参数、对应编译路径和同次内部 trace 证伪；现有采样没有绑定每专家动态计数或内部 Wait 时间，不能据此给出实际比例或收益。本节未操作 NPU。
## cumsum 就绪与 expert-token 输出拷贝的发布边界

既有记录指出 core 0 会输出 `expert_token_nums`；这里补查**该输出写入与 GMM1 启动信号之间的顺序**，不是另一次性能测量。冻结 `dispatch_ffn_combine_kernel.hpp:811–833` 的 AIV 控制流在跨 rank 计数后由 core 0 构造 `cumsumMM`，经 `SyncAll<true>()`，然后 core 0 将末行的 `expertPerRank` 个 INT32 拷至外部 `ptrExpertTokenNums`，之后每个 AIV 才提交 `CrossCoreSetFlag<0x2, PIPE_MTE3>`。`CopyGMToGM:260–309` 使用 MTE2/MTE3 和双缓冲事件，并在返回前发出相应 wait；精确控制流与流水完成范围仍以实际架构契约为准。AIC `GMM1:450–455` 先等 cumsum 通知后，直接读取的是 workspace 的 `cumsumMM`，**不是**刚写出的 `ptrExpertTokenNums`；GMM2 和后续合流仍有自己的 `cumsumMM` 消费。

这是一个不同于按专家组重排 dispatch/GMM 的条件研究边界：如果能证明 `cumsumMM` 的必要写入已经对配对 AIC 可见，且 `expertTokenNums` 输出及其消费者仍在原可观察时点之前完成，可研究将 core 0 的 cumsum-ready 通知与输出拷贝拆开，让通知不再被统计输出复制排在前面。当前源码中 core 0 的 `PIPE_MTE3` 发布排在复制之后；按文档所述 mode 2 配对，也须核对对应 AIC 的两 AIV 参与和各自旗标。不能据此说整个卡或全部 AIC 都等待该复制，更不能算出可隐藏时间：其他 AIV 的通知、实际流水语义、已排队工作及动态时序尚未绑定。

不能直接删除该复制。`moe_comm_method.py:587–614` 将该预分配输出作为 `expert_tokens` 返回；`routed_experts.py:333–354,752–775` 有条件用它记录 MRV2 或 dynamic EPLB 的专家负载。若某次真实计数为 1 而缓冲保留上次的 0，跳过输出会让启用的负载统计读到旧值；此为可证伪的状态反例，不假定该 profile 开启 EPLB。任何重排还须保持输出完成对外部读取的可见性、跨轮覆盖前的最后消费者、`cumsumMM` 在 GMM/复制消费期间不被修改，及 core 0 的 UB/事件安全：该复制和后续 `CopyGMToGMPerToken` 都使用同一 `resource.ubBuf` 的 0/96 KiB 与事件槽，不能未经新协议在同核交错。

若历史源码、具体 cumsum→flag 与同核 copy 事件被绑定，可用同次 flag/copy/AIC 时间顺序反驳或支持该候选；现有聚合 `DispatchFFNCombine` TASK、AIC 周期代理、输出 shape 和同名算子不足以识别这段内部耗时。当前既没有历史构建绑定，也没有该段时序，因此它只是受约束的具体重组机会，不是已证瓶颈或可兑现收益；未操作 NPU、未修改目标代码。

### 跨 microbatch 通信隐藏：实验前的工程账本

这项估算比较**相同总 token、相同输出合同**的两种排程；并非测量、物理硬上限或 SGLang 在 169 上的实际能力。先固定一次 prefill 能形成 `N` 个 microbatch，并确认它们在允许的语义下可交错；90% 前缀命中后的有效数据量、实际可形成的 N、每个 EP rank 的对应顺序仍需同次确定。每个 microbatch 的路由/dispatch、专家计算、combine 分别记 `d、c、b`，仅在大小相近且三阶段资源独立、允许下一个 microbatch 的 dispatch 与前一个的后续阶段交错时，理想工程模型为：

```text
T_serial = N × (d + c + b)             # 同一 N 个 microbatch 不重叠的参考
T_pipe_ideal = d + c + b + (N−1) × max(d,c,b)
T_candidate ≈ T_pipe_ideal + H + Δ_shared
```

第一项包含流水首块填充和最后块排空；`H≥0` 是相对于同 N 串行参考的拆分、额外 kernel/collective、事件、调度和 buffer 管理等开销；`Δ_shared≥0` 是通信与计算同时运行造成的额外资源争用延迟，需由共同运行条件估算或测量。该模型不包含重叠带来缓存改善等其他收益，也不保证该排程可在现有复合 `DispatchFFNCombine` 接口实现。`d、c、b` 是分别定义的阶段时间，**不可把完整通信往返作为 dispatch 后整个阶段**：真实 combine 依赖相应专家结果，必须放在 c 后。EP 组中每 rank 的 collective 次序、不同请求及层的依赖、块间 KV/slot 和事件需另行满足。

若用相同 N、相同微批工作的串行参考，候选减少总时间的简单条件是

`H + Δ_shared < (N−1) × (d + c + b − max(d,c,b))`。

这只是模型筛选阈值；若 d 和 b 共用链路，或通信和计算共用 NPU/片上带宽，不能各自当成独占满速，须把碰撞所造成的延迟计入 `Δ_shared` 或改用实际共享资源排程。微批大小不同、专家倾斜和不同 rank 完成时间不一致时，以每块的独立阶段时间和真实依赖作事件模拟，不能继续用同一个 d/c/b。真实基线若是**一个未拆分的融合大批次**，比较应使用其同负载实际时间 `T_original`，判断 `T_pipe_ideal+H+Δ_shared < T_original`；不能把 `N(d+c+b)` 冒充原大批次时间，因为切块可能改变 GEMM 效率、权重读取、通信轮次及 launch 次数。若左侧理想值本身已不小于 `T_original`，在本模型的非负 H/Δ 假设下即可先不做该方案的性能实验。

**额外显存门槛。** 比较相同批量下的峰值活跃缓冲：

`ΔM = max_t(候选同时存活且不可别名的 buffer 总字节) − max_t(基线对应 buffer 总字节)`，

再加入新增事件/metadata、allocator 对齐/碎片及图捕获的稳定地址开销；需能放进对应 rank 的实际可用预算并保留用户设置的余量。候选通常可能同时保有一个 microbatch 的 dispatch 接收/量化数据、另一个的 GMM 中间值或 combine 输出，以及各自请求/slot mapping；权重若安全共享不应因 microbatch 数自动复制，不能把所有 buffer 都乘 N，也不能忽略禁止覆盖的生命周期。上述是容量可行性筛选，不由 tensor 形状直接推定实际增量；`ΔM` 的峰值须按实际时间与 alias 计算。总耗时改善也不保证单请求 TTFT 或尾延迟改善；两项都要按实际业务目标分别比较。

**纯人工示例，单位为微秒，并非设备数值。** 假设 N=4，三个阶段分别 `d=100,c=120,b=80`，同 N 串行时间 1200，理想流水 `100+120+80+3×120=660`；可用于覆盖额外开销和争用的预算为 `1200−660=540`。

- 假设 `H=80、Δ_shared=130`，模型候选 870，比同 N 串行参考短 330，约 27.5%；这是在假设排程与内存条件都成立时**值得实验的方向**，不是预测收益。
- 若相同 `H=80` 而 `Δ_shared=500`，候选为 1240，比串行长 40；通信与计算虽发生重叠，净效果仍变差。

```sh
python3 - <<'PY'
n,d,c,b=4,100,120,80
serial=n*(d+c+b)
ideal=d+c+b+(n-1)*max(d,c,b)
for h,shared in [(80,130),(80,500)]:
    candidate=ideal+h+shared
    print('serial',serial,'ideal',ideal,'overhead+contention',h+shared,
          'candidate',candidate,'time_saved',serial-candidate)
PY
```

目前 169 已保存的 L14/L39 是融合 MC2 的整体 task，尚无同次 d/c/b 内部阶段、实际 microbatch 数、shared contention 或额外活跃 buffer 的可绑定数据，不能将这两条时长代入本模型判定收益。先用现有源码明确后端是否可拆分/异步发布、按同一负载定位可并行 microbatch，再在被授权且资源空闲时，以原大批次和等量微批串行为控制、单次受控采样取得实际阶段/额外内存；若接口或事件不支持，停止这一候选而非靠假设填数。本节不新增设备工作。
