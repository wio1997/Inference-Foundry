# F23：shared/routed 合流到 HC post 的逐 tile 依赖

## 本次增量和已有覆盖

F23 先前的 `HOTSPOT-FUSION-REVIEW.md`、`WRAPPER-REVIEW.md` 已记录 tuple/tensor 分支、既有 `muls_add_triton`、MC2/HC 外部调用；`agents/scheduler/F23-COMMON-DEPENDENCY.md` 仅对受限值域证明双方输入可能影响合并值。本次只补该合并结果向 HC post 的**每 token/feature tile 所需数据与浮点顺序**，不重复声称全部 F23 已扫完，不将这些条件结论升级为 169 普遍必要下界。

## 实际源码依赖与已有融合

- `model.py:269–326`：独立 shared 是否存在取决于 `mix_placement` 与模型配置；`self.experts(...)` 可以返回 tuple 或 tensor。tuple 非 FP16 且有 shared 时，已有 `muls_add_triton(routed,shared,routed_scaling_factor)`；tuple FP16 且有 shared 时是 `muls_add_triton(shared,routed,1/routed_scaling_factor)`，两个分支不应按同一公式重写。无 shared、tensor 返回又有不同路径。frozen `mul_add.py:8–55` 中这个 Triton kernel 本身把 `x*scale+y` 组合并写 `empty_like(x)`；表达式如何被编译、特殊值、是否 FMA 仍未知。已有复合 `DispatchFFNCombine` 不能再次拆成新待实现的融合。
- 在返回 tuple 的条件下，上述合并结果**之后**源码还有条件 `maybe_all_reduce_tensor_model_parallel`；其他 `is_sequence_parallel and not already_sequence_parallel` 的分支使用 `sp_all_gather`。正常 decoder 在 `model.py:910–919` 传入 `already_sequence_parallel=use_sequence_parallel`；这使第一个 if 不成立时仍可能检查后续 `elif tp_size>1 and tuple`。这里没有 runner 内部 `maybe_all_reduce` 实现/状态，无法把它武断判定为实际通信，也不能仅因 SP=true 就断言该调用必定跳过。tensor 返回的源码注释称 runner 已归约，仍需看 runner 和真实对象验证。
- 同一 decoder `hc_post(x,residual,ffn_post,ffn_comb)` 将 MoE 返回值、attention post 的 residual、上次 FFN hc_pre 输出的两个系数交给 `npu_hc_post`。`ffn_pre` 是独立结果，作为下层 pre_mix 返回，不能融合后丢弃。`frozen-source/csrc/moe/hc_post/op_kernel/hc_post.cpp:50–55` 的 A3 路径选择 `hc_post_d_split.h`；该源码按 token、feature 段处理，读取对应 feature 上 `x`、全部 HC 路 residual、当前 token 的 post 和 comb，以 FP32 逐路乘加后对两字节输出 `CAST_RINT`。没有跨 feature/token 数值归约；源码存在设备 tile，不证明对外逐 tile ready 或已绑定历史二进制。

## 可交叠边界与必须保持的顺序

对于给定 `(t, d块)`，可研究只在 **该块的最终 MoE x（包含选中分支的 routed/shared、缩放及需要的跨 rank 归约）**、对应 `residual[t,:,d块]`、`ffn_post[t,:]` 与 `ffn_comb[t,:,:]` 都已就绪后输出这块 HC residual。其他 token/feature 的未完成数据在数学上不必作为该块的输入；但当前 `muls_add_triton`、collective 与 HC 接口不保证块级事件、可见性、固定地址或不会覆盖，不能据数学独立性删除整 tensor 等待。各 rank 分块次序、空/填充 token、重复输出、graph replay、alias/消费者和 stream 的 happens-before 需验证。通信与 Cube/Vector/MTE 同样可能竞争。

一个新观察是：在最终 MoE x 等待期间，**可以条件地研究提前读取 residual 与系数、计算各独立乘积**，但不能默认先把它们相加成一个 residual 汇总后再加 x。`hc_post_d_split.h:305–349` 先得到 `x*post`，再按 `hcIndex` 的源码顺序逐项 `DoAdd`。浮点加法不结合；下例只说明无条件重排不成立，不证明真实模型输入可达：对某一输出、post 和两个 comb 系数都取 1，BF16 x=`2^80`，BF16 residual 两路为 `-2^80` 和 `1`，其余路径贡献为 0。按该源码的 FP32 顺序，`round(round(2^80 + (-2^80)) + 1)=1`；先求 residual 汇总再加 x，则 `round(2^80 + round(-2^80 + 1))=0`。所有所列输入都可在 BF16/FP32 精确表示，且最终 0 与 1 的 BF16 不相同。真实 comb 的模型可达范围、设备实际编译与输入数据仍未知；该例只否定未经条件证明的任意重排。

如果将 tuple 的合并直接接入 HC kernel，至少要模拟原 `muls_add_triton` 写到 `empty_like(x)` 的 dtype 舍入，之后 HC 把该值转换到 FP32 并保持其具体乘加顺序与最终 BF16 舍入；若中间还有 collective，还须先按原布局、通信顺序及规约数值生成最终 x。不能把 `scale*routed+shared` 的高精度临时结果直接代入 HC，当作和原先 BF16 store 后读取等价；也不能把不同 rank 的未完成 partial sum 当成最终 x。输出给后续层和 DSpark 等可见消费者必须保持全部 HC 路的最终值。

## 单个已保存样本和未知

`f22-layer24-run2-evidence.json`（stage157、EP8、model49、stream89）里 `DispatchFFNCombine` task1298 为 955.061–1255.835 µs，之后一串与 shared MLP 形状相容的量化投影/`DequantSwigluQuant` task1299–1302 到 1312.954 µs；`HcPost` task1304 为 1336.534–1344.373 µs，其输入记录 `[1,3,5120] BF16;[1,3,4,5120] BF16;[1,3,4] FP32;[1,3,4,4] FP32`。这说明相关 task 类型和 shape 同窗出现、指定流中按此时间排序；**不能证明** 1302 是 shared 输出、1298 是 routed 输出、两者相加或最终进入 1304，亦不能把空档认作同步、设备空闲或可节省时间。查询仅挑选部分类型，没有充分的跨流 task、event identity、runner 返回值和 tensor storage。

下一步优先只读现有材料中 factory/runner 的 tuple/tensor 实际实例及 ready event；若无同次精确绑定则保持 UNKNOWN。此记录未修改实现、未运行 NPU，也不量化候选收益。

## MC2 内部最终合流与 HC 的额外块级边界

冻结的 `sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp:958–978` 中，AIV 先按 `isCombineV1` 进入 `CombineV1` 或 `CombineV2`；之后依次执行本地 `SyncAll<true>()`、`ResetTokenPerExpert`、`shmem.CrossRankSync()`，再调用 `KernelMoeTokenUnpermute`。后者传入 peer 输出内存 `offsetD`、`expandedRowIdx`、`probs`，写 `ptrOutput`。在已保存源码中，`CombineV1` 按专家组等待信号，并向 epilogue 传入各 peer 的目标内存及对应行；`CombineV2` 在专家组推进时等待 cross-core 标志，再按 scheduler 的 tile 执行 epilogue（981–1092）；不能将其理解为每个 tile 都单独等待一次生产者标志，详见 F21 的同步范围审计。分支判断是 `problemShape.m()*topK <= 4096` 时选择 V2（253–256），但历史调用的参数、选中分支和编译对象未绑定。

这提供了此前未细化的**重组候选**：研究用细于整个跨 rank 同步的就绪协议，在某 token/feature tile 所需的所有远端 expert 贡献、索引及概率都已可见且原有 unpermute 数值语义可以保持时，提早形成 routed 输出，再与 F23 所述 shared/归约/HC post 边界衔接。该候选须视实际 unpermute 接口和路由为准，不能因一个 expert 或一个发送 rank 完成便认为整个 token 可消费；其余 tile 的计算及远端返回理论上可能继续。原代码的专家组和 tile 信号只表明源码已有的 combine 阶段内部同步，不构成对外逐 token 输出 ready 接口。

最初本地未保存 `moe_token_unpermute.h`、`hccl_shmem.hpp` 和相关 epilogue 实现；现已找到前两个文件的部分摘录（见下节），但仍不能完整断言 `CrossRankSync` 的内存序、实际通信完成语义、unpermute 的具体累加与舍入次序，或现有 barrier 是纯等待；也不能直接删除它。若继续研究，必须先补齐同版本接口：验证所有相关发送者写完成及接收端可见、topK/屏蔽/capacity 与空行、排列映射及概率加权、各 scale 与中间 dtype/舍入、跨 rank 通信顺序、`ResetTokenPerExpert` 和 workspace 生命周期；后续 shared 合流和可能的 collective 也必须完成，才能按 F23 的条件交给 HC。现有 profile 未拆出这段内部耗时，无法量化收益或将其认定为草稿 MC2 低活动根因；此处无新设备操作、没有改变实现。


## stage186：同步协议与 unpermute 的部分源码摘录

本机 ZCode `deepseek-flash` 执行 stage186，退出码为 0 且模型验证通过；查询在指定 stage 容器内的已知目录查到两个文件。摘录与哈希保存在 `f23-missing-interfaces.json`（SHA256 `46d61268f3a4dd81808d031c7b5b0c8ff27c7d14bf855fbd6ee9a7761d6c5984`）。这只证明容器内被查文件的存在和内容哈希，并未证明它们曾编入历史采样时实际运行的二进制。`moe_token_unpermute.h` SHA256 `08982836ca51157eb674d447c66730cfeffdf8080d7172a157d887c7e58c3a0d`，375 行，筛选行**超过 200 行而截断**；`hccl_shmem.hpp` SHA256 `596bc8a0c28748aade2e5f2e978f18a6e593d146a8cf07fe4aeac8d36c19bad6`，327 行，筛选片段没有因行数上限截断，但仍是按关键词及上下文摘录、不是全文。

在所保存的 `hccl_shmem.hpp:166–183` 中，`CrossRankSync()` 从本地 sync_base 读取值加一作为 count；每个 vector ID 遍历分配给它的 rank，对于每个 rank 先向该 rank 的以本 rank 为索引的远端同步槽 `gm_store(count)` 并 `gm_dcci`，再调用 `gm_signal_wait_until_eq_for_barrier` 等待本地对应 rank 的槽等于 count 或 count+1（stage188 的完整源码纠正了此前仅写 count 的表述）；最后执行 `SyncAll<true>()` 并写回本地 sync_base。这个源码序列比“可能存在跨 rank 同步”更具体：**是逐 peer 发信号、等信号，再本地汇合的协议**，其中循环包括本 rank。不能据摘录断言 `gm_dcci`、`gm_store` 和先前远端数据写之间的完整硬件内存序、事件进度、具体同步参与者或等待时长，也不能仅因该协议存在就认定历史低活动由它造成。

`moe_token_unpermute.h` 片段显示 `CalPartOutToken` 对首个索引满足 `cal_token_idx < num_out_tokens` 的输入调用 `CopyTokenIn` 和 `CalFirstToken`，否则初始化 float 向量为零（250–263）；随后片段可见对其他满足该检查的索引读取概率、复制对应 token 并调用 `CalToken`（267–277）。在 `PROBS` 分支，`CalFirstToken` 将非 float 的输入 cast 到 float，再按概率 `Muls`（303–320）；`CalToken` 片段显示非 float 输入 cast 至另一个 float 向量，再按概率 `Muls`（323–335）。**截断遗漏了循环头、`CalToken` 后续计算（包括可能的加法）及 `CopyOut` 等代码**，所以仅凭当前摘录不能确认完整遍历顺序、加法与融合指令、最终写回 dtype/舍入、无效索引的全部边界，或者同一 tile 的精确数值公式。随后已取得相同哈希的完整源码并核查这些分支，见 stage187。

新证据没有改变 F23 的关键约束：重排需要依赖对应 expert 输出已可见并遵循同步与数值协议，仍不能直接跳过全 rank 同步。当前不知历史任务是否采用这些文件、不知该阶段实际耗时，也没有收益估计；未进行任何新 NPU 操作。


## stage188：信号轮询和远端写回的源码边界

本机 ZCode 只读取指定 stage 容器中三个固定文件，在任务目录保存源码快照。`hccl_shmem.hpp` SHA256 `596bc8a0c28748aade2e5f2e978f18a6e593d146a8cf07fe4aeac8d36c19bad6`，与 stage186 相同；`block_epilogue_pertoken_row.hpp` SHA256 `806f6c84ec1c0c0e8653ac64e589bca18b509b251d7b1a54d9d926513f3e215d`；`block_epilogue_pertoken_v2.hpp` SHA256 `212617897e7a50c960be900989d8ca3222c4637eee0e1a2cca0dd6f0b759ef90`。快照在 `agents/fusion/frozen-source/csrc/mc2/dispatch_ffn_combine/op_kernel/utils/`；源码存在与哈希一致不等于历史 profile 的运行二进制绑定。

`hccl_shmem.hpp:31–63` 明确：`gm_store` 是普通 `__gm__` 指针赋值；`gm_dcci` 包装指定地址后调用 `DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>`，前后各有空的 volatile asm；轮询每轮先对本地信号地址调用该操作，再读信号值，**等于 count 或 count+1 均返回**。因此 stage186 对只等于 count 的叙述已纠正。代码本身不能解释 next-generation 信号设计动机或证明完整内存可见性。

对于所保存的量化 epilogue：`block_epilogue_pertoken_row.hpp:159–187` 将矩阵结果取入 UB、转 float、乘 per-token scale、以 `CAST_RINT` 转换后，经 `copyUbToGmD` 写目标；写回调用后设置 `MTE3_V` 标记，`Finalize()` 对各 UB stage 等待相应 `MTE3_V`（117–122）。V2 实现 `block_epilogue_pertoken_v2.hpp:143–228` 将块按目的 rank 和路由行范围切分，构造 `shmem(params.offsetD, dstEpIdx)` 地址，对相交块调用 `copyUbToGmD`，写回调用后设置 `MTE3_V`；`Finalize()` 等待两个 MTE3_V 标记（105–114）。这是源码次序和事件标记；具体实例选择和历史运行分支仍需验证。

外围 `dispatch_ffn_combine_kernel.hpp:958–978, 981–1092` 的源码次序为对应 Combine 内 `Finalize()` → 同 rank 的 `SyncAll<true>()` → `ResetTokenPerExpert` → `CrossRankSync()` → 从本 rank 的 `shmem()+offsetD` 读取做 unpermute。远端目的地址与后续本地接收缓冲地址的对应体现了协议设计结构；**尚不能证明** MTE3 标记对跨卡写到达的确切完成语义、`SyncAll` 的跨管线内存顺序、对信号缓存行的 DCCI 是否为之前的数据写提供 release/acquire 或 peer 可见性，以及本地轮询和后续 unpermute 读取的一致性。信号 DCCI 针对信号槽，不能据此说已经刷新 payload 全部缓存行。需取与当前 CANN、目标 A3 和编译模式匹配的 `DataCacheCleanAndInvalid`、MTE3 事件与跨设备共享内存协议的正式说明或同版本接口实现，再绑定实际源码和二进制；不能因为代码里有同步就认定耗时或安全删掉屏障。本轮未启用设备，也未修改目标实现。

## stage187：完整 unpermute 的确定顺序与数值边界

ZCode stage187 将完整文件保存为 `frozen-source/csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute/moe_token_unpermute.h`，大小 16454 字节，SHA256 与 stage186 一致（`08982836ca51157eb674d447c66730cfeffdf8080d7172a157d887c7e58c3a0d`）。这补齐了 stage186 被截断的源码，但仍不是历史运行二进制绑定。

在该版本中，`Process` 分批处理输出 token，每 token 再沿 hidden 维切块（174–187、215–235）；每个块的 `CalPartOutToken` 依次访问该 token 的 top-k 索引：首先处理 `start_token`，如其满足代码的 `idx < num_out_tokens` 检查，则使用第一个贡献，否则初始化 float 向量为 0；之后从 `start_token+1` 到 `start_token+top_k-1` 按递增索引依次处理通过同一检查的贡献，最后写回该块（241–281）。源码只显示这一上界检查，索引的合法下界以及 sentinel 语义还要结合生成索引的实现，不应自行假定。

调用点使用 `KernelMoeTokenUnpermute<ElementD2, int32_t, float, true>`（MC2 kernel:975），所以在此源码实例中概率类型为 float 且 `PROBS=true`。当 `ElementD2` 非 float，首个有效贡献读入后 `CAST_NONE` 到 float，`Muls` 概率；其他每项分别转换为 float、乘概率，再通过 `Add` 按上面的遍历顺序加到 float 累加值；最后 `CopyOut` 以 `CAST_RINT` 转回 `ElementD2`（303–357）。当 `ElementD2` 为 float，首项复制到 float 累加缓冲并乘概率，后续项直接 float `Muls` 后 `Add`，输出直接使用 float，不执行该末尾 cast。这里描述的是源码算子顺序，精确机器舍入及编译器生成指令仍需具体二进制和硬件语义确认；`ElementD2` 的实际类型也未绑定历史调用。

这将候选的正确性要求进一步收紧：当一个 token 尚有其他通过检查的贡献未就绪时，不能仅因某个 expert 或 rank 已返回就输出整个 token，也不能任意改变 top-k 累加顺序、先后乘概率、越过原有写回 `ElementD2` 的舍入再衔接 shared/HC；允许按 feature 分块独立推进不代表现有设备接口公开 ready 或保证重叠。还需结合实际 peer 写入和 index 生成证明每项就绪，以及确认后续是否有 collective、tuple 分支与数值约束。没有内部时间或实际二进制绑定，仍不能推断改造收益。

## 厂商缓存接口文档的适用边界

华为 [CANN Community Edition 9.0 DataCacheCleanAndInvalid 官方文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0177.html) 将配置 dcciDst 的接口列为 Atlas A3 支持，解释 `SINGLE_CACHE_LINE` 对传入地址所在行进行刷新，`CACHELINE_OUT` 管理该核 data cache 与 GM 的一致性；非 64 字节对齐时处理范围有文档所列限制。这与 stage188 看到的信号槽单行 cache 操作相容，但这里只核对到**9.0 文档**，历史 profile 自述 CANN 9.2.0；没有核实所安装运行库、接口文档版本、编译目标与该文档一致。该接口说明未直接给出 MTE3 到其他 rank 的 payload 完成、跨设备信号 release/acquire 或完整同步协议的承诺，所以不能从这一缓存操作推导 payload 可见或可删 barrier。

## 核内事件与跨卡可见性：公开文档所能证明的边界

[华为 CANN Community 9.1.X SetFlag/WaitFlag（ISASI）文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/ascendcopapi/docs/api/SIMD-API/%E5%9F%BA%E7%A1%80API/%E5%90%8C%E6%AD%A5%E6%8E%A7%E5%88%B6/%E6%A0%B8%E5%86%85%E5%90%8C%E6%AD%A5/SetFlag-WaitFlag.md) 将该接口列为 Atlas A3 支持，描述 SetFlag 在**源流水前序读写完成**后置标记，WaitFlag 在目标流水等待相应标记；同核同步表列有 AIV 的 `MTE3_V` 组合。因此，对 stage188 中所见的 `MTE3_V` 配对，若实际编译确按此文档语义实现且 `copyUbToGmD` 确实向该核 MTE3 流水发出写操作，可将对应标记视作**本核源流水前序读写完成后，目标 V 流水可继续**的条件性证据。必须区分源码控制流返回、标量 S 流水和目标 V 流水：不能仅据 `WaitFlag<MTE3_V>` 说所有后续标量指令均被阻塞，也不能单凭该同核事件定义推出对端卡对 payload 的可见性或 payload 与信号的跨卡 release/acquire 次序。

[华为 CANN 商用版 8.5 基础数据搬运文档](https://www.hiascend.com/doc_center/source/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0103.html) 对 A2/A3 的跨卡通信算子场景写明 `DataCopy` 类接口支持跨卡搬运，且限 HCCS 物理链路。这提供了跨卡本地搬运的一种厂商支持路径，并不是该历史 MC2 一定使用该路径的证据：`copyUbToGmD` 的模板实际实例和实现、历史设备链路、编译库与二进制尚未绑定；该版本接口也没有为此程序证明完整 payload→signal→peer-read 的跨设备内存次序。[华为 CANN 9.1.X HCCL 通信模型](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/commopdev/docs/en/comm_op_dev_guide/prog_models_concepts/comm_model.md) 说明注册通信内存与远端地址映射这一一般模型，但不能据此认定 stage188 的 `shmem` 实现已满足具体注册、映射和一致性条件。

以上厂商页面分别为 **9.1.X / 商用 8.5**，不是已验证与历史 profile 中报告的 CANN 9.2.0 的接口、编译选项及二进制完全一致。尚需核实对应实例的 `copyUbToGmD` 实现、MTE3 对远端写完成的架构定义、`SyncAll<true>` 对参与核/流水的真实语义，以及信号写入和数据读取之间的跨卡内存序契约；无法因这三个概念性文档便认定现有全局同步可删除、分块重组正确或可量化收益。本次仅查阅公开资料与既存源码，无设备操作。

## 返回 peer 的偏移还依赖更早 expert 计数

进一步核对已存 `dispatch_ffn_combine_kernel.hpp:655–731,985–1021` 与 `block_epilogue_pertoken_v2.hpp:185–221`：每个 `dstEpIdx` 的计数向量进入临时数组 `tmpBuffer` 后，源码按索引递增扫描，构造 `preSumBeforeRank[dstEpIdx, groupIdx]`；其值是 `tmpBuffer` 从索引 0 到 `params.rank * expertPerRank + groupIdx` **之前**所有值之和。CombineV1 和 V2 均使用这一值定位写到对应 peer 的目的行，V2 还加上当前 tile 相对于该 expert 段的行偏移。

这比仅说“每个专家数据完成后即可按专家分块发送”更强：若要沿用现有地址布局，某个 group 的目的行偏移可能依赖排序在前的**其他 group 和 rank 槽位的计数**。例如假设 `expertPerRank=2`、`params.rank=1`，该 peer 的计数数组为 `[3,2,4,1]`，则 group 0 的偏移是 `3+2=5`；即使 group 0 自身对应的计数 4 保持不变，若前一 rank 的 group 1 计数由 2 变 3，偏移就变为 6。上述示例仅演示源码下标算术，不代表历史实际计数。提前发一个 group 时，必须先得到并确保对应前缀计数有效，或设计并验证与消费者映射一致的新布局；仅有本 group 的 GMM tile ready 与 scale ready 不够。需要另行保证计数写入可见、空专家、截断、跨 rank 行顺序、buffer 生命周期和 unpermute 的索引/浮点语义。现有源码与历史二进制及计数未绑定，不能推断实际等待或收益。

此处还有一个关于规模的**源码适用条件，未确认为错误**。同一文件 `:660–689` 的 AIV 循环以 `dstEpIdx=coreIdx; dstEpIdx<EP; dstEpIdx+=coreNum` 分配 peer，但所存 `shmem(localTokenPerExpertOffset, coreIdx)` 使用的是 `coreIdx`；`utils/hccl_shmem.hpp:129–141` 对 `rankId` 索引 HCCL 远端窗口或调用 `shmem_ptr`。当 `EP<=coreNum` 时，一次迭代内二者相等；如果假设 `EP=5, coreNum=4, params.rank=1`，coreIdx 0 的 peer 循环分别访问 dstEpIdx 0 和 4，传给地址接口的 rankId 都是 0。该局部反例意味着不能不加条件地把此源码当作任意 EP/coreNum 配比下的正确逐 peer 计数发送协议；也不能由此断言系统存在历史 bug，因为 host 是否允许该配比、其他限制及具体编译运行条件未闭合。host `dispatch_ffn_combine_tiling.cpp:246–252` 从平台取得 AIC/AIV 数并计算 blockDim，但本材料没有将该值绑定到这次 kernel 的实际 `coreNum` 和 EP。扩展或拆分时须先核实这些约束，再逐 peer 验证地址和参与核。

核验 kernel 文件 SHA-256 为 `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`，为静态源码观察；没有进行 NPU 工作或修改目标实现。

### peer 索引条件与已存采样的适用范围

本轮沿保存的 host → tiling → kernel 源码追踪：`op_host/dispatch_ffn_combine_def.cpp:76–81` 的第六个属性声明为 `worldSize`；`op_host/dispatch_ffn_combine_tiling.cpp:39,112–117` 的这一属性数值检查允许 `1..768` 并存入 `info.worldSize`；`op_kernel/dispatch_ffn_combine.h:143,272` 将该 tiling 字段作为 `EP` 传给 kernel。这个**属性数值检查本身没有按 AIV coreNum 收紧**，不证明其他校验、通信后端或硬件支持所有这些值。`kernel.hpp:195–201` 区分 AIC 和 AIV 的 coreNum，AIV 使用 `get_block_num() * get_subblockdim()`；host tiler 通过平台查询及 `CalcTschBlockDim` 设 blockDim（`:246–252`），无法仅凭这些调用知道历史实际返回值。

设备入口 `dispatch_ffn_combine.cpp:25–31` 的 key `1000010` 声明 `KERNEL_TYPE_MIX_AIC_1_2`。华为 [CANN 9.1 GetSubBlockNum 接口文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetSubBlockNum_ISASI.md) 对 A2/A3 自定义算子该 kernel 类型列出 AIV 的 `AscendC::GetSubBlockNum()` 返回 2。另一个 [CANN 9.2.0-beta.2 编译器说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/programug/Ascendcopdevg/asc-devkit/docs/zh/guide/programming_guide/compilation_and_execution/operator_compilation/bisheng_compiler.md) 以 MIX 示例说明底层 `block_idx` 与 `GetBlockIdx()` 的逻辑索引可能不同。**以上是对应版本接口的一般规则**：尚无证据表明这个源码中小写 `get_block_num()`、`get_subblockdim()` 与文档 API 在历史编译配置下分别等值，也未证明 profiler 的 Block Num / Mix Block Num 就是这两个返回值，不能直接把 24×2 写成观测到的 kernel `coreNum=48`。同样，平台 ini 的物理核数不能替代该实例的运行参数；版本和编译绑定均未闭合。

对于 stage160 的 48 个采样，导出 Block Num=24、Mix Block Num=48、MIX_AIC 是与 1:2 **相容的字段**，不是实例内部函数返回值；历史 kernel 名也不是编译对象身份。stage169 的 EP0/EP8 两份 profiler `mc2` 组分别列出 16 个成员，且同两条第 0 层样本的 `attr_0` 哈希与组键匹配；二者未证真实 communicator 调用绑定。stage167 两条 `attr_5` 的 UTF-8 值哈希都等于 `SHA256("16") = b17ef6d19c7a5b1ee83b907c595526dcb1eb06db8227d650d5dda0a9f4ce8cd9`、长度均为 2，与当前 schema 的第六属性和该组 16 成员相容；没有独立证明 profiler 的 attr 序号与该 schema 在历史版本一一对应，故这也不是实际 `EP=16` 的正式确认。

因此，**如果另行证明**该历史调用确实使用上述 kernel、实际 EP=16、实际 AIV `coreNum>=16`，则上一节 `EP>coreNum` 的 peer 反例不适用于该调用；当前尚不能解除任一条件。现有安装对象有相同 key 只是候选线索，不能证明该对象由此源码编译或被该样本加载；也不能由这些信息推断实际参与核、错误或性能。核验 SHA-256：host tiler `783f76d7869371df949fc5ee80e33dfca8cf81231a65da410fcfea40df89b6f8`，入口 `0a91cedd604c58ce530388ea1d8e8bb7c01ab1191a2ddc6ce2dc064c940f989c`，host def `0a1acc96532a03d9dac2b4e0ab47fcf37e967719909d93f380f876eedc545951`。无 NPU 工作。

## shared/routed 合并跨 TP 归约的条件反例

此节补充该既有 F23 多消费者边界中，合并与条件 TP 归约之间的数值/归约域依赖；不把它登记为独立融合收益。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py` SHA-256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`，`:300–326` 的 tuple 返回、`hidden_states.dtype != torch.float16` 且 shared 存在时，在每 rank 调用 `muls_add_triton(routed, shared, routed_scaling_factor)`；后续在第一个 sequence-parallel 分支不满足且 TP size > 1 等条件下调用 `maybe_all_reduce_tensor_model_parallel`，返回结果由 decoder `:913–919` 的 HC post 消费。冻结 `agents/fusion/frozen-source/vllm_ascend/ops/triton/mul_add.py:23–34` SHA-256 `cd1845442d11fecd7f9de1ee1020e457a40483348bb511150039e51bb7791c20` 表达 `x*scale+y` 并输出到 `empty_like(x)`；编译后的乘加指令与计算精度未绑定。`evidence/model-config.json` SHA-256 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7` 的 text config 为 `routed_scaling_factor=1.5`。

条件化地研究把合并移到通信后，必须同时保留 routed 和 shared 的完整 rank 归约域。例如，假设相同 token 的两个 rank 分别有 shared 值 `3,3`、routed 值 `1,2`，且该方法确为两 rank SUM，忽略舍入则原结果 `(3+1.5×1)+(3+1.5×2)=10.5`；先 SUM routed 后只加一次 shared 得 `1.5×(1+2)+3=7.5`。这个例子仅否定不经条件证明就只加一次 shared；对 shared 也保持正确的跨 rank 求和、或另证具体分片/缩放和消费者契约时，还可以研究其他重排。它不声明历史 shared 确实跨 rank 复制、两 rank 拥有同一 token、原方法执行了 SUM 或通信存在。

即使 shared 均为 0，移动 routed 的比例因子跨归约也不自动保持 BF16 舍入。以下 CPU 模型假设两 rank 每次合并后写 BF16，归约相加后再次以 BF16 最近偶数舍入：`r0=-3.8125`、`r1=32`、`α=1.5`。先各自乘比例再归约得到 `42.25`；先归约再乘比例得到 `42.5`。可复算：

```python
import struct

def bf16(x):
    u = struct.unpack('I', struct.pack('f', x))[0]
    u = ((u + 0x7fff + ((u >> 16) & 1)) >> 16) << 16
    return struct.unpack('f', struct.pack('I', u))[0]

r0, r1, alpha = -3.8125, 32.0, 1.5
before = bf16(bf16(alpha*r0) + bf16(alpha*r1))
after = bf16(alpha * bf16(r0+r1))
assert (before, after) == (42.25, 42.5)
print(before, after)
```

该例限定有限值、所示 BF16 舍入和两 rank 算术，不模拟目标 Triton/HCCL 的 accumulation、舍入、可见性或其它树形顺序，亦不证明真实输入会产生该差异。FP16 shared 分支原代码使用不同缩放公式，tensor 返回分支及 SP all-gather 也不可混为一谈。原合并/归约完成与 HC post 之间的输入 shape、dtype、全 rank 贡献、tile-ready 事件、buffer 及其它消费者仍须分别证明。既有 EP0/L0 中两个 BF16 `[595,5120]` 输入、BF16 `[595,5120]` 输出的 `ForeachAddListV2` 记录见 F22，但名称/shape 不能证明其为此 `muls_add`、同一份 shared/routed 或历史实际触发 TP SUM。此处无新设备采集，也不修改推理代码。
