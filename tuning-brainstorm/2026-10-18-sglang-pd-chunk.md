# 2026-10-18｜PD 分离下的 chunk 优化：价值筛选与验证入口

这次审阅的是 **Prefill、Decode 使用独立实例和计算资源** 的场景。目标是判断此前九个方向中，哪些有具体的收益来源，哪些只是已有配置取舍，哪些应当排除。这里的“有价值”指值得用证据继续验证，**不等于已加速，也不等于新发明**。

优先保留两类问题：一类是发送准备中确实存在的多余数据构造；另一类是 KV 已具备发送或交接条件，却还在等待调度。前者更接近“少做等价工作”，后者需要先画出真实依赖和等待，不能直接删除同步。

## 结论：两个配置对照，六份候选记录

| 原方向 | 本次判定 | 具体理由与去向 |
| --- | --- | --- |
| 1. P 端选择高效计算 chunk | 保留为已有配置对照 | 改变算子形状、轮数、排队和传输时机；不存在普遍最优值。见下方配置 A。 |
| 2. 长短输入公平性 | 保留为已有策略对照 | 短请求提前往往让长请求推迟；上游已有为短请求预留预算的逻辑。见配置 B。 |
| 3. P 计算与 KV 发送重叠 | 保留，但改为检查“重叠为何没发生” | 中间 chunk 发送已经存在。与第 9 项合并为 [PD-CHUNK-002](#pd-chunk-002) 的时序审计。 |
| 4. 计算块与传输段联合选择 | 有条件保留 | 发送次数、首发时刻、带宽利用和计算效率互相制约。见 [PD-CHUNK-003](#pd-chunk-003)。 |
| 5. 减少每次发送的 CPU 工作 | 优先保留具体候选 | Ascend 路径构造每段全部页号，但消费者只读取首项和长度；有可指认的中间表示开销。见 [PD-CHUNK-001](#pd-chunk-001)。 |
| 6. 减少续块管理重复工作 | 条件保留，复用已有缓存记录 | 要查明哪一段切片或复制确实重复，不能整体跳过 checkpoint。见 [PD-CHUNK-004](#pd-chunk-004)。 |
| 7. D 端预留 KV 的时机 | 保留为已有机制及适配研究 | early / prefill_complete 已有；first_chunk 有关闭未合并的方案。Ascend 与并行限制是前提。见 [PD-CHUNK-005](#pd-chunk-005)。 |
| 8. 根据 D / 网络积压控制 P 准入 | 条件保留，优先级较低 | 已有容量准入和握手约束；必须先证明它们仍允许有害积压。见 [PD-CHUNK-006](#pd-chunk-006)。 |
| 9. 最后一块的交接尾延迟 | 保留为时序审计 | 发送前等待和传完后的完成检测是不同段，分别测量、分别改动；并入 [PD-CHUNK-002](#pd-chunk-002)。 |

**实施顺序：**先取得 002 的时间线；如果发送准备 CPU 暴露在关键路径，做 001；如果是块间组批空隙，查 004；如果是传输粒度问题，再做 003。005 / 006 只有在容量或积压成为瓶颈且目标配置支持时进入实验。

## 范围与证据等级

- 固定上游源码：SGLang [57db5089c207ae4ce189cc092a89a2027c8aea29](https://github.com/sgl-project/sglang/commit/57db5089c207ae4ce189cc092a89a2027c8aea29)。
- 审阅日期：2026-10-18。PR 状态见文末；打开的 PR 不能当作该源码已具备的能力。
- 目标后端关注 Ascend，但**没有在本次任务中建立用户实际 Ascend PD 基线**。模型、量化布局、P/D 数量、TP/EP/CP/PP、传输后端、缓存设置及运行 commit 均须实机确认。
- 历史单引擎教学数据不作为 PD 的性能证据；源码中的通用路径也不自动代表目标模型会走该路径。
- 本次只保存逻辑分析和验证计划，没有修改推理实现、运行模型或报告实测加速。
- 阅读入口：[官方 PD 文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)。具体实现判断以本文固定源码链接为准。

## 先明确实际链路

P 接纳请求 → 分块计算 → 安全就绪的 KV 分段发送 → 最终结果及必要附加状态完成 → 传输完成与提交 → D 请求进入可调度状态 → D 继续生成。

这些阶段可以部分重叠，不能把所有耗时相加作为请求延迟。

- **已经有分块发送。** overlap 路径在结果处理后发送中间块；非 overlap 路径在处理 chunk 时发送。提前发送已缓存前缀也有独立入口。[中间块发送源码][p-middle]、[缓存前缀提前发送][p-cached]。
- **传走不等于 P 可以释放。** 对普通 FULL attention，后续 P chunk 仍读取前缀 KV；异步传输也可能仍在读源地址。正常路径在传输成功后才解除请求的缓存引用，缓存页本身还可能继续驻留。[完成与释放][p-success]。
- **D 收到第一段不代表可以正常继续生成。** 所需 KV 和附加状态须完整、可见，并通过提交步骤。[D 的完成检测与提交][d-success]。
- **首 token 生成、D 开始继续生成、用户收到首 token 是三个事件。** P 最后一次 prefill 处理会记录采样 token；正常 P 输出又位于传输完成后，D 接收该 handoff token。用户 TTFT 还取决于路由与流式输出路径。[P 首 token][p-first]、[P 输出][p-success]、[D 提交 token][d-token]。
- **PD 与 PP 不同。** 固定源码的动态 chunk 初始化要求 PP > 1；不能因此声称独立 P/D 部署默认已有适用的动态 chunk 策略。[初始化条件][dynamic]。

<a id="pd-chunk-001"></a>
## PD-CHUNK-001｜用连续区间摘要替代全部页号的中间列表

**状态：源码已定位，待建立基线。类型：已有通用改法的 Ascend 适配候选；目标是保持传输语义。**

**当前怎么做。** Ascend 普通发送路径已合并源、目标同时连续的页区间。公共 helper 对每个区间生成完整 Python 页号列表；Ascend 后续构造地址时只读取每个区间的首项和长度。非 custom pool 路径也已经将各层描述符合并提交，不能再把这些称作新优化。[分组 helper][groups]、[Ascend 消费者][ascend-send]。

例如，页号为 100、101、102、103 的连续段，消费者实际只需要“从 100 开始，共 4 页”。这是表示方式示例，非性能测量。

**准备改哪一步。** 保留原始页扫描与连续性判定，让适用消费者直接接收 source_start、destination_start、length；避免把每一页再装箱到 Python 子列表。源或目标任一侧不连续，都必须切段。先限定普通 Ascend 发送分支，再评估其他布局，不能一次重写全部后端。

**收益与代价。**

- 希望减少页号装箱、中间列表分配和遍历。原始连续性扫描仍约为 O(页数)，不能声称整个算法变成 O(区间数)。
- 连续长段、分组准备 CPU 占比明显时更值得验证；几乎每页都碎片化时，区间数接近页数，收益可能很小。该 helper 在层循环之前调用，所省的逐页列表构造不随层数增加，各层地址描述符仍需生成。
- 新摘要结构、转换和兼容分支也有开销；小请求可能回退。CPU 改善若完全被计算隐藏，不一定降低用户延迟。
- 源 / 目标地址、长度、描述符顺序、层与 KV 布局必须等价。页映射可能变化时，不能跨发送缓存旧地址。取消、失败、源页释放与正在执行的传输仍要遵守原生命周期。

**已有工作。** [PR #45003](https://github.com/sgl-project/sglang/pull/45003) 已提出类似的通用 Mooncake 描述符优化；审阅时 open、未合并，变更文件不含 Ascend conn。本文保留的是 Ascend 路径适配及验证价值，不宣称思想首次提出，也不移用该 PR 的 GPU 性能数字。

**最小验证。** 固定相同页映射、层数、布局与传输数据，唯一变量为描述符构建方式。先比较描述符及实际拷贝字节，覆盖空输入、单区间、源连续而目标不连续、反向情况、碎片化、不同 stride、高位地址、失败清理；只在目标配置支持的分支上验证。再分开测 helper CPU、全部发送准备、提交时间、P 计算时间及端到端延迟。

**否定条件。** 目标运行路径未调用该 helper；没有明显中间分配成本；替代路径维护成本抵消收益；或只有微基准改善、真实服务收益低于噪声。任何传输正确性或生命周期差异都不能接受。

<a id="pd-chunk-002"></a>
## PD-CHUNK-002｜检查 KV 就绪后的等待，以及传完后的交接空隙

**状态：源码已定位，待建立时间线。类型：已有重叠机制的时序优化候选。包含两个独立测量点，每次实验只改一个。**

**当前怎么做。** 分块发送已经存在，但 overlap 模式下，中间块会推迟到结果处理时发送；传输完成后，还需轮询、提交元数据、入队并等待 D 调度。[P 的 defer 逻辑][p-process]、[D 完成队列][d-success]。

**准备改哪一步。**

1. **发送前：**区分“结果尚未安全就绪”和“结果已就绪却排在下一批调度 / stream 依赖后面”。只有后者才可能通过独立事件驱动的发送路径缩短。
2. **传完后：**区分“D 已经忙”“状态尚未完整”和“已满足条件却没被发现或提交”。只针对第三种情况考虑更及时的通知、减少重复提交或无效队列周转。

第一项已有 [PR #43811](https://github.com/sgl-project/sglang/pull/43811)：等待 batch result copy 完成后由独立 worker / stream 发起发送。审阅时 open、未合并；其支持条件限定 CUDA + Mooncake，并排除 staging、部分附加状态及其他模式，**不是可直接打开的 Ascend 优化开关**。Ascend 需要重新检查事件、stream、索引可见性和模型状态。

**收益与代价。** 如果传输早已具备条件却被后续计算挡住，提前提交可能隐藏更多传输；如果 D 已可运行却在等待完成检测，则缩短交接空隙可能改善继续生成时延。代价是 worker、锁、事件、通知和状态维护；更早搬运还会争用 HBM / 互联，可能使 P forward 变慢。D 长期饱和时，提前入队未必改变完成时间。

**正确性边界。** 每个请求只有一个发送所有者；chunk 边界要快照，不能读取下一块推进后的可变 end_idx；最后一段携带的 token、logprob、grammar、辅助状态必须就绪；取消和释放须等待正在访问源页的工作安全退出。TP / PP 的完成共识、内存可见性不能省略。

**最小验证。** 先记录每块“计算完成 → 可安全发送 → 提交传输”，再记录“必要数据完成 → D 提交 → D 首次继续 forward”。在冻结负载下，只改变一个已确认的额外等待。保护用例包括多块、最后不满页、共享前缀、取消、失败、重试、附加输出以及目标支持的并行布局。

**否定条件。** 等待来自真正的数据依赖；没有足够大的暴露空隙；D 始终被必要计算占满；或提前传输增加的争用 / 同步超过省下的等待。不得用“轮询次数减少”替代延迟证据。

<a id="pd-chunk-003"></a>
## PD-CHUNK-003｜在后端约束下选择计算块与传输段

**状态：源码已定位，待建立基线。类型：配置与实现联合研究，有明确取舍。**

**当前怎么做。** 计算块决定一次 forward 处理多少 token；传输段决定一次提交哪些 KV。固定源码已有页对齐及分段逻辑，按新增范围发送；后端 `get_max_transfer_tokens` 只在 DCP relayout / pack-buffer 等条件满足时返回上限，其他情况可返回 None。staging 又按固定网格对齐，将其传输段与配置的计算 chunk 关联起来。[边界和分段源码][p-segments]、[后端上限条件][transfer-limit]。

另外，`SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES` 在通用 Mooncake 路径被消费，Ascend 普通 `send_kvcache` 覆盖方法不读取它；某些委托通用路径的分支例外。因此不能把该参数当作普遍可用的 Ascend 传输段控制。[Ascend 路径][ascend-send]、[Mooncake 分批][mooncake-batching]。

**准备改哪一步。** 先确认当前后端是否允许独立改变传输段。若允许，固定计算 chunk，比较少量合法传输粒度；再固定传输策略比较计算粒度。只有单因素结果说明值得组合，才做小范围联合对照。若 staging 网格绑定，应先证明修改后的槽位和边界协议正确，不能只加一个“任意段长”参数。

**收益与代价。**

- 小段可能早发、早重叠，但增加描述符、提交、通知和固定通信成本。
- 大段可能提高传输效率，却推迟首发或让其他请求等更久。
- 计算块缩小增加组批与 checkpoint 次数；放大可能改变算子效率、长短请求等待、HBM 压力。
- staging 本身不是通用前提：所查 staging 实现包含 CUDA buffer / event，P 端对 MLA 有限制。不能据此承诺用户的 Ascend / GLM 配置可直接使用。[staging buffer][staging]、[P 端限制][p-staging-gate]。

**正确性边界。** 非最后段必须符合页与槽位要求；源与目的 token 范围完全一致且无漏传、重传覆盖；最后段和附加状态的完成条件保持；修改段长不允许提前复用还在传输的 buffer。

**最小验证。** 固定模型与请求流，逐次只改变一种粒度。同时记录每请求实际计算 token、发送字节、提交次数、CPU 准备时间、首发时刻、P 总时间、传输尾巴和客户端延迟；覆盖短请求、跨多块长请求、缓存命中、碎片页及尾段。**总字节数必须除以覆盖相同字节的时间窗**，不能用全请求字节除以最后一块耗时。

**否定条件。** 当前后端不支持安全解耦；请求不足一块；已经没有明显通信尾巴；或只提升某次传输带宽，却降低有效吞吐 / 恶化预先约定的延迟目标。

<a id="pd-chunk-004"></a>
## PD-CHUNK-004｜定位续块过程的重复准备，复用缓存优化档案

**状态：入口已定位，具体冗余待确认。类型：条件性等价工作候选。**

**当前怎么做。** P 续块会经过 PD checkpoint 和 extend batch 准备，包括输入切片、长度列表、索引与分配信息。[PD checkpoint][p-checkpoint]、[prepare_for_extend][batch-extend]。不同缓存 / bootstrap 状态有不同分支，不能把所有操作看成每块都可省。

**准备改哪一步。** 对相邻两块逐字段追踪：输入哪些不变、哪些只追加、哪些因映射 / 缓存变化必须更新。仅对重复构造的不可变段使用游标、复用或增量准备。已有 [CACHE-013](2026-10-18-sglang-cache-audit.md#cache-013)、[CACHE-019](2026-10-18-sglang-cache-audit.md#cache-019) 分别讨论 checkpoint 前多次裁剪、索引独立副本的时机；这里增加 PD 适用性检查，**不重复登记成新的缓存发明**。

**收益与代价。** 多块长输入更容易积累重复准备成本；单块、短输入或 CPU 已完全被计算隐藏时收益弱。复用会引入版本、生命周期、容量和失效维护；复杂缓存模式可能得不偿失。

**正确性边界。** checkpoint 还承担缓存状态与引用维护，不能直接删除；树长期持有的索引与请求可变 buffer 不得错误别名；异步 device copy 结束前不能重用 pinned buffer。重试、页迁移、请求取消、缓存匹配变化都要触发正确失效。

**最小验证。** 先 profile checkpoint、切片、组批、索引分配各自时间与规模；挑一个确认重复的操作，只改该操作。比较完整状态和请求输出，覆盖跨块、共享前缀、重试、取消及缓存开关；同时测块间设备空隙和总延迟。

**否定条件。** 没有重复构造；复制用于必要所有权隔离；维护失效比重建更贵；或局部改进没有可见的服务收益。本条目前不能写成“已有可直接删除的 checkpoint”。

<a id="pd-chunk-005"></a>
## PD-CHUNK-005｜D KV 预留时机：现有能力、折中方案与适配限制

**状态：现有机制已确认；用户配置适用性待确认。类型：已有策略对照 / 后端适配研究。**

**当前怎么做。** 默认 early 尽早准备 D 的目标 KV。已有 prefill_complete 将 D 目标 KV 预留及接收地址发布推迟到 P 接受最后一次 prefill 结果后；前期 receiver 初始化和协调仍然存在。变化的是每请求占住容量的时机，**不是物理 KV 池大小，也不是让分配算子更快**。[固定验证条件][allocation-hook]、[已合并 PR #40703](https://github.com/sgl-project/sglang/pull/40703)。

**准备比较什么。**

- early：较早具备传输条件，但未完成 P 的请求可能先占住 D 容量。
- prefill_complete：缩短部分 D 空等占用，但丢失本请求计算中间块时的传输重叠机会，并可能延长 P 保留 KV 的时间。
- first_chunk：已有 [PR #44891](https://github.com/sgl-project/sglang/pull/44891) 探索第一块接受后再触发**整请求的一次性目标预留**。这不是 D 每来一块才分配一块；审阅时该 PR 已关闭、未合并，不能当作稳定可用功能，也不据关闭状态推断方案被性能否定。

固定源码的 prefill_complete 校验只允许 Mooncake，且要求 attention TP、CP、PP 等满足限制；staging 等组合也被排除。[PR #43970](https://github.com/sgl-project/sglang/pull/43970) 提议放行 Ascend 后端，审阅时 open、未合并。仅修改参数校验，不构成 Ascend 正确性、性能和完整生命周期的证明。

**收益与代价。** D KV 紧张、长短 prefill 混合且空等预留显著时，延迟预留有机会改善容量周转。低负载或网络较慢时，失去重叠可能直接增加交接尾巴。更复杂的就绪协议还增加握手、重试、状态和故障清理成本。

**正确性边界。** P/D 策略一致；目标地址发布后遵守原传输和释放协议；取消、过期请求、重入和重试不能把旧地址交给新请求。不能通过无条件停住 P 来破坏推进。

**最小验证。** 先确认后端 / 拓扑支持，再在完全相同请求流下对照两个合法策略。记录 P/D 每请求占用的容量积分（实际保留字节 × 时间）、峰值、队列、重试、交接尾巴、吞吐及客户端延迟；覆盖长短混合、容量边缘、取消和失败。

**否定条件。** 目标配置不支持；D 容量没有形成瓶颈；减少的空等预留未转化为有效服务能力；或 P 积压、丢失重叠造成的回退超过收益。不要仅凭 D 的某一时刻显存数字下降判定成功。

<a id="pd-chunk-006"></a>
## PD-CHUNK-006｜只针对已证实的有害积压，增加有界准入反馈

**状态：条件性系统策略，待证明现有控制不足。类型：改变准入顺序的策略研究，优先级低于具体冗余。**

**当前怎么做。** D 已有容量预分配与传输握手，P 也有等待、重试和资源回收逻辑。不是“P 可以无限生产，系统完全没有背压”。[D preallocation 队列][d-prealloc]。已合并 [PR #43442](https://github.com/sgl-project/sglang/pull/43442) 还专门处理了 parked optimistic prefill 占满请求槽、双方等待的问题。

**准备改哪一步。** 只有观察到网络 / D 排空速度持续低于 P 的产出，且现有机制仍导致无效重试、源 KV 长期滞留或尾延迟恶化时，才考虑用待发送字节、滞留年龄、可用目标容量共同限制新 P 工作。优先检验简单的在途上限是否足够；有证据再增加高低水位与有界反馈。

**收益与代价。** 有望减少竞争、无效生产和长期占用，让已接近完成的请求更快排空。代价是遥测、跨节点信息延迟、控制状态和排序；限制过强会让 P 或网络空闲，反馈不稳会振荡。不同请求的 KV 大小不同，仅按请求数判断容易误判。

**正确性与公平性边界。** 不能挡住解除等待所需的握手、传输、清理及已在途请求；保留超时和恢复路径。不能因 D 已承诺 A、P 却只保留 B 而卡死。长请求必须有进展保证，取消不能留下过期计数。

**最小验证。** 先冻结已有配置，证实积压与重试 / 尾延迟之间的因果；只增加一种简单限流因素。覆盖低负载、稳态饱和、突发、D 暂停 / 恢复、网络变慢、长短混合及取消。记录 P 空闲、网络空闲、积压字节 / 年龄、两端容量、失败 / 重试、各长度组延迟和有效吞吐。

**否定条件。** 当前容量准入已充分控制；积压只是必要流水深度；减少积压却降低吞吐或恶化目标延迟；新增控制无法稳定或破坏长请求进展。本条不能包装成普遍“逻辑非负”的优化。

## 两个保留的配置对照

**配置 A：P 计算块大小。** 比较的是整体工作，而非“每个 chunk 越快越好”。固定实际未缓存输入量、上下文、请求流和并行布局；选择少量合法大小，对照 P 总时间、组批次数、attention / MoE 等实际算子、HBM 压力、首次发送和最终交接。保留既有资源与生命周期边界。新增成本主要是更多批次或更大瞬时资源压力；单块负载和饱和负载可能结论相反。若只有局部算子改善而服务目标不变或变差，不采用。需要动态策略时，先证明静态选择在不同负载下确有冲突，再讨论预测成本与失效。

**配置 B：长短 prefill 的预算分配。** 固定源码的 [shortest_prefill_chunk_limit][shortest] 已能在对应策略下给较短等待请求留预算。对照现有策略即可，不先发明同名机制。收益是短请求更早服务；代价是排序、缓存局部性变化与长请求推迟。固定到达序列、前缀和容量，分别报告长短组 p50 / p95 / p99、总吞吐及最长等待；输出与取消清理仍须正确。若短请求改善以不可接受的长请求饥饿为代价，或全部请求差不多长却多了调度开销，则不采用。这里的目标是 P 队列服务质量，不是避免同卡 decode 被 prefill 挡住。

## 明确排除的说法

1. **“PD 后把 chunk 调小，就能直接保护同卡 decode。”** 本记录是专用 P/D 资源，该混部因果不成立。
2. **“chunk / 传输段越大越好，或越小越好。”** 都忽略计算效率、固定开销、排队和重叠的取舍。
3. **“传完一块立刻释放 P 的这一块 KV。”** 后续 attention 或在途搬运可能仍需要它；必须先证明最后使用结束。
4. **“D 收到第一块就能正常 decode。”** 普通路径仍需完整依赖与状态提交。
5. **“有异步接口就一定重叠，重叠就一定变快。”** 依赖、stream 排队、HBM / 互联争用都可能阻止收益。
6. **“连续页合并、合层提交、增量发送都是待新增能力。”** 固定源码已有相关路径，剩余候选须落到具体额外工作。
7. **“GPU / Mooncake PR 已有结果，所以 Ascend / GLM 直接受益。”** 后端、模型 KV / 状态布局、并行和版本均需复核。
8. **“显存下降、CPU 变快或带宽数字变大，就证明端到端更快。”** 三者都必须关联到相同请求和关键路径。

## 后续实验的共同约定

这是下一次实施的输入合同，本次没有执行实验。

| 必须固定 / 记录 | 要回答的问题 |
| --- | --- |
| 运行 commit、镜像、模型 / 量化、P/D 资源和并行、后端、缓存 / staging 配置 | 当前实际走的是哪个实现？ |
| 相同 token 化输入、输出上限、到达序列、随机种子 / 采样策略、冷热缓存、并发与预热 | 前后是否完成了同样的服务任务？ |
| 单请求、多块请求、长短混合、突发及容量紧张负载 | 改善依赖什么条件，会在哪里回退？ |
| 每块计算、结果就绪、页处理、传输提交 / 完成、D 提交 / forward、客户端首 token 与结束 | 省掉的工作是否在请求关键路径？ |
| 正确性、成功率、取消 / 重试、KV 字节校验、资源是否回归 | 更快是否以错误、泄漏或失败为代价？ |
| 局部 CPU、设备空隙、P/D 容量积分、p50 / p95 / p99、有效吞吐、重复运行波动 | 局部变化与服务收益分别是多少？ |

跨机器时间戳必须对齐或建立可校验的分布式 trace；不能直接相减不同机器的单调时钟。校验模式若增加计算 / 传输，应与正式性能测量分开。采样结果可能受数值顺序影响，须使用与变更相称的 token / 质量验收，不能仅用“请求没报错”替代正确性。

执行时每个 Loop 只改变一个已冻结机制，预先写明预测和否定条件。没有 matched baseline 的数字留空；发现反证追加到本记录，不删除原假设。

## 已有工作核对

下表为 2026-10-18 的 GitHub API 查询结果。只覆盖本次直接相关 PR，不是全库创新性穷尽检索。完整 head SHA 和来源清单保存在 [来源记录](evidence/2026-10-18-pd-chunk-sources.json)。

| PR | 当次状态 | 对本记录的意义 |
| --- | --- | --- |
| [#40703](https://github.com/sgl-project/sglang/pull/40703) | 已合并 | prefill_complete 已有机制。 |
| [#43442](https://github.com/sgl-project/sglang/pull/43442) | 已合并 | 请求槽 / optimistic prefill 的推进问题已有处理，不能用粗暴背压覆盖。 |
| [#43811](https://github.com/sgl-project/sglang/pull/43811) | open，未合并 | KV 就绪后提前发送已有具体实现提案；CUDA / Mooncake 等限制需保留。 |
| [#43970](https://github.com/sgl-project/sglang/pull/43970) | open，未合并 | 提议在 prefill_complete 校验中允许 Ascend；不能等同完整验证。 |
| [#44891](https://github.com/sgl-project/sglang/pull/44891) | closed，未合并 | first_chunk 已有提案；第一块触发整请求预留，不是逐块分配。 |
| [#45003](https://github.com/sgl-project/sglang/pull/45003) | open，未合并 | 通用 Mooncake 描述符已有优化提案；Ascend 适配仍需独立检查。 |

## 结果追加

- 2026-10-18：完成固定源码与相关 PR 的逻辑审阅；六条记录均未建立用户 Ascend PD 性能基线，没有实机加速结论。
- 后续结果按 PD-CHUNK ID 追加 Run、配置、唯一改动、正确性、重复次数、局部与端到端结果、反证及适用边界。
- 本文件不改变任何模型研究线的 Current / KEEP / next_action。

[groups]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/common/utils.py#L109-L134
[ascend-send]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/ascend/conn.py#L305-L410
[p-middle]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/prefill.py#L1076-L1085
[p-cached]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/prefill.py#L1417-L1452
[p-success]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/prefill.py#L1179-L1230
[p-first]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/prefill.py#L971-L1032
[p-process]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/prefill.py#L1369-L1407
[p-checkpoint]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/prefill.py#L564-L581
[p-segments]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/prefill.py#L1473-L1694
[d-success]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/decode.py#L2743-L2882
[d-token]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/decode.py#L2595-L2633
[d-prealloc]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/decode.py#L361-L525
[dynamic]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/managers/scheduler.py#L1357-L1375
[shortest]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/managers/schedule_policy.py#L414-L437
[batch-extend]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/managers/schedule_batch.py#L2811-L2940
[allocation-hook]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L198-L240
[transfer-limit]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/common/conn.py#L2034-L2045
[mooncake-batching]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/mooncake/conn.py#L1178-L1194
[staging]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/common/staging_buffer.py#L138-L164
[p-staging-gate]: https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/disaggregation/prefill.py#L192-L204

## 2026-10-19 PD-CHUNK-001 CPU实证

[同一候选的151组等价测试和局部CPU成本](sglang-pd-spec-36/evidence/2026-10-19-descriptor-cpu/README.md)：已有连续区间合并保留，只减少中间页号列表；三个连续Case局部收益可区分，碎片时延不确定。没有NPU或端到端收益结论，继续真实consumer与服务时间线。
