# F12：单窗口 Indexer → Attention 可恢复记录

## 固定样本与已观察事实

本记录选既有 **EP8、model 49、main stream 89、配置角色第 24 层、内部采样 run 2**；窗口 `[1793152960208017846, 1793152960209362219] ns`。定位依据是 `captured-layer-attribution/module-call-metrics.json` 的 source-validated role/window 映射；对历史 binary 真正调用栈的绑定仍有限。`agents/diagnosis/indexer-runtime-metadata-ep8.json` 中 QLI TASK 在窗口起点后 `193.216–691.707 µs`，耗时 **498.491 µs**，task ID 1275，输出 shape 字符串 `"3,1,512;;"`，输入包含 `[3,32,128] INT8`、`[3,1,2048] INT32` 及其余输入；这是物理 shape，未提供动态有效长度或 tensor 内容。

同一窗口 `agents/fusion/f07-q-kv-streams.json` 保存的 `SparseFlashMla` TASK 在起点后 `706.886–751.985 µs`，耗时 **45.099 µs**，同 model/stream、task ID 1277；输入序列出现 `[3,1,512] INT32`，输出 `[3,64,512] BF16`。两记录之间间隔 15.179 µs，并不等于空转、无其他任务或可节省时间。QLI 的输出与 SMLA 的输入 shape 相容，但这里尚无实际地址、传递对象和 call stack 的逐条绑定，无法证明是同一 tensor 或 QLI 输出未经转换后直接供 SMLA。`globalTaskId=19466`/connection ID 在多轮重用，不是请求唯一 ID。不能由 attr 字符串中的未命名 `attr_8:2` 等直接证明参数含义或实际候选数。

三轮这一层的同形 QLI 用时分别为 32.020、498.491、498.170 µs；未知 query/cache 的有效长度和候选值，同形不等于同工作量，不能从差值直接归因融合、调度或带宽。数字仅为该 rank 的历史采样，不代表稳态、整机端到端收益或可复现实测的理论上界。

## 源码约束下的预期链

冻结配置的层 24 为 compression ratio 1、index source 24、KV source 20、candidate-filter 角色。若历史调用确实使用当前 A3 non-packed 源码及对应 metadata，`dsa_v41.py:567–603` 在 index source 等待量化 query ready 后调用 `indexer.select_projected`，后者在 `indexer.py:254–296` 先等待 metadata，再以 mode 2、candidate blocks 输入调用 QLI。QLI 返回 token 位置后经 `selected.squeeze(1)`→`prepare_indexer_indices`：按 `(position+1)//compress_ratio` 过滤无效位置，恢复 `-1` sentinel，并将有效位置按原 helper 的顺序规则排序，写入 graph-stable INT32 `shared.topk_indices[:3]`（这里 `3` 仅是该 trace 输入/输出的匹配物理行数，不证明源码该次取值）。随后 `dsa_v41.py:604–611` 可能自复制共享切片，返回共享索引；`_forward_attention` 使用 `pad_sparse_indices` 并将其作为 `cmp_sparse_indices` 提供 SMLA。第 25–27 层的非 source 在该配置下复用 24 的索引。候选列表是 block ID，与 SMLA 的 token 位置不同；层 20 发布的 candidate 必须仍对层 24 有效。

以上是调用与角色条件推导；当前无法以这些 shape 推断具体对象别名、可见位置、实际请求顺序或该层某次是否进入空缓存早返回。QLI 设备记录支持该工作窗口存在该算子，但不单独证明全部源码路径。

## 已融合、候选与阻碍

- **已融合**：`prepare_indexer_indices` 的单个 Triton kernel 已组合过滤、排序、写输出；满足阈值时还融合长度求和。helper 已支持调用方提供稳定 `indices_output`。`SparseFlashMla` 本身为复合算子；不能把它们拆成未经证实的多个待消除 HBM 写或多个可节省 launch。
- **候选**：研究在 QLI 的最终输出阶段完成同等有效性过滤、位置排序与稳定地址发布，或让完全完成的每行/行块交付 SMLA；仍需保留 source 的 candidate block 输入/作用、topk tie 行为、所有 sentinel、长度、图地址和跨层复用。非 packed 分支局部上 SMLA 没有显式消费 topk_lengths，是否可精简仍受外部消费者、图和 packed 分支限制。
- **阻碍**：当前 Python QLI 调用无 out buffer 参数；现有接口未证明按行完成通知、kernel 中的 topk 稳定顺序或可分块消费；模型的后续 source 写同一共享 buffer 时不能覆盖仍在读取的索引。涉及辅助流 query quantize、metadata wait 及注意力前缓存写入，不能凭近邻时间删除同步。`prepare` 输出 dtype/shape、排序和每个实际 token 取值需保留，FP16 query/key scale、INT8 输入以及 SMLA 语义亦须保留。

## stage147：补齐同一窗口内的三段设备记录

ZCode 只读查询 `query_f12_layer24_window.py` / `run_f12_layer24_window.py` 成功，`/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-147-f12-layer24.exit-code` 为 0，`.verified.json` 记 `model_verified=true` 且 observed `deepseek-flash`。结果 `f12-layer24-run2-evidence.json` SHA256 `4011fa057341ccb02b1436d6ee4beec66dd54124415664b3ad47a38e640591b2`，查询脚本 SHA256 `094bd4228af1c9ce3eb0ca1e08627c427b151d629348921eedafeb788be13a86`；输入 EP8 CSV SHA256 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7`、窗口 SHA256 `d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a` 均与原始清单匹配。

| 记录 | 相对窗口起点 µs | 时长 µs | stream / task | 记录的关键数据 |
|---|---:|---:|---|---|
| `VllmAscendQuantLightningIndexerV2` | 193.216–691.707 | 498.491 | 89 / 1275 | 输出 `[3,1,512] INT32` |
| `_prepare_indexer_indices_kernel` | 692.527–705.606 | 13.079 | 89 / 1276 | model ID `4294967295`；输入输出 shape / dtype 为 N/A |
| `SparseFlashMla` | 706.886–751.985 | 45.099 | 89 / 1277 | 输入中有 `[3,1,512] INT32` |

QLI、helper、SMLA 在同一 stream 顺序出现，helper 名称与源码转换步骤一致；这使三段链条在该工作窗口中有更强的存在证据。但 helper model ID 与两侧不同且 profiler 未提供其输入/输出，连续 task ID、dtype/shape 相容、源码角色和时间顺序**都无法证明三个事件的数据指针一致、helper 确是同一次调用、过滤/排序具体值、属于哪个请求或关键路径**。查询只列完全位于 QLI 结束与 SMLA 开始之间的事件，边缘重叠事件可能被排除；不能说此窗口不存在其他并行执行。13.079 µs 是该 task 的观测时长，不等同可通过融合节省的时间。

理论边界仍是每行所选位置与可见性确定后才能完成原排序并为该行注意力提供完整索引；不要求其他不相关行都完成，但现有 operator 接口和事件未证明可逐行交付。若要进一步唯一绑定值和生命周期，需要历史采样中的 tensor 地址/数据依赖或调用栈证据；若不存在，保持 UNKNOWN，不做无针对性测试。

## 对 498.491 µs QLI 的内部重组边界：分片局部 TopK 与合并（源码/代数候选）

上述 F12 已讨论 QLI 的输出过滤与逐行发布；这里进一步研究**单行内部**：把可见候选划成互不重叠的块，各块完成与原实现同样的分数计算和同一个全序比较规则，保留每块前 `k`，再对这些局部结果合并取全局前 `k`。若全局候选 `(score, index)` 的比较规则是确定且一致的全序，这个方案严格保留全局结果：任何没有进入本块前 `k` 的元素，其同块已有至少 `k` 个元素排在它之前，因此它不可能进入全局前 `k`。不足 `k` 的块保留全部有效元素，mask/可见性和重复候选必须按真实算子规则处理；不能假设 QLI 的 tie、NaN 或重复位置规则已知。目标 `k=512` 是此历史样本输出维度所示的物理值；不能从候选 block 输入形状推断有效候选数量。

划分可让一个分数块的产生与另一块的局部选择流水执行，但这只是**可调度性候选**：各块如果分别占用同一 Vector、Cube、MTE 或共享带宽，就有资源争用；局部结果缓冲最多可能达到 `块数 × k` 项，分数重排、写回、合并和同步可能抵消好处。QLI 本身是已存在的复合算子，历史 trace 看不到内部 tile/topk 图，不能声称目前没有此类流水或将它的 498.491 µs 当作可节省时间。公开 [DeepSeek DeepSelect](https://github.com/deepseek-ai/DeepSelect) 提供面向 DSA 的 TopK，README 声明支持 Ascend BF16、`topk=512` 示例、按索引排序和外部输出缓冲；这是另一个库的能力，**不证明**它与此 QLI 的内部 score dtype、量化误差、比较和数值语义兼容，也不证明分离现有融合算子后会更快。Ascend 版本对 NaN 的契约、环境、安装及本工作负载是否命中也需独立核实；不能直接套用其相对 `torch.topk` 的宣传倍数。

另一条跨 QLI/SMLA 的条件候选是在最终选择尚未完成时预取候选 KV 页并进行不依赖最终集合的地址准备，但**不能把局部 topk 候选直接作为最终 attention 贡献**。例如 `k=2`，两块得分分别 `[9,8,1]`、`[10,7,6]`，第一块局部前二里的 8 被全局前二 10、9 排除；若已累加 8 的 value，必须撤销且仍需维持 softmax 归一化。更稳妥的假设是只准备无副作用且可丢弃的数据，真正计算/发布前等待确定集合，原压缩 block、mask、cache 页版本和 F13 的 sink/softmax 语义均不可变；预读可能浪费带宽、挤占 QLI、破坏原有 overlap。没有看到 169 的设备源码暴露可访问的 score tile、候选 readiness 或 prefetch 接口；现有证据也不包含页内容、cache 命中、资源时序或足够的事件绑定，无法量化。

下一步优先核查**已保存**的 QLI 对应实现/接口证据，判断 score 精度、选择和比较约定、内部是否已有局部规约及 tile 可见性；无此证据则保持设计级候选与收益 UNKNOWN，而非将库适配或新测试当作本阶段完成标准。该分析与 F12 的输出阶段、F13 的分支 softmax 及 F19 的缓存版本约束重叠，不能将预计收益相加。

## stage220：撤回“当前源码内部局部排序未知”，保留历史二进制未知

**修正上节**：当前容器的 arch22 QLI 源码已包含分块排序、跨块归并和内部流水，上文“当前源码内部是否已有局部规约未知”已不再成立；“分块局部 TopK 再合并”的基本思想不能当作新增机会。实际 EP8 第 24 层 498.491 µs 的历史二进制是否由这份源码构建、是否运行这一分支，仍未绑定，因此不能把源码静态行为当成历史调用直接观测，更不能据此计算收益。

- **源码来源和范围**：ZCode stage `operator-ceiling169-v2-220-qli-native-source` exit 0，`model_verified=true`，模型为 `deepseek-flash`；快照只读现有镜像 `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b` 的指定源码子树，无目标子树覆盖挂载。来源文件清单见 `qli-v2-source-provenance.json`，SHA256 `061246e242618916398367fd1f742392902c17eb028a02e3bb44c5574f8dac6c`。`op_kernel/quant_lightning_indexer_v2.cpp:18–22,85–87` 在编译宏分支中区分 arch35 和 arch22，并对 arch22 指定 INT8 query/key；`op_host/vllm_ascend_quant_lightning_indexer_v2_def.cpp:104–105` 注册 `ascend910b` 和 `ascend910_93`。下述分析仅适用于所读 arch22 源码，不转用 arch35。
- **分块排序已有实现**：`op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h:93,201` 的 S2 基本块为 2048；`quant_lightning_indexer_v2_service_vector_arch22.h:29–30,838–845,863–888` 对每块生成 score 和索引、执行 `SortAll`，再 `MergeSort` 到 `globalTopkUb_`，内部 `BASE_TOPK=2048`。`:890–917` 在该行 S2 结束且无需 LD 归约时提取索引，按实际 `sparseCount` 写出。历史样本输出为 512，并不等于内部缓冲容量也是 512；LD 分支另有 workspace 归约，不能无条件认定所有行无需跨核合并。
- **类型与中间缓冲**：`quant_lightning_indexer_v2_service_cube_arch22.h:201–230,519–554` 的 QK 经 `DEQF16` 和 ReLU 送 L1；`:502–515` 后续加权矩阵计算使用 half 输入和 float 累加，`:558–579` 写出 FP32 GM score。vector `:830–845` 将 FP16 key scale 转为 FP32 并与 score 相乘。因此不能把整个 QK 到最终 score 的计算说成全程 FP32；历史实例的实际输入和值、位级舍入仍未绑定。
- **候选块是另一层语义**：`service_vector_arch22.h:880–882,919–921` 的 mode1 条件下还产生候选 block 的 TopK。mode2 `:846–857,876–879` 将候选外位置分数降到 `NEG_HUGE`，源码注释明确候选内有效位置不足 topk 时，候选外可达位置可以补入。不能以旧设计文档里的“候选外一律为负无穷”代替这份源码的当前行为；样本是否实际命中 mode2、输入候选和结果仍未知。
- **资源与同步**：`kernel_arch22.h:473–493` 将 score 放入按 loop%2 使用的 GM 双缓冲，另有权重和条件性归约 workspace；`:735–758,780–815` 显示 Cube `ComputeMm1` 与 Vector `ProcessVec1` 对相邻 tile 使用双槽与跨核 flag 协作，Vec0 和 Cube 亦有同步。此机制只证明源码设计有流水与等待边界，不证明该历史调用的实际重叠率或等待耗时，也不能证明 498.491 µs 的根因。潜在消除 GM 往返或改变分块的方案必须重新考虑 buffer 容量、数值、先后依赖和 Cube/Vector/MTE 争用，不能直接套用 arch35 的路径。

关键文件 SHA256：`kernel_arch22.h` `d35c26e6cb781aebad163a2c85cbd84f68c7c838ebe5ef9f67e6a42777949df6`；`service_vector_arch22.h` `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8`；`service_cube_arch22.h` `2b586bb8c894fbb0ea6887c7d0bd69332f2ab54570294ee6f407ecf346296352`；host def `9522ae20fe820bfc3f7094778c1c5f7c46b813ca2bc5e3e57fecc703d97390fb`。下一步仅在既有证据中查源码到历史 binary 的绑定和 kernel 内阶段计时，缺失则保持未知。

## QLI score 的生产—消费约束与条件性流量

以下仅描述 stage220 绑定的当前 arch22 源码，不是历史 498.491 µs 调用的内部实测。`service_cube_arch22.h:183–195,502–515,558–579` 先以 half 输入、float 累加计算加权 score，再由 Fixpipe 写入 `mm1ResGm`；实际写入调用使用 `s2L0RealSize` 和 `s1L0RealCount`，并带 NZ→ND 设置，尾块、对齐和设备实际传输粒度不能从这一处当成已知字节数。`kernel_arch22.h:473–480` 按 AIC 和双槽分配 score workspace；分配容量是空间占用，并非实际传输流量。

`service_vector_arch22.h:810–840` 中，当某行的 `cuRealAcSeq > 0` 且 `cuS2Len > 0`，代码单次 `DataCopyPad` 请求从 score workspace 读入 **`4 × cuS2Len` 字节**，然后加载 key scale、转换并乘到 score；`846–888` 再执行 mode 2 的候选降级、块内排序及与已保存 TopK 归并。令有效 `(行, tile)` 集合为 E，则该实现的 score 消费侧请求字节为 `4 × Σ_(r,t∈E) cuS2Len(r,t)`；这不包括 producer 写、scale、候选、LD 归约、输入和输出，也不等于物理 HBM 字节（缓存、传输粒度未知）。实际 E 和长度未由现有 trace 给出，无法替 498 µs 代入数值。历史输出 shape `[3,1,512] INT32` 仅对应 6,144 字节的**逻辑输出张量容量**，不证明当次写出字节、有效位置数或带宽。

`kernel_arch22.h:790–815` 的 Cube 在处理当前有效 tile 前等待反向 flag，完成 `ComputeMm1` 后发送正向 flag；Vector 等待前一有效 tile 的正向 flag、调用 `ProcessVec1` 后再发送反向 flag。结合 `loop % 2` 的 workspace 选择，这是源码内生产完成、消费和复用次序的证据；不能仅凭指令位置推断等待持续时间、传输是否与计算重叠或断言历史瓶颈。

**结构重组候选**：研究让 score tile 在生产后经设备支持的更近存储或分块交付给排序侧，减少现有 GM workspace 的逻辑写读。该方案须证明当前 910C Cube→Vector 路径、缓冲容量、可见性与复用同步接口确实支持，并保留相同 FP32 score 的舍入顺序、key scale、候选降级、尾块及跨核 LD 归约。尤其 mode 2 的候选外位置在不足 topk 时可补入，不能简单把候选外 score 全部跳过。减少的逻辑访问量不自动等于减少物理 HBM 流量，也不意味着消除全部等待；与现有 Cube/Vector 流水重叠、资源争用和首个结果交付时间均需单独判断。现有设备内部计数、历史二进制绑定和有效长度均缺失，该候选没有收益数值或已经可实现的结论。

## 既有匿名属性与当前源码的条件性对照

历史 EP8 层 24 run 2 的原始 TASK 属性字符串保存于 `agents/diagnosis/indexer-runtime-metadata-ep8.json`，其中 `attr_8:2|attr_9:2048|attr_10:8`。当前镜像冻结的 `op_host/vllm_ascend_quant_lightning_indexer_v2_def.cpp:84–96` 按注册顺序列出属性，序号 8–10 分别是 `candidate_mode`、`candidate_topk_blocks`、`candidate_block_size`；`op_host/quant_lightning_indexer_v2_tiling.h:73,109–111` 将 `ATTR_CANDIDATE_MODE_INDEX` 定为 8，并将值 2 定义为 consumer。

**仅当**（1）该版本 profiler 的 `attr_N` 序号与 host 注册属性索引对应，且（2）历史 task 实际使用与当前冻结源码相同的属性注册定义，才可推断该 task 的 `candidate_mode=2`，并对应候选块参数 2048、8。这两项尚未证实，当前只是值和顺序一致的线索；不能把它升级为历史运行模式、实际有效候选数或实际执行分支。TASK 的 `cann_api_connection_candidates` 为空，现有保存记录也没有可绑定到该调用的源码构建 hash 或 tiling key。

定向查阅本任务已保存的 profiler 相关脚本及结果，未发现该版本 profiler 将 `attr_N` 编码为 host 属性索引的实现或规范。后续 stage224 对当前容器 CANN 9.2 的 `tools/profiler/profiler_tool/analysis` 根下 759 个 Python/SQL 文件，以 `attrInfo`、`attr_info` 和若干 `attr_` 形式进行有界关键词查询，没有跳过文件；16 个命中文件的命中行没有直接给出序号与 host 注册索引的映射。查询仅覆盖这些路径、文件类型和关键词，其他路径、拼写和 native 实现仍可能存在；`NodeAttrInfo` 解析文件的出现也不能直接证明它处理目标 `COMPUTE_TASK_INFO.attrInfo`。固定脚本 `agents/diagnosis/query_qli_attr_export_contract.py` SHA-256 `da628f0a7a329fe728fc38d6f67dcd08f65c4d4a455ffa58d1c51ed8b419aa27`，结果 `agents/diagnosis/qli-attr-export-contract.json` SHA-256 `b220b79f62032c6757bdf0d0f35e2ac23ac90a6356b4629088c122627d8d98d2`；stage224 exit 0、`model_verified=true`。profiler 的序号语义仍为 UNKNOWN；当前不能从已保存证据将这条历史 task 唯一绑定到 stage220 的 arch22 二进制或具体 tiling，不能把静态代码路径当成观测事实。

补充公开资料核查：昇腾 [CANN 8.2.RC1 Graph 采集数据表文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/devaids/Profiling/atlasprofiling_16_0053.html) 将 `COMPUTE_TASK_INFO.attrInfo` 描述为映射到 STRING_IDS 的算子属性信息，但未定义 `attr_N` 的排序规则；该文档版本也不等于历史采样自报的 CANN 9.2，不能据此证明两版本实现一致。对公开同版本格式源码的定向搜索未找到能够证明映射的实现。这个缺口不会改变上述两项前提。

### 进一步约束：一个 Cube 的 score tile 分给两个 Vector

在上述同一份 arch22 源码里，AIV 的 `GetBlockIdx()/2` 和 AIC 的 `GetBlockIdx()` 选择同一个 `aiCoreIdx`（`kernel_arch22.h:455–460,478–480`）。`service_vector_arch22.h:785–795` 则把一个 tile 的有效 S1 行按 AIV 奇偶切成 `ceil(n/2)` 与 `floor(n/2)` 两段，分别从该 Cube 的 workspace 不同偏移读取；尾块甚至可能让一个 AIV 无行处理。因此，片上交付候选需要处理 **一对 AIV 的行分发及各自完成**，不能简单设想整个 tile 只有一个消费者，或把当前一个 AIV 的私有 UB 当作可供两个 AIV 直接共享。这里未证明 910C 提供可行的 Cube 到两个私有 UB 的直达接口，也未从当前源码证明跨核 flag 的底层实现细节。任何改写必须保证两个分支相关读完成后，生产者才可安全复用相应数据区，并保持零行、尾块和跨核 LD 的正确性。

该实现固定 `S1_BASE_SIZE=4`、`S2_BASE_SIZE=2048`、`MM1_OUT_T=float`，故源码分配的 score workspace 每 AIC 每槽 `4×2048×4=32768` 字节、双槽 65536 字节；满四行时各 AIV 负责两行、每槽各 16384 字节。这是**逻辑容量**，不是实际读写量，也不证明物理上可直接交付。

`service_vector_arch22.h:214–239,826–845` 还给出一个资源约束：每个 AIV 的输入队列每槽 `2048×4×2=16384` 字节、共两个槽；每行消费时，score 与 FP32 key scale 共用这一槽，FP16 key scale 又复用其中后半区域，之后才释放。不能把它当作整块空闲的 score 暂存空间。依据该源码 `InitBuffer` 的参数计算，基础分配约 144 KiB；candidate source 额外 32 KiB，candidate consumer 额外 32+8 KiB，分别为 176 KiB 和 184 KiB。源码同段注明的 192 KiB 是实现侧预算注释，而非本次已证实的硬件可用上限；同段较早注释将 consumer 说成 168 KiB，和实际申请的 32 KiB candidate buffer 不一致，应以具体申请式而非旧注释作推算。在这个 **192 KiB 条件预算**下 consumer 只剩 8 KiB，因而无修改地再叠加两个 16 KiB score 暂存槽会超预算；重排或复用现存内存、改变流水粒度仍可能，不能由此断言所有片上方案不可行。当前未绑定历史二进制、实际模式与可用 UB，所以这些容量分析不是历史调用的实测，也没有性能收益数值。

## stage222：精确任务的 profiler 名称和复用范围

ZCode stage `operator-ceiling169-v2-222-qli-provenance` 的 `.exit-code` 为 0，`.verified.json` 的 `model_verified=true` 且记录模型 `deepseek-flash`。固定只读查询 `agents/diagnosis/query_qli_compute_provenance_exact.py` SHA256 `d291947d04350b1d64906761849b694eea0fcb9cd62bf939639d5c8bebfb37eb`；输出 `agents/diagnosis/qli-compute-provenance-exact.json` SHA256 `b7f796be644d969ce48c5dc14e238756a51399457877bab8ac4b113349a9186f`。查询先核对 EP8 profiler 数据库 SHA256 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`，再按 model 49、stream 89、task 1275 及精确起止时间定位唯一 TASK，通过 `globalTaskId` 查 `COMPUTE_TASK_INFO`。

该记录的 `name` 和 `opType` 均解析为同一个泛化字符串 `VllmAscendQuantLightningIndexerV2`，没有提供具体编译对象文件名或可识别的 binary hash。`globalTaskId=19466` 在该数据库的 TASK 表中关联 **7 行**，故它并非此 replay 的唯一标识；精确起止时间用于定位本条观测，不能因此还原设备内部阶段或动态 tiling。记录另有 `blockNum=24`、`mixBlockNum=48`、`gridDim=5`、`blockDim=5`，但编码含义及与 tiling 的关系未核实，不能将这些值解释为真实有效 tile 或算力。

本次还检查了该数据库 `sqlite_master` 中名称含 `tiling`、`kernel` 或 `binary` 的表，**无名称匹配项**。这个检查只覆盖指定名称模式和这一个数据库的表名；不能排除其他名称的表中有相关列，也不能推断数据库其他内容或其他 profiler 文件中不存在 binary/tiling 证据。结果未能把历史 498.491 µs 任务绑定到 stage220 的源码对象、编译产物、tiling key 或实际运行分支；该绑定仍为 UNKNOWN，也不能以当前源码的静态逻辑解释此任务耗时。


## 候选筛选发生在 Cube QK 之后：提前跳过的条件

对现存 arch22 源码沿输入到计算的静态追踪：mode 2 将候选块输入交给 Vector（`kernel_arch22.h:504–516`）；`CalcS2LoopParams` 依据实际长度、mask 与分片确定连续 S2 tile 范围（`:540–565`），`ProcessMain` 逐 tile 遍历并在 `ProcessBaseBlock` 对每个有效 tile 调用 Cube `ComputeMm1`（`:701–815`）。Cube 进一步遍历 S2 子块、装入 Key 并执行 QK `Mmad`（`service_cube_arch22.h:201–227,254–297,519–536`），该调用链未以候选块决定是否执行。之后 Vector 读取 score 和 scale、相乘，才加载/应用候选掩码、降级分数，再排序归并（`service_vector_arch22.h:448–523,810–888`）。因此，这一静态路径存在**先算候选外位置、再降级**的结构；有效长度、causal mask 和 core 分片本身仍可能限定计算区域，不等同计算所有物理输入。

研究让 Cube 提前跳过候选外数据时，不能只检查候选块数或候选 token 数达到 K。反例是候选内可达位置不足 `sparseCount` 时，源码允许候选外可达位置以降级 score 补入并保留真实索引（`:846–847,875–879`），直接丢弃会改变结果。即使候选数不少于 K，严格等价仍须针对每个相关行证明原实现实际浮点结果中候选第 K 名**严格优于**所有省略位置的降级结果；若发生相等则必须完整保持排序与归并的 tie 规则，并保留所有可见输出、偏移和 sentinel。原实现的降级是依次计算 `score + (NEG_HUGE - score) * isOut`（`:850–857`），不能不经有限值范围、舍入和特殊值论证就把它当成精确常量。多行共用 Cube tile，候选可按行不同；只有对该 tile 的全部相关行都满足相应排除条件，才有可能整体跳过该 tile，否则需要细化粒度。还须保留 causal 可达长度、尾块与 LD 跨核归并，以及跳过 tile 后首尾状态、双缓冲和 cross-core flag 的安全推进。

上述是源码分析和有条件的等价要求，不证明当前接口支持跳过、候选数量充足、实际条件成立或任何节省；历史调用的二进制、mode 和动态候选尚未绑定。mode 1 要生成候选块、mode 3 关闭候选机制，不能套用 mode 2 的前提。

## stage225：候选 NodeAttrInfo 解析文件的限定核查

对 stage224 命中的四个固定文件读取并核对 SHA-256，结果 `agents/diagnosis/qli-attr-parser-files.json` SHA-256 `d36c5e3aaf95eeaa67d099048af7a8aba2458129a13cef0b647a7bbc3430337b`，查询脚本 `agents/diagnosis/query_qli_attr_parser_files.py` SHA-256 `4a391150a128da54e66c322632ed775f1c58ecca881a16febcbfd90acafe2350`；stage225 exit 0、`model_verified=true`。当前版本 `node_attr_info_bean.py:25–29,38–43` 从结构体取 node_id 和属性信息 hash_id；`node_attr_info_parser.py:41–49,64–73,87–96` 解析对应数据，转换结构类型和 node 标识并将 hash_id 转为字符串；`node_attr_info_model.py:21–35` 负责写入其数据库。检查的四个文件未给出 `attr_N` 字符串的生成、排列规则或其与 host OpDef 注册顺序的契约，也未证明这条解析路径就是历史 `COMPUTE_TASK_INFO.attrInfo` 的生成路径。

这次只排查四个文件；不能据此排除其他实现、二进制或文档中存在契约。当前 task 的 `attr_8:2` 依然不能确认实际 candidate mode，原有历史源码与二进制绑定缺口也保持不变。没有使用 NPU。

## mode 2：不预先计算候选外 score 的一个更强条件性方案

在当前 arch22 源码中，常量 `NEG_HUGE_F32` 的位型是 `0xF14A3E31`，按 IEEE 754 binary32 解码为约 **-1.0014585084909675e30**；代码旁的 `-1e30` 是近似注释。上下相邻浮点数间距均为 `2^76`，半个间距为 `2^75`。在以下**必须逐项证明**的条件下，mode 2 候选外位置的原算术结果可不读取该位置 score 而确定：

1. `Mul(score,key_scale)` 后的原始 FP32 结果 `x` 有限，且严格满足 `|x| < 2^75`；这项是对每个需要省略计算的位置的真实执行值的界，不能从张量 dtype 自动推定，也不因历史张量形状而成立。
2. 设备对源码中 `Sub`、`Mul`、`Add` 分别使用 IEEE binary32 的 round-to-nearest，并无会改变本计算结果的融合或其他精度/舍入行为；当前源码本身尚未证明这些设备语义。
3. `isOut` 对该位置确为 1，`NEG_HUGE_F32` 按上述位型解释。

在上述条件下，令 H 为该常量：先有 `round32(H-x)=H`，再有 `round32(H + x)=H`，故 `x + ((H - x) * 1)` 的结果按 FP32 位型精确为 H。这里两次取整都必须成立；上界取严格小于以避免中点 tie-to-even 的讨论。利用这一事实，可以研究**不删候选外元素，仅省掉它们的 QK、缩放并在对应原位置补写 H**：仍按原位置号和原有排序/合并拓扑处理，所以即使候选不足 K，候选外元素依然可作为填充，tie、索引、-1 padding 和 output offset 都不必改变。和“直接剔除候选外”相比，这个候选不要求事先证明候选内至少 K 个；但仍须验证补写按原行为生成全部记录及排序输入，设备可访问候选信息并且跳过 Cube 工作的调度不破坏跨核旗标、缓冲复用、LD、尾块和其他输出。整个 Cube tile 仅当对所有被省略的相关行与位置均已确定候选外且满足数值界时才能整体跳过；部分行/位置不满足时仍需保留它们的计算或证明更细粒度实现。

反过来，仅有“候选外降到 H”的注释不能证明任意输入都满足：本机 CPU 用 NumPy 按分离的 binary32 舍入复算，`x=1e38` 得到候选外结果 `0`，而非 H；`x=+inf` 时 `H-x` 为 `-inf`，候选内同样会遇到 `(-inf)*0`，产生 NaN。它们是**条件的 IEEE 算术反例**，不是对 910C 实际指令舍入、模型中出现这些输入或历史任务实际结果的断言。对于无穷、NaN、溢出、排序 NaN 规则或不能证明 `|x|<2^75` 的情形，此方案不能套用上述证明。若要直接丢弃候选外位置而非补写 H，还需确保每行至少 K 个实际优先于 H 的候选，并保持 tie 和跨核归并等语义；不能从候选 block 的物理容量推断该条件。

`op_host/quant_lightning_indexer_v2_tiling.cpp:411–413` 显示当前 DAV_2201 分支只接受 `return_value=false`，但 OpDef 仍声明 `sparse_values` 和可选 candidate 输出；候选设计仍需维持所有可观察输出、ABI、地址和生命周期，不能只因不存在 score 返回就擅自改变输出约定。该源码与历史 binary 的绑定、实际数值范围和收益仍未知。

## 内部 2048 与实际 K 的条件性收紧、同步审查

当前主 score 路径每个 tile 排序后以 `BASE_TOPK=2048` 合并并保留中间结果，最终普通输出仅写 `sparseCount` 个索引；LD 路径还以 2048 条结果作为每核归并输入（`service_vector_arch22.h:885–917, 1003–1102`）。如果独立证实实际 `sparseCount=K<2048`、每个中间片段的比较与最终比较采用同一确定全序，且保留每个片段前 K 项，则任一丢弃项已有同片段 K 项排在它前面，无法进入全局前 K；多片段重复应用这一论证也成立。这提供按实际 K 收紧 token score 中间归并的**条件性候选**，不是历史 512 输出形状对所有调用 `sparseCount` 的证明。尤其应证明实际硬件排序的 tie、NaN、无效项和归并顺序与既定全序兼容，或证明 relevant 排名边界上无歧义；否则缩短列表可能改变同分结果。还需重新核对硬件 `MrgSort` 列表长度及对齐约束、LD workspace 布局、跨核归并、零行、尾块、offset、sentinel 和输出稳定地址。mode 1 的 `ProcessCandBlockTopk` 在主 token 排序前独立按 block 分数维护 `globalBlockTopkUb_` 并输出 candidate block；其 `candidateTopkBlocks` 是独立数量，不可随 token 的 K 一起裁剪。实际设备工作量、可支持的长度和收益未知。

同一快照 `quant_lightning_indexer_v2_vector.h:152–174` 的主 score `MergeSort` 使用 UB→UB `DataCopy` 后跟 `PipeBarrier<PIPE_V>`；`:184–190` 的候选专用实现注释称若 DataCopy 走 MTE，则该 barrier 不构成其 fence，并报告多 tile 竞态，改用纯 V 的位精确 int32 回拷。主路径 `service_vector_arch22.h:885–889` 确实调用原 `MergeSort`；但注释中的条件是否适用于该调用的实际 DataCopy 路径、是否有其他同步、以及历史二进制是否包含这段代码均未建立。此处只能列为应核实的同步语义问题，不能宣称观察到 bug、归因于该问题或建议未经验证的替换。LD 也有类似的 DataCopy 与 PIPE_V 排列，应独立按各路径检查。

CPU 反例复现（仅检查普通 binary32 数学模型，不使用 NPU）：

```sh
python3 - <<'PY'
import numpy as np
H = np.array(0xF14A3E31, dtype=np.uint32).view(np.float32)
x = np.float32(1e38)
pen = np.subtract(H, x, dtype=np.float32)
y = np.add(x, np.multiply(pen, np.float32(1), dtype=np.float32), dtype=np.float32)
print('H:', H, 'masked score:', y)
PY
```

### 从半精度中间量推导 `|x| < 2^75` 的条件边界

当前 arch22 `service_vector_arch22.h:730–746` 把 `w` 与 `q_descale` 作为 half 读入并做 half `Mul`；`service_cube_arch22.h:502–515,558–579` 用两个 half 输入做 `gSize` 项乘加，结果以 float 输出；vector `:838–845` 将 half `k_descale` 转 float 后与 score 相乘。当前 host 对该路径规定 `headDim=128`，`gSize=32` 或 `64`（`quant_lightning_indexer_v2_tiling.cpp:908–935,990–1011`）。在额外假设下可从这些事实推出前节所需界：

- 假设**进入第二个矩阵乘的两个 half 中间量的每一项均有限**，而非仅假设初始 `w`、`q_descale` 有限；`k_descale` 也有限。每个有限 half 的绝对值最大为 65504，严格小于 `2^16`，每项实数乘积绝对值小于 `2^32`。
- 假设 Cube half×half 的乘法和至多 64 项的 FP32 累加按通常 binary32 round-to-nearest 进行（融合乘加或分别乘加都可，只要总相对误差满足下述界），无其他放大变换，后续 Fixpipe 以同样数值意义传递 float；key scale 转 float 和最后乘法也按此语义。对每项至多一次乘法舍入和一次累加舍入，取保守因子 `(1 + 2^-23)^128 < 2`，得 score 绝对值小于 `64 × 2^32 × 2 = 2^39`。再乘有限 half key scale 并计入一次宽松舍入因子 2，得到 `|x| < 2^56 < 2^75`。该上界刻意宽松，仅用于证明前节排除条件，不是速率界，也不代表实际数据规模。
- 若中间出现零、符号变化或次正规数，证明仍需设备实现满足所述绝对上界；不能把常规 IEEE 运算自动视为已验证的设备契约。上述每一个算术和传递条件均需设备接口/指令资料或能绑定的证据支持。

**初始输入均有限并不足以推出中间 half 有限**：在常规 IEEE half 算术中，`65504 × 65504` 超出 half 有限范围，可能产生 `+inf`；再参与乘加就可能得到无穷或 NaN。`q/k` 为 INT8、headDim 128，整数 dot 的绝对值按最大项计算不超过 `128 × 128 × 128 = 2^21`，但 `FixpSToL1` 经 `DEQF16`、ReLU 和预量化标志 `0x3a800000` 才形成 half：未先核实该 flag 的设备契约，不能据标志位直接证明该转换有限。即使日后证实转换有限，独立的 `w × q_descale` half 结果仍须受限。历史输入和中间值、该历史 kernel 二进制、实际指令舍入均未知，故本节是可核验的条件性推导，不是已证明历史 workload 可跳过计算。

### arch22 选择路径与 2047 比较证书的适用性复核

离线复核保存的 QLI V2 arch22 快照：`quant_lightning_indexer_v2_vector.h:86–139,152–169` 的 `SortAll` 先调用 `AscendC::Sort32`，再调用多路 `MrgSort<float>`；主分数路径 `quant_lightning_indexer_v2_service_vector_arch22.h:776–889` 生成、缩放和掩蔽分数，再对每块排序并合并。候选模式 2 路径 `:433–523` 对候选 **block ID** 排序、以 tile 边界做 `CountGE` 二分并结合向量算术生成 mask；该阈值查找针对 block ID，并不是证明分数选择使用了桶排序。就可见源码而言，分数路径调用排序和归并接口，候选处理是排序、阈值查找与算术混合；未发现足以证明分数路径具体采用数值桶或 radix 的证据。单凭高级接口名称也不能断定硬件内部如何实现这些指令，或把一次向量排序换算成二元比较次数、指令次数或周期。

先前的 **2047 次二元比较**只是以下抽象域的条件证书：2048 个互异候选、需要精确选出非空且非全集的 top-k、允许任意相对次序、无预先可用的顺序摘要，并且算法只能通过候选之间的二元大小比较获取顺序信息。为确定所有入选项均不小于所有未入选项，比较关系必须连接所有候选，因此至少有 2048−1 条比较边。数值阈值、分桶、从输入编码直接推断顺序、预存摘要等超出此模型，证书不能约束它们；有大量相同值、受限输入域和已知结构时也不能直接套用。

**该证书不适用于任何已保存的 169 运行作为已证实的数值界。** `BASE_TOPK=2048` 是内部容量，块内排序长度还受实际长度、对齐和候选 mask 影响；它不证明运行有 2048 个有效且互异的待比较分数，也不证明所有排列可由该模型输入产生。已保存的 profiling 任务没有完成快照源码与执行二进制、tiling、候选模式、运行时有效候选数和输入域的绑定；`Sort32`/`MrgSort` 的内部实现与可计量容量亦未获证明。即便以后绑定源码，若任务允许融合、重算、改变选择算法或共享摘要，仍需独立证明这些算法均受同一比较模型限制，才可用于宽域上界。这里不将 2047 当作实际算子耗时、指令数或吞吐界。

本次复核文件 SHA-256：`quant_lightning_indexer_v2_vector.h` 为 `3fcfa7a0f8ace8f4197e4e295992e0c93ba167e9421bcb9cea55f42a3c8ced29`；`quant_lightning_indexer_v2_service_vector_arch22.h` 为 `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8`；原快照来源记录 `qli-v2-source-provenance.json` 为 `061246e242618916398367fd1f742392902c17eb028a02e3bb44c5574f8dac6c`。本节只读取既存文件，没有访问设备。

## 分 head 裁剪不能替代聚合后按 key 的 TopK

前文允许在**完整 score 已形成后**按 key 分块求局部 TopK 并合并，但不能把这条结论移到多 head score 聚合之前。已存 `agents/fusion/frozen-source/csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_cube_arch22.h` SHA-256 `2b586bb8c894fbb0ea6887c7d0bd69332f2ab54570294ee6f407ecf346296352`：`:201–230,519–554` 对 query/key 做 QK，再以 `DEQF16` 和 `reluPre=1` 经 Fixpipe 将中间 score 送 L1；`:179–195,500–515` 使用 head weights 做后续矩阵计算，按 key 写聚合结果。F12 的 Vector 路再结合 key scale、候选条件及局部排序。这里列的是保存源码的调用和配置字段；`DEQF16` 标志、设备舍入、mask、实际权重和历史 binary 的具体合同仍未绑定，不能用源码名自动推定精确指令效果。

一个与历史形状规模相容、但**纯构造且只基于所声明数值假设**的反例：对一条 query 的两个 head，INT8 向量分别取 `[127,0,0]`、`[0,127,0]`，其余 30 个 head 权重取 0；各向量可补零至 128 维。构造 1025 个有效 INT8 key：512 个 A=`[10,0,127]`、1 个 B=`[6,6,127]`、512 个 C=`[0,10,127]`，亦可补零到 128 维。假设两个 head 的权重、query scale 和全部 key scale 都为 1、无 mask，且 QK 到 ReLU 后进入 head 加权的转换在这组有限值上等于同一个正比例 `c` 并精确保序，则 A、B、C 的 head 分数依次为 `(1270c,0)`、`(762c,762c)`、`(0,1270c)`，聚合结果分别是 `1270c`、`1524c`、`1270c`。

B 因此是**全局最大项**，也应在完整 score 的 top512 内；但每 head 先只取各自 top512 时，第一个 head 的 512 个 A、第二个 head 的 512 个 C 完全占满候选，两者并集不含 B。A/C 内部如何处理并列不影响这个结论。以下只读 CPU 整数模型可复算：

```python
K = 512
A, B, C = (10,0,127), (6,6,127), (0,10,127)
keys = [A]*K + [B] + [C]*K
queries = [(127,0,0), (0,127,0)]
head_scores = [[max(0, sum(q*x for q,x in zip(head,key)))
                for key in keys] for head in queries]
candidates = set()
for scores in head_scores:
    candidates.update(sorted(range(len(keys)), key=lambda i: (-scores[i], i))[:K])
full_scores = [sum(scores[i] for scores in head_scores) for i in range(len(keys))]
winner = max(range(len(keys)), key=lambda i: full_scores[i])
assert (winner, full_scores[winner], len(candidates)) == (512, 1524, 1024)
assert winner not in candidates
```

该例证明的是：即使权重为正、ReLU 后各贡献非负，在给定条件下“每 head 先保留 top512，再合并”不是普遍正确的 head 聚合后 top512。要提前剪枝，需要证明被保留候选覆盖全局赢家、或给每个被丢 key 保守的跨 head 剩余贡献上界；也可以使用其他经证明语义等价的方案，不能据此认定任何剪枝都不可能。两 head 的全部必要 QK、weights/query scale、key scale 和候选可见域仍须满足原顺序与类型；tile 完成事件、两路 AIV 消费、workspace 生命周期和输出消费者不由数学公式消失。

已有 EP8/L24/run2 的 QLI task1275 输入含 INT8 `[3,32,128]` query、INT8 `[17307,128,1,128]` cache、两个 FP16 `[3,32]`，输出 INT32 `[3,1,512]`；详细事件及来源在 `agents/fusion/f12-layer24-run2-evidence.json`（SHA-256 `4011fa057341ccb02b1436d6ee4beec66dd54124415664b3ad47a38e640591b2`）。这些形状与示例的 head 数、128 宽和 K=512 规模相容，却不绑定有效 key 数、head 数值、mode、scale、source/binary 或数据生成过程；不能据此声称该例真实可达、在线索引有问题或节省任何计算。无新设备工作或推理实现修改。


## 2026-10-30：公开接口对 Cube→Vector 直达方案的约束

[CANN 9.1 DataCopyL1ToUB](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/DataCopyL1ToUB.md) 和 [Ascend C Tensor Copy L0C→UB](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_TensorAPI/cube_compute_store/Copy_L0CToUB.html) 产品表均列 Atlas A3/A2 不支持，不能套用 950 的上述直达 API。接口限制只针对这些公开 API 和版本，不证明所有私有或其他路径不存在，也不证明历史 kernel 的二进制身份。

[CANN 9.1 CrossCoreSetFlag](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0273.html) 支持 A3 模式 2：AIC 发布后两个 AIV 可分别继续，AIC 要等待两个 AIV 都发布；模式 4 的单 AIV 同步标为 950 专用。冻结 arch22 QLI 源码 `quant_lightning_indexer_v2_kernel_arch22.h:793–815` 使用 mode 2 信号及等待，相容但未绑定历史 binary。同步能力不等于跨核数据可直达。下一步应优先研究保留 GM 中转的分块和数据量调整，考察 L2 缓存和流水，而非假定可用上述直达 API；不改变 score 舍入、两个消费者的可见性和排序规则。没有实测收益。


## token TopK=512 与 tile/candidate=2048 解耦的条件候选（2026-10-30）

保存源码 service_vector_arch22.h:29-30 固定 BASE_TOPK=2048；887-888 用它作为 token 累加长度，894-898 提取2048索引，916-917只写 sparseCount；926-957及1011-1122在LD中按2048布局处理。vector.h:152-169按两列表完整归并，再回拷mrgDstNum项；172-182的提取repeat与长度成正比。针对确认 sparseCount=512 的分支，令token保留K=512、tile=2048、candidate block K=2048：合并输入4096→2560对，回拷16→4KiB，提取64→16repeat；两行专用token状态若同步缩分配，32→8KiB。完整输入归并和其后拷贝不自动线性定时，不能外推整体收益。

这不是把所有BASE_TOPK替换成512：candidate索引初始化/排序、globalTopkIndice、source输出、共享outQueue仍可能需要2048；LD工作区步长、读写、合并、初始化、输出须统一变更或先限定非LD。保留原同分优先级、候选外有限降级填充、invalid和offset；确定全序下的局部前K性质支持结构候选，尚未验证具体硬件的逐结果等价。

官方接口来源：https://asc.gitcode.com/api/SIMD-API/basic_api/memory_vector_compute/sort_and_merge_ISASI/MrgSort.html ，页面标 master eceb7aa00646、构建2026-10-29且含未提交修改，属于开发预览。其独立elementLengths和两队列validBit=3支持本长度设计的接口层判断；并非9.2安装版编译或性能证据。最小后续验证限于实际sparseCount/LD身份、安装头文件及该长度组合的编译与定向正确性，由后续开发完成。本次无NPU运行、无目标代码修改。
