# Indexer：冻结 helper 后的 F09 / F12 边界复核

## 证据及范围

ZCode stage `operator-ceiling169-v2-087-indexer-helpers` exit 0、verified `deepseek-flash`。`indexer-helper-provenance.json` 登记两个容器相同镜像、目标源码路径无覆盖挂载、两个文件的双方字节及 SHA 一致；本次快照不触碰 NPU。以下是当前镜像源码条件推论，不是历史某一轮的 branch、张量身份或执行器行为的观测；在线更新、进程实际引用对象等仍未知。证据为 `model.py`、`dsa_v41.py`、`indexer.py` 和 `frozen-source/vllm_ascend/ops/triton/` 中两个 helper 的固定行号；可执行核验见本目录 `verify_indexer_helper_sources.py`。

## F12：当前源码正常非 packed 分支的索引发布存在自复制调用

1. `dsa_v41.py:567-585,602` 仅在 long-context 的 index source 调用 `select_projected`，并给出 `indices_output = shared.topk_indices[:hidden_states.shape[0]]`。
2. `indexer.py:233-254,275-297` 中 `self.packed_cache_ops is None`、query 非空且 cache 非空时，调用 QLI，再把 `indices_output` 传给 `prepare_indexer_indices`。helper 第 80–93 行验证形状、dtype、设备、连续性后令 `output = indices_output`，第 149 行直接返回同一 Python 对象；其他 `indices_output` 分配不是该调用分支。在成功返回的前提下 output 的行数等于 selected 的行数，否则 helper 会报错。
3. `dsa_v41.py:604-608` 在 selected 的第二维非零、`not getattr(attn, "uses_a5_packed_cache", False)` 时，重新切 `shared.topk_indices[:selected.shape[0]]` 并 `copy_(selected)`。正常 Tensor 切片语义及上面的行数相等条件下，两个 view 指向同一段存储，shape 和位置一致：**源码调用是按值把同一位置复制回自身**。这并不等同于已经发生了一次 NPU memcpy；底层 PyTorch/NPU 可能将其短路，是否产生任务或占用时间 UNKNOWN。
4. 空 query / 空 cache 在 `indexer.py:208-231` 有单独早返回：有 `indices_output` 时也直接返回它，空 cache 先执行 fill/length 清零。不得因“自复制”删除这些必需的 fill、length 或候选状态操作，也不得把早返回伪记成 QLI 或 fixup 已执行。packed 分支由 `run_a5_indexer` 决定返回值，不在本结论范围。

这是一项值得后续开发 Agent 检查的**冗余调用候选**，并非新融合或已经证明可删。要判断变更是否行为等价，还要审计相关 tensor 版本计数、图捕获、instrumentation、可能的依赖/同步副作用和所有调用路径；实际 branch、编译器降解与耗时须用同一调用的现有证据或专门跟踪确定。不能把 `2*indices bytes` 或一个 kernel 时间计成节省量，也不能据此解释 Indexer 的跨轮耗时差异。

## F12：fixup 本身已经含多种组合，不可重复算作待融合

冻结 `prepare_indexer_indices.py:39-56` 的同一个 Triton kernel 对每行做可见性过滤、编码、排序、恢复 sentinel、写最终索引；在 `lengths_output` 存在且 `num_rows * max(next_power_of_2(topk), 32) <= 1024*512` 时同时写有效长度（第 109–111 行）。如果超过阈值才在第 141–148 行另做 `output >= 0` 及 sum 写长度。冻结 `evidence/model-config.json` 的 `text_config.index_topk=512`，若该配置确实绑定此调用并且 helper 所见 TOPK 为 512，则 `max(next_power_of_2(topk), 32)=512`，有长度输出时至多 1024 行走同一 kernel 写长度；超过 1024 行走额外求和。实际每次行数、是否有长度输出、对象配置及该路径是否命中仍未观察，不能据此断言基线走哪条。 因此 F12 必须区分已组合的过滤/排序/输出和条件 fallback，不能重复声称其过滤和排序尚未融合，也不能默认 fallback 出现。待核实的仍是 QLI → fixup 的真正数据边界、当前 shape、历史 graph lowering 与精度/顺序要求；QLI 的 TopK、positions、candidate 输出、空 cache 行为及跨层共享消费者均不能省略。

## F09：Q 投影、RoPE、量化的数值与状态要求

在非 packed 模式，`indexer.py:167-179` 先 Q 投影、对 query 部分维度原地 RoPE，再进入 `quantize_indexer_query`；`dsa_v41.py:401-409,421-430,582-592` 中 quantize 在辅助流上执行，并由事件连接到后续 QLI。冻结 helper 的数值顺序是：每 head 在 FP32 求最大绝对值，除以 127，**先转成 FP16 再转回 FP32**，再取至少 `2^-24`，用该舍入后的 scale 做除法、`nearbyint`、截断到 [-127,127]，输出 INT8 和 FP16 scale（helper 第 23–37、40–69 行）。任何融合都必须保持这个顺序/舍入及零 head 处理，不能拿一般动态量化接口直接替代，也不能把主 Q 的 W8A8 scale 与此 per-head scale 混为一谈。

`select_projected` 非 packed 正常路径在预先有 quantized_query 时使用 query 的 shape，而 QLI 调用传的是量化结果和 scale；但 helper 的 `.contiguous()`、RoPE 原地操作、其他调用路径、空 cache、stream event 和 `prepared.query` 生命周期都须保留或证明替代。投影 → RoPE → 量化可以是条件组合边界，不保证可以消除中间物化或节省 HBM；不能凭源码推断跨 Cube/Vector 的实际效率。

未修改模型、算子或 Task 状态。具体 trace 的 tensor/storage identity、self-copy 的底层实现、F09 的硬件融合支持和性能变化仍 UNKNOWN。

## 已有设备样本对应的一手文献复析（2026-10-30）

本节收敛已登记候选，不代表已经测量性能收益；下列 GPU 论文的具体性能结果和硬件指令不能直接迁移到 910C。当前仅复用保存资料、文献和官方文档，未运行 NPU。样本窗口的计时、不同类别与整请求不可相加。

1. **FlashAttention，NeurIPS 2022。** [原论文](https://proceedings.neurips.cc/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) 用分块和在线归一化，减少大注意力中间矩阵在 HBM 和片上存储之间的往返。关联本任务 F13 已有 `SparseFlashMla` 复合边界：28 个选定 Indexer→整理→attention 窗中 Indexer 占比中位 82.264%，该范围不能推出 attention 本身是首要瓶颈。不能将整套 dense FlashAttention 的 IO 省量重复记在已经融合的 sparse kernel 上；还须先绑定实际稀疏索引、KV 页、mask/sink、缓存及中间输出，才能判断是否仍有可消去的 GM 往返。

2. **FlashAttention-3，NeurIPS 2024。** [原论文](https://papers.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf) 利用跨迭代缓冲，在不同块之间交错矩阵、softmax 和搬运；这是跨块流水，不是取消同一块的生产者依赖。对应 F21 MoE dispatch→GMM1→激活量化→GMM2：保存代码已按专家组重叠 dispatch 与 GMM1，而 AIV 激活受 dispatch 循环及同步限制。可研究下批 dispatch 与前批激活交错，待证明 AIC/AIV 资源、共享 UB、scale、workspace 别名和 event；不能假设原实现完全没有重叠，也不能将 Hopper TMA/WGMMA 的结果套到 Ascend。

3. **昇腾 Ascend C Matmul 异步，CANN 社区版 8.1.RC1 beta1。** [官方接口说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/opdevg/Ascendcopdevg/atlas_ascendc_10_0077.html) 对 MIX 场景列出 Matmul Iterate/IterateAll 的同步及异步模式，纯 Cube 场景只支持同步模式。该资料限定重排条件，不能据此断言实际 910C、历史 CANN 或此 MC2 二进制支持某个异步变体；应绑定设备版本、所用 API、每个队列的完成事件及 workspace，减少不必要等待必须与同步安全共同证明。

4. **RadiK，ICS 2024，arXiv 2025-01-24 版本。** [论文](https://arxiv.org/abs/2501.14336) 以 radix 选择和批处理资源调度降低大规模 top-k 对片上列表容量的压力，并讨论数据分布、直方图和中间输出开销。当前 QLI 的 tile 长至 2048、候选最终输出形状 512，源代码逐 tile `SortAll` 再保留全局 2048，最终只写 `sparseCount`。应先核定实际 sparseCount、有效长度、最终是否要求排序和 tie/无效值语义；相比先改成新选择算法，保留 tile、仅缩短 token 累加列表是更收敛的条件候选。论文部分量化实验明确不计输出排序，且主要展示不同规模的 NVIDIA GPU，不能把速度或不排序输出直接用于本路径。

5. **RTop-K，ICLR 2025。** [原论文](https://arxiv.org/abs/2409.00822) 以阈值搜索做逐行选择，可选择近似 early stopping，目标应用不要求最终 k 个元素有序。与上述 QLI/512 同属按行选择，但本实现还包括 TopK 后索引整理、跨层候选复用、有限负惩罚与 sentinel：若下游依赖顺序、tie 或逐项等价，选集合和允许近似都不够。故它是候选生成方法，不是替代当前有序精确语义的现成证明；必要代价包括阈值查找、重复扫描、边界同分处理和排序恢复。

6. **CANN MrgSort，商用版 8.0.0。** [官方 API](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0232.html) 将 `elementLengths` 定义为各输入序列的 8 字节结构个数，理论范围 0–4095 且受 UB 容量等限制；这只支持检查混合 2048 与 512 在数值范围内，并不单独证明本设备、实际版本、对齐和异长组合合法。保存 `vector.h:152–169` 中两路分别为 tile 长与累加长，输出 scratch 容纳二者总和，且只回拷累加长；满 tile 条件下输入从 4096 对降至 2560 对、回拷从 2048 对降至 512 对，最终提取可从 2048 降至 512；**不能因此说 sort tile 不再执行或内核实现只输出 512 对**。candidate block 单独保持 2048，LD 工作区/四路归并、tie、sentinel、异步回拷与消费者须保持。

7. **MegaBlocks，MLSys 2023。** [论文](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) 将不同专家的动态 token 分布表示为块稀疏矩阵运算，以减少固定容量 padding 和不丢 token 的代价。对应主模型 MC2 在三个选定 40 层主流 span 分别占 35.89%/35.88%/36.30%，以及草稿 162 窗 MC2 时间和占窗口时间和 48.47%。当前 MC2 早已是分组矩阵及通算复合路径；实际每专家 token 数、tile 及 padding 未绑定，不能据论文假定已经存在固定容量浪费或能直接改成 GPU block-sparse 提速。

8. **Tutel，MLSys 2023。** [论文](https://proceedings.mlsys.org/paper_files/paper/2023/file/5616d34cf8ff73942cfd5aa922842556-Paper-mlsys2023.pdf) 针对动态专家工作量，研究可切换并行和 dispatch、专家计算的分块通信流水。对应 F21 和 DSpark 草稿 162 窗中位 723.376 µs；只能在已存在的通信计算交叠基础上寻找进一步重叠，并考虑各 rank collective 顺序、数据就绪、workspace 活期、计算/网络共享资源及流水首尾。论文训练 GPU 场景的加速和硬件拓扑不能移植为本次数值；草稿每窗 MC2 中位 181.186 µs 与窗口中位不是同一窗口配对。

9. **vLLM 作者 Berkeley 技术报告 2025，算术强度模型。** [报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/Archive/EECS-2025-192.pdf) 指出在假设 BF16 权重每批只读一次时，矩阵乘算术强度随参与的有效输入行数增长。关联 Engram 形状相容但身份未绑定的 `[3,6144] × [25600,6144]ᵀ`，其 96 条任务中位 275.765 µs；矩阵权重逻辑大小为 `6144×25600×2 = 314,572,800 B`，单次逻辑工作 471,859,200 MAC。针对共同权重、同时就绪的多个输入行可考虑分批复用，代价是排队/TTFT、快照和 buffer 生命周期；不能假设两处 Engram 候选权重相同或从 HBM 读取，更不能用逻辑权重字节数除以耗时当实测 HBM 带宽。

数据依据：`publication-draft/README.md`、`publication-draft/FUSION-INVENTORY.md`、`agents/fusion/F21-MC2-DISPATCH-ACTIVATION-PIPELINE.md`、`agents/diagnosis/ENGRAM-MATMUL-DISPATCH.md`、`agents/diagnosis/module-priority-evidence.md:298–307`；历史 CUDA/GPU 论文、不同 CANN 版本及该机实际运行边界不同，留待逐项身份与接口核对。
