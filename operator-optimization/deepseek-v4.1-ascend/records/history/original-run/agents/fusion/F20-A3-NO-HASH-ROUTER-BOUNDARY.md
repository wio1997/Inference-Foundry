# F20：无表文本路由的源码路径及重组边界

## 结论与前提

在保存配置按当前快照正常构造、没有额外属性注入时，`num_hash_layers` 补为 0；若同一实例采用对应 A3、without-group 源码路径，无表的文本行不是按 token 查表选六个专家，而是按**全部专家的动态分数与文本 correction bias**选择。此前“先查表，仅计算选中六列”的候选不适用于这个条件路径。以上是源码条件推导，非历史运行分支的直接证明；数据和模型权重未读取，未启动 NPU。

该模型中现有 Python router 已把 score、bias、选专家和选中分数归一化封装在一次 `moe_gating_top_k_hash` 调用中；另一个 `dispatch_ffn_combine` 已封装 dispatch、专家 FFN 和 combine。研究跨 gate/router 的重组、或者分块输出交接，不能重复把它们的内部操作记成尚未融合的独立收益。

## 条件路径与源码证据

1. [静态配置溯源](../diagnosis/module-priority-evidence.md) stage162：当前冻结配置 root/text 均未提供 `num_hash_layers`，在该配置类、基类、配置归一化及无运行时注入条件下最终设为 0。`model.py:210–234` 将 `tid2eid` 设 None，为非 hash 层创建 `e_score_correction_bias`；还依据 vision 层条件创建 `bias_vl`。模型对象在历史运行时的字段值未捕获。模型源 SHA-256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；配置 SHA-256 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`。
2. [Python router](frozen-source/vllm_ascend/ops/fused_moe/router/fused_topk_router.py) `:156–223` 在 `sqrtsoftplus` 且满足本处 A3 capability 等条件时调用复合 op，`tid2eid=None` 并不妨碍调用。`bias_vl` 存在时即使无表仍准备 input IDs，选择 text/image bias 必须保持 input ID 与路由行对齐；无法根据 `tid2eid=None` 删除此输入准备。`out_flag=False`、`norm_type=2`、`eps=1e-20` 是该源码调用的静态参数，而非历史对象实测。
3. [host tiling](frozen-source/csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_tiling.cpp) `:202–212,309–310` 仅在 input IDs 和 tid2eid 的可选 shape 都存在时置 `hashFlag=1`；`:239–246,617–646` 要求 softplus 的 group_count=1，并在普通非 950 tiling 选 without-group 家族。若保存源码与实际加载实现相符且 table 真为 None，`hashFlag=0`。历史 tiling key、二进制及实参未绑定。
4. [without-group](frozen-source/csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_without_group.h) `:129–140,144–203,244–269,272–310,402–420` 每行先加载 logits，再以 `Exp → +1 → Ln → Sqrt` 得到未加 bias 的 score；文本行加 text bias 形成排序 score；由于无表而走 `SelectTopKExpertIdx` 中的 `Sort`，再从未加 bias 的 score gather 所选项，`ReduceSum + eps` 后除法、缩放并输出 ID 和权重。文本和图像均动态排序，但 bias 不同。原生公式应按源码逐步和 dtype 理解，不能直接替换成回退函数的公式。

kernel SHA-256 `f18d63bda201eb407a3861da0f5189d45d439b6630abf52058a6eaa0420d8fdb`，router SHA-256 `06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838`，二者来源和现存镜像一致性见 [原生 hash 分支](F20-A3-NATIVE-HASH-DEPENDENCY.md)。

## 可研究的非表路径重组

- **gate + 路由按专家块计算：**外部 router 条件下，`model.py:282–297` 先 `F.linear` 形成全专家 logits，再传 experts；内部 router 使用不同路径。可以研究逐 expert tile 输出 logits，同时计算 score 和 bias，只保留每 token 的局部前 K `(排序分数、未加 bias 的 score、ID)`，再准确合并为全局前 K，最后沿用原有归一化和缩放。这仅是候选，并非当前实现、实际硬件可用接口或已证收益。
- 数学排序在**全序、同一 tie 次序**下有 `TopK(union of local TopK) = global TopK`；但实际 NaN、相同值、Sort 稳定性、padding、精度舍入可能不满足简单全序，必须先给出与原 kernel 一致的比较器和输出顺序；GEMM 分块还可能改变归约顺序/精度。不能仅凭 sqrtsoftplus 的实数单调性替换加 bias 后的 score 排序，也不可重算已选 score 而忽略浮点差异。
- 当既无已知的表选定专家、也无可证明排除上界时，一个尚未检查的专家可以通过其 score 加 bias 改变 TopK，不能从先看见的六个专家推出最终专家。此论证基于可独立变化的抽象 score 输入，不证明当前权重对应 logits 可任意独立变化，更不是硬件指令数或严格模型工作量下界；有合法可证界、预处理或特殊结构时可能剪枝。
- 如只需要最终结果、不存在 full logits 的其他消费者，gate/router 重组可能避免将中间 logits 写出再读入；节省的是**有条件的中间张量传输**，不是省去全专家结果的数学贡献，也不等于把 local `xNorm` 变成未发生的 HBM 流量。必须先证明张量的其他消费者、图 buffer 生命周期、内存布局及对应实际任务/流，且要计算新增局部候选、缓冲、精度和资源争用；无带宽或时间数据，不估算提速。
- **token 块到 dispatch：**一个 token 的全局 TopK 只有处理完其全部候选、合并与归一化后才就绪；各 token 彼此之间没有因 TopK 本身产生的全局数学依赖，但 sequence-parallel 对齐、EPLB、共享专家、padding、集体通信顺序及同步可能引入额外约束。现有 `dispatch_ffn_combine` 是批量接口，当前无证据表明允许 tile-ready 逐 token 交接或跨 rank 任意重排；重组的可能性不能当作已支持的通信隐藏。

## 历史参数证据与缺口

历史 target 层窗口按名称见 `MoeGatingTopKHash`；草稿窗口也见该符号。因为该调用有表、无表均同名，名称本身无法确定表参数、图像行、每行 bias 或实际 binary；目标层窗口统计不是 tensor 身份关联。原始本模型 config 未显式指定 `num_hash_layers`，169 同镜像索引无 tid2eid 名称，支持上述静态构造链，但历史加载配置/权重可能不同、表还可零初始化，不能将其作为历史无表证据。见 [router 审计](F20-ROUTER-AND-DISPATCH-REVIEW.md)、[视觉 fallback 复核](F20-VISION-FALLBACK-MASKED-TEXT-ROUTER.md) 与 [module-priority-evidence](../diagnosis/module-priority-evidence.md)。

下一步仅在现有资料中查同一目标 `MoeGatingTopKHash` task 的可选输入形状/slot、op 参数及其与 host/binary 的可靠绑定，并先核实同版本导出器是否保留 None 槽位和位置；若只能得到无名 shape 列表，不能认定某项就是 tid2eid、也不能将缺失项认作 None。仍缺绑定时标记未知，未来要有可信的历史配置对象或逐调用参数/tiling 快照；本阶段禁止为此发起 NPU 运行。
