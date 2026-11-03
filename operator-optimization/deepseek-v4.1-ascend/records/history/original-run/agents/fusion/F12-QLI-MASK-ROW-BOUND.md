# F12 补充：QLI 原生逐行 mask 与后处理可见范围

## 来源与绑定边界

本分析只读取已保存快照，不执行 NPU，不修改实现。`frozen-source/csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h` SHA256 `d35c26e6cb781aebad163a2c85cbd84f68c7c838ebe5ef9f67e6a42777949df6`、`.../quant_lightning_indexer_v2_service_vector_arch22.h` SHA256 `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8`；与本任务保存的 `zcode/operator-ceiling169-v2-220-qli-native-source.evidence.txt` 列出的容器源码快照哈希一致。该快照来自 stage 容器镜像 `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`。这证明快照与上述盘点一致，不证明 profile 时此源编译成已执行二进制，亦不证明具体事件的张量或 native 实参。arch22 是被检查的源码路径；未通过此分析证明历史任务实际使用的模板实例。

参考 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py:254–297` 的非 packed 路径和 `frozen-source/vllm_ascend/ops/triton/prepare_indexer_indices.py:39–43`。前者向 `_C_ascend.npu_quant_lightning_indexer_v3` 传入 `query_start_loc`、`cache_seq_lens`、`cmp_residual`、`mask_mode=3`、压缩率 c，接着将其输出及独立 `positions` 交给 helper。保存事件中有关 V3、V2 的同名统计不足以逐次绑定；见 `INDEXER-RUNTIME-REVIEW.md`。

## 有条件的源码结论

- arch22 `kernel_arch22.h:182,283–314`：tiling 中 sparseMode 为 3 时启用 mask。已传实际 K 长度时还原原长度 L = `seqused_k*c + cmp_residual_k`；Q 长度 Q 对 TND 为相邻 query 起点之差。`dsa_v41.py:902–915,1035–1045` 的非 packed C2 分支从 `common.seq_lens` 计算 quotient 和 remainder，故在该分支及状态正确时 L 是此 `seq_lens`。此处 c=1 对应原长度、c=2 对应 quotient/remainder；该代码并不能保证 `positions` 内容。
- `service_vector_arch22.h:775–825,863–888`：在 mask 开启、L >= Q、实数行 0 <= i < Q、可见项足以构成该行的情况下，源码逐行计算 QLI 可见长度 `V = floor((L - Q + i + 1) / c)`，逐块只为 `[0,V)` 构造真实索引与 score 后排序；超出部分不能简单当作真实候选。源码在不足位置处有 `-1` 与 `-inf` 初始/填充路径，相关 LD、极值、tie 仍须按实现保留。这里的非负前提保证 C++ 截断除法等同数学 floor；边界和填充分支须单独考虑。
- helper 的长度为 `H = floor((p + 1)/c)`，仅保留 `0 <= selected < H`。因此 **p = L - Q + i 是可见长度一致的充分条件**；更一般只需要两个整数商相等。若一致且 native 全部真实输出均在其 V 内，helper 的可见性过滤不会剔除真实项（它仍负责排序、-1/长度/输出地址等）；这不自动证明可以更换 TopK 算法、去掉 helper 或重排所有流程。
- 目前数据尚未逐行绑定历史调用中的 p、L、Q、i。`dsa_v41.py:902–915` 分别复制 common.seq_lens 和 common.positions，未在这里校验等式。CP 路径 `context_parallel/dsa_v41_cp.py:118–150` 则切分 query 与位置、对序列长度减后缀；该代码旨在构造局部长度，但在未核实上游数据约束和 runtime 值时不能据此证明所有历史行等式。若 L、Q、i、p 关系不满足，V 与 H 可以不同；也未证实此情况实际出现。

## 具体、非运行观测的反例

考虑接口层的非 packed、candidate_mode=3、c=2、topk=512，一行 Q=1；设 L=1028，即 seqused_k=514、residual=0。QLI mask 给 V=514；假设独立传入 p=1023，helper 给 H=512。假设原 score 严格满足第 513 项最高、0 至 510 次之、第 511 项次之，其他更低。先选后过滤会选 513 和 0 至 510，然后将 513 换成 -1；先按 H 过滤再选 512 项会选 0 至 511。两者不同。该例是源码公式和 score 顺序下的条件性数学例子，不声称这些元数据组合、score 或索引出现在真实负载；不同于前文 K=1 的一般集合反例，它固定了已有被调查 shape 中的 K=512 及 c=2，但不证明任何特定任务值。

即使证得 p 对每行匹配，还需保留 QLI candidate mode 的打分/块副输出、有限分数与 ties 的排序规则、-1 槽、helper 的位置序排序、原有输出/长度缓冲区和消费者时序；所以可以提出融合候选，不能直接宣称交换 TopK、删除过滤或预估收益。下一步若只读资料中可找到 runtime 输入/metadata 与来源二进制绑定，可验证每行 `V==H` 和 native 输出范围；若没有，则这是明确的数据缺口，而非请求重新运行 NPU 的授权。
