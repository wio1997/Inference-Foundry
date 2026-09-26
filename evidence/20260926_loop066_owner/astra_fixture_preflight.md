# Astra High：Loop066 owner16 私有 fixture 预检

2026-09-26。只读源码及Run287，未操作NPU/服务，未修改脚本。结论：**建议先做 Compressor 输出/state 的 A/A→compact B→A，而不是第一次就串入 quant/scatter/QLI。** 私有fixture仍应按共享storage克隆，避免为后续阶段制造不真实cache ABI。局部通过不证明永久省写或E2E收益。

## 1. 已确认的ABI，及当前manifest缺口

- `ops/dsa.py:_build_kv_cache:228–282` 从实际layer分别取indexer compressor state、k_cache、scale_cache；910B3为六元tuple。
- `device/device_op.py:813–820` 非A5 unpack直接返回tuple[3]/[4]/[5]，**不复制也不解除alias**；不要照搬:1627的A5七元tuple/full_cache。
- Run287 FP32 state为[34090,2,1,512]、stride[1040,512,512,1]、offset0；FP16 scale为[34090,32,1,1]、stride[2080,1,1,1]、offset2048。两者storage 141,814,400 B，物理page stride均4160 B。
- state每页有效元素覆盖base+4160p+[0,4096)；scale覆盖base+4160p+[4096,4160)。storage_offset以各自dtype元素计：scale2048×2=4096 B，不能错作2048 B。
- 按indexer head_dim=128推导的INT8 key应覆盖每页前4096 B；**必须从本次实际unpack返回的key记录dtype/shape/stride验证，不从FP32 state记录反推代替测量。**
- `diagnostics/target_page_audit.py:56,67` alias名称按data_ptr聚合。Run287“kv_cache.4.0同时列state/k_cache”只证明同起始地址，不能说明它们是同dtype/同shape视图。当前layer2_cache_views不是完整typed operator-input ABI manifest。
- `worker/model_runner_v1.py:_adjust_kv_layout:6100–6128` 明确用raw storage、dtype、as_strided构建不同page布局。共享storage只是容器；state逻辑页和compressed-key逻辑页可由不同blocktable选到不同物理page，不能认为所有state/key有效内容同时同址。

**必须先修正的fixture设计错误：** 分别对state/key/scale做.clone()或.contiguous()，或从state tensor clone后假设page stride仍1040，都会丢失alias/64B间隔。只按data_ptr去重也会漏掉INT8 key view定义。

## 2. 私有fixture如何忠实保留跨dtype alias

1. 从实际unpack取state/key/scale及关联视图，每个分别记录storage identity、nbytes、dtype、shape、stride、storage_offset、data_ptr−storage_ptr。state传入native前还有squeeze(-2)，同时记录squeeze后的真实stride。
2. 以唯一untyped storage为单位，复制完整141,814,400 B一次，包含padding与无效区的原始bit；每个arm拥有独立且等初值的raw backing。为每个原视图按原dtype/offset/shape/stride重建共享视图。不要仅复制状态.tensor.numel()对应的逻辑元素。
3. 保留原page编号和blocktable；第一版不重编号物理page、不压缩page集合。compact只select owner的逻辑request行与输入hidden行。这样历史state读域尚未穷尽时仍有正确私有内容，不让未知页读落到零初始化。
4. 检查所有重建视图的byte extent在backing内；同storage关系及相对byte offset与原始一致，且私有backing与原服务、另一arm、输出workspace无交。持有raw backing、views、metadata的强引用到设备完成；采样前与restore/比较前后fence，时间测量中排除clone/fence/audit。
5. 每次A/A/B/A前从同一immutable raw-byte快照恢复**整个backing**，按storage只写一次。恢复正确性使用byte equality；禁止混合typed大视图的float maxabs作语义判据。state/key/scale共享bit中的INT8/FP16数据可能被FP32解释成NaN/巨大值。
6. 使用私有metadata及新的forward-context缓存/显式native metadata调用。不能让同cache_group_key命中原12请求metadata，或让2请求结果污染回原路径。Graph另行独立capture；更换tensor指针/shape不能复用旧capture。

## 3. native历史state与metadata边界

`dsa_cp.py:1694–1718` Compressor输入包括x、两权重、FP32 state、APE/norm/RoPE、state_block_table、cu_seqlens、原token单位start_pos；cache_mode=1，coff依真实compressor_overlap。该函数返回后才rotate/quant/key-scatter/scale-scatter。

native目录：
`_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/`

- `compressor/arch32/compressor_block_vec_perf.h:803–858` 按batchIdx的state blocktable读取/写入，地址包含physicalPage×stateCacheStrideDim0，不能用dense4096B页替代4160B stride。WriteToCacheState跳过physical page0，ReadFromCacheState未在同处看到对0的跳过；需保留原0页语义。
- :863–880 SaveState写当前slice对应输入区间；:903–950 ReadState在压缩输出需要时读取当前group之前的state，OVERLAP还读前一个压缩group。对已读源码，原输入[start,start+8)之外的历史读可能早至max(0,floor(start/4)×4−4)。这是源代码保守候选，不是已验证native精确访问集；实际tiling/sIdx/coff与编译分支仍需匹配。
- 所以Run287“state新输入页无交”没有闭合递归读；只克隆当前输入页不足以搭建正确fixture。全backing克隆可避免先猜裁剪范围。
- `compressor_metadata/compressor_metadata.h:147–171,208–218` 每request有效输出行数为floor((start+qlen)/ratio)−floor(start/ratio)，输出按request前缀排列；slot用compressed位置，RoPE用其原始group-start位置。合法c4 qlen8每request2有效行。B应为两个完整owner请求×8输入=16，cu_seqlens=[0,8,16]，start_pos取原owner值，state与compressed table分别取原owner行。不得把start_pos除4传给Compressor。
- B有效输出共4行，A中对应owner的4行需按request→compressed-prefix映射抽取，不能按local12 query切片。native/metadata可能存在padding，按有效slot/output prefix比较；不要未经核实断言整个返回tensor一定只有4行。每个owner请求完整8行，包括跨rank split请求。
- 记录实际CompressorMetadata slots/RoPE，核对B与A被选owner完全一致，再比较Compressor结果。Q/QLI的原request索引保持不变，后续若接消费者需散回原physical slots而不是compact编号。

## 4. 首个最小 gate 应收敛在何处

**A/A先建立稳定性：** 两次原96输入从同一个私有raw快照调用同一Compressor；clone输出并fence；比较有效输出、typed真实state写集，记录byte变化范围和越界/canary。A/A如果不稳定，先定位native输出还是状态差异，不把B差异归为候选错误。

**B：** 仅compact16输入和2请求metadata，仍用完整私有backing及原physical page IDs。比较A选定owner输出 vsB输出、owner state写集与相关历史read域。非owner的A写而B不写是本实验预期，但只能在证明不与owner read/write物理bytes冲突后忽略；不能通过掩盖所有非owner逻辑页来漏掉别名污染。

初次不调用rotate、quant、scatter或QLI，避免一下跨过三个未闭合边界。可测真实16行tiling、Compressor latency与traffic；这些只回答该算子缩小request更新集合是否可行且有效，不给最终TPS。shape改变可能选不同tiling/数值路径，需按冻结数值门槛判定，不能要求或豁免任意阈值。

**之后才接rotate/quant/scatter：** `device_op.py:742–763` INT8 key scatter与FP16 scale scatter各自写同physical slot的不同typed区间。此时必须审计key-write↔state-read/write、scale-write↔其他view的跨域byte交集；对比typedkey/scale，最后保持原12请求metadata运行QLI比较topk。候选未写nonowner cache对后续Sparse、其他层、DSpark、prefix/refcount/COW/未来cohort消费者是否不可见，仍需独立生命周期证明。

## 5. 未闭合项与置信度

高置信：真实三视图unpack、4160B stride/scale byte offset、当前probe非完整typed manifest、逐tensorclone会改变alias；先做Compressor局部gate可降低歧义。
待测：本次实际INT8 key ABI、installed native tiling/branch、实际state历史读集、所有owner/nonowner物理byte交集、padding/sentinel行为。
未证明：省去非owner更新的全生命周期安全性、QLI/Sparse最终语义、Graph下成本、完整Target/E2E收益。

共享storage本身不是停止候选的理由，但**保留跨dtype alias与历史状态是fixture真实性的硬门槛**。
