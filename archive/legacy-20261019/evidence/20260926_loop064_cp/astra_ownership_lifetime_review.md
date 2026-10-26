# Astra High：ownership 消费闭包与生命周期续审

2026-09-26。仅只读源码；未运行 NPU、未改服务。读取时 HEAD 07cb354345bf8ebc1a23cb7f7ec270b9856b2294，另有诊断工作树改动。本文不作 KEEP 裁决。

## 结论

不能从 local query rank-owner 直接推出非 owner 的 KV/state 更新可删。可支持的条件命题仅为：固定 cohort、request→slot、CP local query geometry 和 native metadata 一致时，一个 Target replay 的正常语义 KV 读取按有 local query 的请求行访问本 rank cache。owner 请求需要可访问的完整历史，不是仅本 rank 的 12 个新 token。延长到整个 Runtime/后续 cohort，必须排除物理 alias、共享 prefix 可写页、Compressor state 递归、DSpark/connector 消费和 hash validity 的影响。

96→16 输入行只是 [0,8,...,96] query_start_loc 下的条件推导，不是 6 倍算术、HBM 或 E2E 收益。一个 c4 indexer update 是值得验证的候选，尚未证明可删。

## 1. Native 消费（事实）

以下相对路径以 /data/wio/vllm_ascend_26/framework/vllm-ascend/ 为根；native 根为 vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/。

- attention/context_parallel/dsa_cp.py（前缀 vllm_ascend/）:1430–1431 取 local_qsl/local_seq_lens；:1781–1805 QLI 使用 local_qsl[1:]、local_seq_lens，本 rank key/scale tensor，indexer req_metadata.block_table。
- vllm_quant_lightning_indexer/arch32/quant_lightning_indexer_kernel.h:438–452 在 S1 或 S2 为 0 时设 curActSeqLenIsZero；:583–598 清无效输出并 continue，可能 drain 上一流水线任务。:312–340 的零长度处理无本请求 key 读。
- quant_lightning_indexer_service_cube.h:315 按 runInfo.bIdx * maxBlockNumPerBatch + s2BlkId 取 key 页；service_vector.h:113–137 对 scale 同样按 batch 行取页。因此有效查询正常消费不隐式切换到其他请求/rank。
- dsa_cp.py:1593–1644 三种 sparse attention 均传 local_qsl/local_seq_lens；ori/cmp KV 是本 rank tensor，ori/cmp blocktable 是各自请求行表。c4 另传 QLI topk。
- sparse_attn_sharedkv/arch32/sparse_attn_sharedkv_scfa_kernel.h:325–350 按 bIdx 取 Q/K 长度；:699–750 零 query batch 可能做流水线 flush，零 query 不能直接等于零物理访存。
- sparse_attn_sharedkv_scfa_block_cube.h:384–395 SWA DataCopyPA 带 info.bIdx；compressed 先经 vector gather，scfa_block_vector.h:547 按 runInfo.bIdx 的 cmpBlockTable 行取 key，进入 kvMerge workspace 供 Cube。这个 gather 是本设备 staging，不是跨 rank KV gather。
- dsa_cp.py:_restore_tp_head_layout 的 AllToAll 搬运已完成 attention output，不是“各 rank 必须代其他 rank 持有历史 KV”的直接证据。

置信度：源码逻辑高；实际加载 native binary 与所读 arch32/tiling 是否一致须留 SHA/op/tiling 证据。这里只闭合了 c4 SCFA 的主要消费链，c128/其他 tiling 不自动覆盖。

## 2. 时间范围与消费者

### 固定 Runtime

runtime/target_metadata.py:97–105 初始化 local spans/_local_active；:219–220 非 local 请求 keylen 为 0。runtime/fixed_serving.py:96–117 对 parked slot 回退 position、置 active_mask，但仍保持固定 slot 执行。已读路径没有 cohort 内 refill/migration。

条件推论：若实际 bootstrap qsl 是 [0,8,...,96] 且 Graph metadata 无陈旧，owner 请求集合可保持到本 cohort 结束。parked 不表示 local_qsl 清零，不能顺带删除 parked owner 更新。split 请求第一 owner 当前 causal offset 少掉后半 query，并不表示尾部 KV 永远不需要。

Indexer Compressor state 是后续 cycle 的真实消费者。永久停止某非 owner 请求更新，要证明其在整个区间不会成为 owner，state 区域也不与 owner 或其他消费者别名。单次 QLI 正确不证明 state 连续性。

### DSpark

models/deepseek_v4_dspark.py:107–116 构建 mtp.* draft layers；:170–219 用其自身 wkv/kv_norm 与 swa_cache_layer.kv_cache 生成并存 context KV。target_layer_ids 是 aux hidden 来源；未见直接读 target indexer cache 的源码边。

但模型名不同不证明物理隔离。bootstrap/vllm_extreme_handoff.py:95–140 manifest 用 data_ptr 精确相等发现 alias；只输出 target 的 storage_ptr/offset，不足以穷尽 draft tensor 的重叠区间。不同 data_ptr 的 view 可重叠；应记录所有 Target/DSpark/state/scale 实际 storage byte region、offset、stride。不同 gid 也不等于隔离。

### Prefix / 后续 cohort：纠偏

patch/platform/patch_kv_delivery_preemption.py:970–982 extreme_bulk_output 直接调 KVDeliveryScheduler._update_request_with_output；:1186–1203 仅 append/check_stop，绕过常规 AsyncScheduler placeholder/cache bookkeeping（vllm/v1/core/sched/async_scheduler.py:69–74）。
vllm/v1/core/sched/scheduler.py:2220–2275 free 进入 KVCacheManager.free；kv_cache_manager.py:506–516 只 coordinator.free，未新增 cache_blocks。

所以“终端自动把全部 Runtime 1024 输出标为有效 prefix”不是源码事实。但 allocate 时 kv_cache_manager.py:495–502 会以 min(total_computed+new, request.num_tokens) 提前 cache；handoff 前已哈希页、尾页 padding、共享 prompt/COW、state cache 恢复、connector 导出仍需实际证据。dsa_cp.py:1325/1396 wait/save connector 和 :1576 notify 也须确认当前启用状态与消费范围。

下一 cohort 可以换 request 顺序/owner；被 hash/cache/connector 宣称有效的省略区间必须先补齐、失效或证明确实不会命中。不能把本 cohort owner 外推至服务全生命周期。

## 3. 最少运行探针（先原始路径）

1. 逻辑/Graph：rank、cohort/request ids 的 slot 顺序、cycle、global qsl、CP local_qsl/local_seq_lens、start_pos、active/parked、acceptance advance；受审 layer 的 native op/tiling/binary id、sas/qli metadata generation、capture/replay id。至少首 cycle、跨压缩/page 边界、首次 parking、末 cycle；owner 不变性可每 cycle 累计 hash/违规数。
2. 物理：受审 indexer key/scale/state 及潜在 alias 的 Target/DSpark tensor storage base+bytes/data_ptr/offset/stride/shape/dtype；每 request block ids，实际 Compressor/scatter slot(page,offset)，owner 历史 read 页与 nonowner write 页交集，必要时下钻 byte/slot。state read/write 与 recurrent startpos 单列。现有 target_page_audit 证明 write coverage，不等于 consumer read closure。
3. 发布/复用：handoff 前和 terminal/free 的 scheduler num_computed_tokens、num_tokens、num_output_placeholders；各 group hash 页/有效 frontier、refcount/共享页、COW、connector；下一 cohort prefix hit 页、有效区间及 slot/owner，证明不读取未更新区间。

1+2 是单层候选前置 gate；3 是 warm-cache 产品 E2E 前置 gate。只跑 cold/singlecohort 不充分。同 state A/B/A 要恢复全部被写的 Target/DSpark/state/scale/metadata，比较 topk、attention/logits、acceptance、next draft、后续 owner cache/state 和完整 token 输出。被刻意删除的 nonowner 字节不必全等，但必须证明不再作为有效状态消费/发布，不能忽略其差异。

## 4. 实验与 Bound

优先原始路径 ownership/physical validity 记录，通过后仅改一个 c4 layer 的 indexer Compressor+quant/scatter 非 owner 请求工作，保留 QLI 全量请求索引及下游形状。compact 输入必须同时正确映射 cu lengths、原始 request→blocktable/startpos/slots；不能只 slice x 或复用全96的 compressor_metadata。

T_resource >= max(W_cube/P_attained, W_vector/P_vector, B_hbm/BW_attained, B_link/BW_link, ...)；
T_cycle >= max(T_resource, critical_path(DAG, attainable joint costs))。
各资源量是固定算法与合法 ownership 下必要工作，不能直接以当前重复96更新充当 compulsory。重分配历史KV、失效/补齐和生命周期通信也进入 DAG。96→16 通常不同比例减少权重读取/tile；same-state Target/cycle/allrank rendezvous 与未 profile E2E 才能校准收益。

当前可证数值潜力仍为 unknown。源码收窄语义不确定性，不提供 TPS ceiling。
