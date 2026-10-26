# Astra High：ownership 原路径 probe 最小实施清单

2026-09-26；只读设计，未操作 Run285/NPU/共享源码。目标是验证“一个固定 c4 layer 的非 owner 请求更新是否存在未闭合消费者”，不执行删写、不改变 Graph/metadata。

## P0：下一项 probe

选 Target layer2（c4）完整一个 c12 cohort，记录 all8 ranks；沿用原路径，不开 ownership 优化。完整覆盖首轮→压缩/页边界→首次 parking→终轮，并连接下一 cohort 的 prefix 命中记录。历史前8轮记录不足以证明 cohort 生命周期。

| 位置/频率 | 最少字段 | 检查 |
|---|---|---|
| handoff，一次 | run/source/binary id，rank，cohort_id，request_id→slot，group→layer/cache kind，全部Target/DSpark cache storage base/bytes/data_ptr/offset/shape/stride/dtype | 用storage hull判别实际alias；不要只比gid/data_ptr |
| derived metadata之后、Target之前；小向量每cycle | global_qsl[13]、global_seq_lens[12]、positions[12,8]、CP local_start/end、local_qsl[13]、local_seq_lens[12]、start_pos[12]、active_mask、parked、acceptance推进量 | 所有受审native调用实际metadata一致；owner集合是否变化；parked是否仍有query |
| 同一点；首轮/页变化/parking/终轮 | layer2 indexer key/scale/state、main compressed/state、SWA 的 block_size/ratio、每request相关blocktable行及物理页；实际scatter slot(page,offset) | 构造 owner历史read集合与 nonowner实际write集合；按cache storage区间关联，不把不同cache同号page当alias |
| Target之后，只读诊断派生 | native op/tiling/capture/replay id，实际compressor_metadata输出slot，输入cu lengths/startpos对应请求标签 | 映射必须与native一致；只记录真实实际slot或显式标“保守推导”，不混用 |
| proposer之后 | DSpark组/cache interval、context slot实际值、blocktable、context positions/请求标签、关键valid length | target候选写页是否被DSpark引用；区分同号block与真实同storage |
| terminal/free及下一handoff | scheduler num_computed_tokens/num_tokens/placeholders，group hash有效frontier、hashed page/refcount/共享/COW、connector启用/导出区间；下一cohort prefix hit页与slot/owner | 被省略写入的区间会否仍作为有效prefix/state复用 |

## 两种行集合必须同时保存

由**实测** global_qsl 得每请求完整新输入区间 [qsl_i,qsl_i+1)，由CP区间交集得 local_query_rows。定义owner为交集非空的请求，而候选要保留的update_rows是这些owner请求的**完整新输入区间**，不是local_query_rows。

在实测均为8行/请求且rank分12行时，rank0有local请求0的8行+请求1的4行，但应保留请求0、1的完整16行更新；请求1后半4行还可能成为未来可见KV。不要在probe中预填12→16结果；先从实际qsl计算并assert总数。
输出schema至少分开 local_query_rows、owner_request_slots、owner_full_update_rows、nonowner_full_update_rows；记录query causal key frontier local_seq_lens，不把它当未来永久有效前沿。parked请求仍占slot且local_qsl可非零，不能自动归入nonowner。

## 无需 native 改码也可先回答的问题

P0只需保守read集合：QLI按owner请求blocktable取其可见全部compressed key/scale历史；SWA按窗口与local长度；c4 main compressed可先取owner全部可见历史页，不必先采精确topk。保守集合与候选write集合无交集能排除这一类冲突；有交集应按page内offset/byte及是否共享只读prefix细分，不能自动判失败/安全。
Compressor state单列递归read/write范围；native确切state读域未知则标unknown，不用仅scatter输出覆盖代替。
不必为精确topk或native内部访存新增热路径hook才开始P0；需要判别时再缩小到该未解项。

## 复用现有资产与验收

- 扩展 diagnostics/target_page_audit.py 的source→group与actual compressor slot coverage；当前返回write coverage，不等于read closure。
- 扩展 KVSlotAudit：它的通用positions//block_size核验不能代替compressed ratio/格式专属映射；使用真实req_metadata及compressor_metadata输出。它目前只记录limit前若干cycle，须改为小向量全程+大表按变化保存。
- 跨worker/engine日志以(run,cohort,request_id,rank,phase,cycle)连接，不能只靠slot或时间。
- 持久记录全部cycle的owner/qsl/hash变化计数；大blocktable按首次与diff保存。异步CPU导出若使用，须用事件保证数据取自目标phase；简单同步诊断也可，但此Run不用于正式性能裁决。
- Gate：all8身份齐全；native/推导slot覆盖无遗漏；owner稳定或每次变化可解释；物理alias和当前/下一消费者闭包全部分类为已排除/冲突/unknown。unknown不得自动视安全。

优先级：Run285数值gate结束后，安排上述原路径probe；它与CP overlap的FULL Graph gate是两条独立证据线。先P0，不先删96→16，也不因收集到16行owner就宣称HBM减少5/6或给TPS预测。
