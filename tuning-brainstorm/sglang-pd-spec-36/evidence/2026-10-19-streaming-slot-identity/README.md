# M036：流式会话真实组合路径与 slot 身份候选

本轮核验固定 SGLang `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的54个相关源码文件。原实现参考 R004 完成52组1,104项 CPU 检查，**1,068通过、36项身份隔离失败**；离线 slot 身份候选 R005 完成55组1,146项，全部通过。共同1,104项中只改变36项隔离结论，新增42项检查覆盖旧类 slot、预中止身份变化和默认 base。**这是 slot 路径的局部 CPU 条件通过，服务候选未接受、未应用；原 raw-tree 键重合与持久存储世代仍未解决。**整体 E0，accepted baseline=null，无原生性能结论。

## 当前行为与实际调用链

原 Session.create_req 拒绝并行在途、replace、offset 和 drop_previous_output；普通追加时共享并修复 token 数组，把新请求的 lora_id、extra_key、cache_salt 交给 Req。它没有比较前后身份。原 StreamingSession.find_active_slot 按 session ID、持有状态和预中止标志查找；原 facade.match_prefix 先尝试 streaming，再查 Core。[Session 构造](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/session/session_controller.py#L201)、[slot 查找](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/session/streaming_session.py#L171)、[facade 次序](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/unified_radix_cache.py#L549)。

本轮实际执行原 GenerateReqInput 归一化 → 真实 registry 解析 → 原 tokenizer carrier → Scheduler 的 session 分支 → 完整 Session/SessionController → Req.init_next_round_input → 原 facade match 方法 → 完整 StreamingSession → CPU row 索引恢复。首请求经过真实空 Core/FULL 回退；后续命中原 slot 时 Core 不再被调用。加载记录和 AST 哈希见结果 source_components。

矩阵使用已冻结512/32768输入，保留128/256输出预算，PREFILL/DECODE 枚举，以及三种 page/分支组合：page1普通控制、page128普通控制、page128模拟 is_npu=True 的 Python 分支。每种分别测试同身份、改 adapter、改 caller key、改 salt，共48组；另4组检查闲置关闭、在途关闭延迟到完成、预中止恢复和小于一页的回退。请求顺序执行，没有2/3并发服务压力或容量结论。

参考中三种身份变化全部通过原 admission 并恢复旧索引，共36项合同失败。512输入时普通分支返回515项，模拟 NPU 页对齐分支返回512项；32768输入时分别返回32771和32768项。这里包含夹具提供的上轮3个输出 token，**不是实际模型输出或物理 KV 值**。索引逐项等于原 slot 所在 CPU row 的前缀，重试继续恢复同一 KV record；同身份和所有生命周期对照通过。[参考原始结果](streaming-reference-result.json)。

## 原释放路径与夹具边界

实际运行完整 BasePrefixCache 与 StreamingSession 类，包含 _trim_overshoot、_free_tail、_free_kv_aligned、release_session，以及原 coalesce_ranges/free_kv_row_segments。CPU 页账本拒绝重复页释放和未持有页释放；row 账本拒绝重复释放。测试保存/恢复真实 ReqKvInfo 对象，并验证中途 abort、闲置 close、延迟 close/reap 均只释放一次，预中止保留旧 slot，下一次有效追加修复 token 数组。小于一页时原模拟 NPU 分支释放 slot 并回退空 Core。[释放分段](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/common.py#L121)、[尾部处理](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/session/streaming_session.py#L506)。

页/row allocator、树锁释放、配置、容量、队列、时间和 metrics 是明确的 CPU 外围账本或夹具。facade 仅运行原 match 方法，rank_consensus 使用 identity decorator，未构造完整 facade 或进程组。is_npu=True 只是 Python 分支控制，没有 NPU。输出3个 token 和额外分配间隙用于结束/回收状态，仍保留128/256预算，不能作为缩短负载的 benchmark。没有 HTTP/IPC、真实 forward、采样、PD 搬运、MTP worker、LoRA 权重、物理页回收、Mamba/SWA 状态或实际树锁所有权证明。

helper 复用 L034/L035 的依赖装配，原始结果 boundaries 中保留了早期子组件的历史说明；本轮实际范围以本报告及 source_components 为准：完整 GenerateReqInput/SamplingParams/Session/StreamingSession 确实参与本轮，媒体检查没有重跑。本地 Python3.12.14、Torch2.8.0、msgspec0.19.0、numpy2.3.3、transformers5.12.1 与独立 Pillow12.0.0 overlay 不代表目标镜像。

## 候选、对照与成本

[离线补丁](candidates/L036-streaming-slot-identity/slot-identity.patch)仅增加 slot 身份：首次保存时记录 `(req.lora_id, req.extra_key, req.cache_salt)`；原预中止处理之后，若 incoming 身份不同或旧 slot 没有身份字段，释放旧 slot，再走已有 raw lookup。保留 token 历史，让后续路径重新计算；同身份维持 streaming 复用。没有叠加上轮 JSON Req 补丁，没有改规范源码或服务。

R004 的命中/回退、来源与重试断言对两个分支均有效，隔离失败数不参与正向断言，避免修复跨命中反而被旧观测断言误判。R005 在共同52组1,104项全部通过，只改变36项隔离结果。新增3组42项也通过：实际原 SessionSlot 类对象缺少新字段时先释放；身份不同但已预中止时保留旧 slot 并可恢复原身份；已知默认 `(None,None,None)` 与未知 legacy slot 不混淆。[候选原始结果](streaming-candidate-result.json)、[逐项比较](correctness-comparison.json)。

新增成本是每 slot 的三字段引用 tuple 与字符串比较；不同/未知身份会损失热 slot，可能重新计算完整历史、增加分配压力和时延，尚未计时。当前只验证 FULL KV、单进程 CPU 外围账本；SWA/Mamba/host/CP 等组合还需保护。局部正确性通过不代表实际新输出正确、性能更好或可直接上线。

## 未闭合的缓存世代

原 storage_namespace_seed 对 extra_key/cache_salt 作带长度与 presence 的哈希；它区分这两个输入，但相同的内部字符串仍产生相同 seed。[存储 seed](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/utils.py#L125)。facade 的存储 attach 还传递 served_model_name/config，外层 backend 隔离未完整追踪。因此不能声称所有存储层均无隔离，也不能把 session ref tracker 的 ID incarnation 当作全缓存/模型世代。

本候选只在 streaming 入口隔离旧 slot，**没有修复 L035 原 Req 直接拼接的 raw-tree 键重合，也没有使上轮 JSON 候选通过旧缓存迁移**。它的安全回退最终仍依赖 raw cache 身份正确，持久层/混版本/模型切换必须继续闭合。上轮反例原样保留。

## 执行记录与下一步

R001 收集 common.py PASS。R002 因装配遗漏 ceil_align 所需原 ceil_div 而 INVALID：完成4组92项后中止，部分结果全部排除。v2 仅补原依赖，R003有效52组1,104项、36隔离失败；R004调整观察断言成为候选参考，覆盖不与R003相加；R005候选55组1,146项通过。5条冻结命令均由 ZCode 简单执行器各执行一次，外层 CLI0 与内层0/2/1/1/0分别审查，输入哈希未变。审计初稿误把 `ls` 中的脚本文件名计为执行，v2按 shell token 的解释器调用识别纠正；两份审计脚本保留。[执行审计](zcode-audit.json)。Loop 创建时漏显式 ID，在任何 Run 前仅纠正标识，合同未改，记录见 [管理记录](admin-correction.json)。

下一步先追踪存储 attachment/controller/backend 的独立 namespace 与模型更新/清空生命周期，并检查 slot 候选在 SWA/Mamba/CP 下的释放边界；随后形成单一整合方向，不能把两个局部候选拼合后直接接受。错误复用、预中止误清旧状态、重复释放、同身份失效或无法承受的重新计算成本均否定候选。

任务44/48资源归属、原模型 config 哈希、量化/KV dtype、安装 flags、真实扩展业务和 Ascend PD+MTP 组合仍 UNKNOWN/待确认。历史 NEXTN steps3/topk1/draft4 只作参考；没有用 CPU 分支模拟替代支持性。原生进入条件见 [native-next-contract.json](native-next-contract.json)。**Loop INCONCLUSIVE，slot 候选仅 LOCAL_CPU_CONDITIONAL_PASS，整体任务 ACTIVE/E0/null。**
