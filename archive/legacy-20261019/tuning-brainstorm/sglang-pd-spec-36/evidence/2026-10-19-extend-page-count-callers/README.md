# M005/M017：alloc_extend 页计数的实际调用合同

本轮完成 30 组、173 项 CPU 合同检查，确认了 ordinary prefill、PD preallocation、EAGLE/MTP 调用中的主机长度来源，以及 NPU allocator 的默认页计数读取点。22 次带标记的 CPU 观测在原 allocator 的显式 `.item()` 处停止。没有执行原生小页 kernel，没有测同步耗时或服务性能；整体保持 E0，已接受 PD+MTP 原生基线为空。

源码固定为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。R001–R004 核对 7 个 canonical 调用/producer 文件，R005 preflight 将被测主源码与 snapshot/git HEAD 比较；执行由 ZCode deepseek-flash 配置别名完成，真实后端 release 未知。

## 实际路径

ordinary `ScheduleBatch.prepare_for_extend` 从请求的 extend range、prefix_indices 等主机数据构造 seq_lens 列表，再由同一列表分别创建 CPU 与 device tensor。`allocation.alloc_for_extend` 同样从 batch.prefix_lens 创建 CPU tensor 和 nonblocking device copy，随后把两份长度交给 `alloc_paged_token_slots_extend`。

普通 helper 的 extra kwargs 只在 page-interleave 或 DSV4 条件下加入 rotation_bases、request indices/pool。所审阅的常规 NPU full、non-DSV4、shard1 分支没有 num_new_pages 参数，因此真实 NPU allocator 使用其默认 None 分支。

PD `alloc_for_decode_prealloc` 从 total_prefix_len、fill_len 这两个主机标量分别构造 int64 的 CPU/device tensor。non-SWA/non-DSV4 的直接 alloc_extend 调用也没有给页计数。这里只执行该局部函数，未执行完整 DecodeTransferQueue 或 PD 网络。

EAGLE 的 `page_aligned_decode_alloc_lens` 由每请求 host kv_allocated_len、kv_committed_len 和 reserve 计算 cur/next；next 至少为当前 allocated，并把 committed+reserve 向页边界对齐。`eagle_prepare_for_decode` 先创建 int32 CPU tensor，再将其 nonblocking 拷到 device，随后传给 `alloc_for_spec_decode`。

NPU speculative dispatch 通过 dsv4_allocator 文件中的 wrapper。该 wrapper 在 non-DSV4 时仍原样委托普通 helper，不能因为文件名含 DSV4 就排除它。CPU 运行中 EAGLE 使用真实 dispatch map 的 CPU 默认分支；NPU wrapper 另有直接调用检查。没有把 CPU dispatch 当作实际 NPU 调度执行。

上述是固定源码与局部 CPU 合同的证据；任务实际 flags、完整 scheduler、并发流和原生数据生命周期仍未确认。

## R005 的执行范围

完整导入 NPU allocator、allocation_sizing、page-interleave helper 与 utils.common。调用方使用保留原始函数体、装饰与 AST 哈希的源码函数，包括普通 allocator helper、PD prealloc、EAGLE prepare、spec allocation、get_last_loc 的 Torch 分支和 NPU wrapper。没有导入完整 allocation/PD/EAGLE 模块。

真实 NPU allocator 实例的 entry observer 记录参数，再调用其未修改的原方法；仅把逻辑 device-length 输入换为带标记的 CPU Tensor 子类，其显式 item 会抛出观测异常。它不模拟 NPU 执行延迟，也不返回伪造 kernel 输出。观测停止前 free page tensor 保持一致，kernel 尚未被导入或执行。

| 分组 | 组数 | 观察 |
| --- | --- | --- |
| 普通 prefill helper | 8 | B2/3、page128/256、prefix0/127；CPU/device 镜像相同、默认 count=None |
| PD preallocation | 8 | prefix0/127/128/129；主机标量构造的两份长度一致 |
| EAGLE producer + spec helper | 4 | 执行 int32 producer 与 page-aligned sizing；reserve8/topk1 为明确 context 参考 |
| 直接 NPU non-DSV4 wrapper | 2 | wrapper 委托普通 helper，仍到达默认 scalar-read 分支 |
| 现有显式 count 的容量拒绝 | 4 | 真实池已占满，显式 CPU count 跳过带标记 item 并在 native 前返回 None |
| 现有显式 count 的大页路径 | 4 | B2/3、page128/256；与默认算法输出 int32 loc 和剩余页池一致，无带标记 item |

共 173 项检查。前四类共 22 次显式 scalar-read 观测；后两类使用的是**已有** optional-count 参数，不是新增优化。本轮没有应用候选修改。

普通 helper 的长度是显式 fixture；prepare_for_extend 的 producer 关系由源码核对。PD 与 EAGLE 的 producer 函数体实际执行。eviction 仅记录调用、context 与 req metadata table 为边界夹具；prefix slot 来自真实 allocator 的 CPU 分配，kvcache=None，没有 KV payload。

EAGLE observer 之前的 decode_batch_idx、PD 的 kv_allocated_len 等原始状态改变仍存在；测试不作异常回滚或完整调度事务保证。

## 可研究的候选及边界

NPU allocator 在未给 num_new_pages 时，目前从 device prefix/seq 做 ceil-page 差、sum、item。所审阅的 producer 已保有同源主机镜像，因此可以继续设计“对合格 CPU 镜像使用 CPU 页计数”的条件候选。现有显式参数能力属于基线。

候选必须保留显式 count 的原行为，并为 None、非 CPU、不合适 dtype/shape 等既有调用保留原路径。主机镜像必须是该次 device 输入的权威副本；CPU/device 不同值不能仅凭 shape 检查视为安全。page 必须使用实际 allocator 的 page_size，不能把 base page、DCP loc 与物理字节混用。

新增成本包括主机 tensor 算术、metadata gate 与回退分支。短路径的 CPU 成本可能抵消收益；device 已就绪时 item 的暴露等待可能很小。只有原生 trace 才能判断是否为实际瓶颈。任何页数、分配结果、返回 None、剩余页身份、dtype 或生命周期不一致均否定候选。

本轮 ≥200 页的大输入 Torch 分支仍不能证明 16K chunk 的小页 native 成功路径。后者、原生 H2D/stream 依赖、PD 传输和端到端收益均需同条件原生测试。不得用带标记 CPU 观测的次数当作 NPU 等待次数或加速比。

任务 CPU 容器已删除，参考服务的 ID、启动时间与 restart count、四个只读模型 metadata/tokenizer 文件保持不变。没有权重挂载、accelerator 设备或目标模型推理请求。

Loop INCONCLUSIVE：调用合同与候选前提得到局部支持，原生同步成本及优化正确性/收益未验证。下一轮先冻结独立候选设计与回退合同，再做隔离 CPU 差分和原生验证。
