# M005/M006/M011：真实 NPU allocator 类的 CPU 页元数据合同

本轮完整导入了安装源码中的 NPUPagedTokenToKVPoolAllocator，完成 75 组、485 项有效 CPU 语义检查，并把真实 allocator 接到上一轮的 tree/FULL 源码类与原始 free action 方法上。没有执行 NPU kernel、模型前向、物理 KV 分配或 PD 服务；整体 serving 保持 E0，已接受原生基线仍为空。

源码固定为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。R001–R003 新采集 15 个 canonical 文件，均与同一 git HEAD 一致。CPU preflight 又核对所有被测主源码与 snapshot/git HEAD 一致。ZCode 执行冻结命令，主控编写与审阅；配置别名 deepseek-flash，实际后端 release 未知。

## 当前分支及页单位

`kv_cache_configurator._build_token_to_kv_pool_allocator` 在 in-tree、NPU、attention backend 为 ascend（或其另两个条件）、非 hybrid SWA 且未复用 supplied allocator 的分支，构造 NPUPagedTokenToKVPoolAllocator。out-of-tree、hybrid/unified pool、HiSparse/DSV4、supplied draft allocator 等分支必须另行判断，不能只凭硬件名称推定本路径。

该分支同时把 `size` 与 `page_size` 乘以 `attn_dcp_size`。因此在构造参考中，base page 128、DCP 2 对应 allocator page 256；可分配页数不因两者共同扩大而翻倍，编号空间是全局虚拟 loc。这个单位不能直接读作每卡 HBM 字节。PD 模式 prefill/decode 设置 need_sort=True；释放页可先进入 staged 队列，available_size 已计入这些页，分配遇到现有 free list 不足时才合并排序。

9 组 factory 源码方法测试使用显式 context 夹具，调用真实 NPU 类构造器，核对 DCP 1/2/4、prefill/decode/字符串 null 的上述关系以及 nonhybrid draft 复用 supplied allocator。没有运行完整 configurator 模块，也没有取得任务实际启动 flags。

PageInterleavePoolAllocator 的独立源码使用不同的 min-class 容量定义。当前 in-tree SRT inventory 只在其定义文件发现类名；本轮没有把该类当作上述 Ascend 分支的实现，也不据此排除外部平台扩展的可能性。

## 实际执行及证据边界

| Run | 执行范围 | 结果 |
| --- | --- | --- |
| R004 | 完整 NPU allocator 模块；4 组生命周期、32 组大 extend、3 组阈值、16 组 decode、1 组耗尽 | 56 组、250 项通过 |
| R005 | 首次耦合 harness；完整预算模块导入尝试 | harness NameError，0 项检查，主控判 INVALID；另有真实依赖 UNAVAILABLE |
| R006 | 修正 harness；10 组真实 allocator/tree/budget 耦合与 9 组 factory context | 19 组、235 项通过；保留 R005，未重试预算模块导入 |

R004 是正常完整模块导入，使用原始类、继承方法和 helpers；没有提取 allocator AST 或替换其 kernel。Torch 为目标镜像的 CPU dependency view；device=cpu、kvcache=None，dtype 构造参数仅为夹具，不是实际模型 KV dtype。它管理真实类自己的槽位/page tensor，不持有 KV payload。

有效检查覆盖整页 alloc、容量不足不分配、debug 对未对齐 alloc/free 的拒绝、单槽 free 归还整页、free_segment 的整页边界、need_sort 的 staged 容量与合并、free_group 对传入 view 的 clone、组结束前容量不暴露，以及组结束后唯一页池恢复。debug gate 是本次显式启用，不是生产配置事实。非法输入被拒绝的检查不表示所有 API 具有事务性回滚。

R004 的长输入在 B=2/3、raw length 32768、prefix 0/127/128/129、page 128/256、need_sort 两种配置下，新增页数均达到 200，执行 NPU allocator 原有的 alloc_extend_naive Torch 分支。独立逐位置 page oracle 核对输出、唯一性、空闲页身份与容量；返回 dtype 为 int32。decode 的 mixed/all-new/no-new/empty-prefix 也执行真实 Torch 路径。

## 200 页分界与 16K chunk

NPU alloc_extend 在新增页数低于 200 且容量足够时，导入并调用 sgl_kernel_npu 的 alloc_extend_kernel；达到 200 时走 Torch 分支。本轮 200/201 页边界执行了 Torch 路径；199 页用例因池只有 198 页，在 native import 前返回 None。它只证明容量拒绝，**不证明 199 页成功路径**。

这一区分直接影响目标 fidelity：若单次 extend 总量不超过 16384、B≤3、allocator page≥128，则新增页数的保守上界不超过 `ceil(16384/128)+3=131`，低于 200。故已通过的大 extend Torch 路径不能证明上一轮 16K chunk 参考中的 prefill pass 原生路径可用。实际 page、DCP、host load、batch 与 chunk 配置仍需任务运行合同确认。

小页成功分支、kernel 编译、原生 int32 loc 消费和设备/PD 可见性均未执行。没有用 spy kernel 或 CPU 替身把这个门槛计成通过，也没有调节 200 页阈值。

## 真实 allocator 与 source tree 的耦合

R006 使用完整真实 NPU allocator，保留 UnifiedTreeCore/FULL 的完整原始类体，以及 facade 的原始 action 消费、free dispatch、容量查询方法体；这些方法绑定到最小字段 adapter，未构造完整 facade。FULL 的原始 apply_component_action 把回收槽位交给真实 allocator.free_segment，不再使用上一轮的 free 计数夹具。

在 B=2/3、Eagle、page 128/256、raw 512/1024/32768 的 10 个新组合中，先 insert/pin 第一请求，再插入共享分支；重复前缀 action 确实归还 page 容量，后续 receipt 只保护新增尾部。所有 cached page 被 pin 后无法回收；释放 private tail 后，tree evictable 转为真实 allocator free，预算总容量守恒。清空 tree 后仍保留 caller-owned 未缓存尾页，最后由请求清理，原始唯一页池完全恢复。

本轮还验证 page 256、raw 512 的 Eagle 可缓存 256、未缓存尾部 256；该组合明确采用无共享逻辑页，避免把 128 个共享 token 当成一个 256 页的可缓存共享边界。

完整 PrefillBudget 模块导入在 R005 失败：SWA → BaseSWAKVPool → memory_pool → cache_move 缺 sgl_kernel。R006 保留其 traceback 而未重试，执行原始完整 PrefillBudget 类体并使用真实 allocator 与 tree getter。**真实 BaseAllocator.create_prefill_budget 动态导入入口没有执行成功。** 完整 UnifiedRadixCache facade 的既有导入失败也未重试。

## 失败、隔离与后续候选

R005 因主控 harness 漏定义 future AST node 而没有开始任何 semantic case；其 CLI/inner exit 虽为 0，主控结果仍为 INVALID，未获得检查数量或支持性信用。修正与哈希记录在 R006-derivation，原始失败、冻结输入与 receipts 保留，没有修改旧 Run。

R004/R005/R006 三个任务 CPU 容器均已清理；参考服务的 ID、启动时间、restart count 未变，四个只读 metadata/tokenizer 文件未变。无 accelerator 设备、无网络、无模型权重或目标模型推理请求。

本轮验证现有实现，未应用优化。下一项候选审查是追踪真实 alloc_extend 调用是否已传 num_new_pages：NPU 方法未给参数时从 device tensor 求和并 item，已有参数是否可避免同步必须由真实 caller 证明，不能把现有支持重复当作新优化。额外缓存页计数会有失效、分支与维护成本；不同 prefix、host-load、DCP、取消与重用场景可能回退。任何 slot/page 身份不一致、提前释放或漏释放均否定候选；没有同条件原生 trace 与端到端结果，不填加速比。

Loop INCONCLUSIVE：真实 allocator 的 CPU 逻辑元数据合同得到支持，短 chunk 的 native 成功路径、物理 KV 生命周期与 PD+MTP 性能仍未达到验证门槛。
