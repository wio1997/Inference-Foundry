# M009/M010/M011：默认 unified cache 的引用、分裂与可回收预算

本轮确认了当前安装源码的普通 cache 构建路径，并完成 14 组、300 项源码类 CPU 语义检查。没有修改生产实现，没有性能结论。整体 serving 证据仍为 E0，PD+MTP 的已接受原生基线仍为空。

固定源码为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。R001–R006 只读采集 canonical 文件并与同一 git HEAD 比较；R007/R008 在任务自有的隔离 CPU 容器中执行。ZCode 按冻结命令执行，主控编写与审阅；配置别名为 `deepseek-flash`，真实后端 release 未知。

## 当前实现与本轮范围

`kv_cache_builder.py` 把 speculation algorithm 的 `is_eagle()` 传入 CacheInitParams，并经 registry 选 cache。registry 的普通 fallback 为 UnifiedRadixCache；注册 backend、chunk/disabled radix、SWA、LMCache/FlexKV 等分支可以改变选择。UnifiedRadixCache 再经 tree core registry 选择后端，内置默认 Python 路径构造 UnifiedTreeCore。上述是源码选择关系；任务原生 PD 服务的实际 flags、环境覆盖与组件组合尚未取得。

因此本轮使用 UnifiedTreeCore 与 FULL component，未用 legacy RadixCache 替代默认路径。复用 legacy 文件中的 RadixKey 是 unified core 自身的真实依赖。

CPU 执行保留整个 UnifiedTreeCore、UnifiedTreeNode、UnifiedLRUList、TreeComponent、FullComponent、相关 records/actions/interface 的原始类或函数体，逐项记录 AST 哈希。component_type、CacheInitParams、evict_policy 三个 leaf 文件完整导入。PrefillBudget 使用原始完整类体，通过实际 tree 的 bound getter 读取 evictable；non-Mamba 分类、allocator 容量及 free/action 接收端为明确夹具。

仅执行 FULL、无 host/storage/linker、无 session radix 的路径，事件记录关闭。StorageMedium 仅作禁用事件调用的标签夹具。没有构造完整 UnifiedRadixCache facade，没有执行实际 KV allocator、H2D/PD 传输、原生 kernel、模型前向或 scheduler 服务。

## 可复查结果

| Run | 新增范围 | 结果 |
| --- | --- | --- |
| R007 | page 1/128 × Eagle 开关 × raw length 512/32768 的 8 组共享三分支；另 2 组 extra_key/cache_salt 隔离 | 228 项检查通过 |
| R008 | page 128、Eagle 开启，512/32768 输入先 pin，再由 match 或 insert 引发 split | 72 项新检查通过；未重跑 R007 组 |

FULL 首次加锁把路径中 lock_ref 为 0 的槽位从 evictable 转入 protected；已有引用时继续增加引用次数，但不重复扣容量。三个分支共享逻辑前缀 128 时，首次 pin 扣完整可缓存长度 C，后续分支各扣 C−128。重复 pin 的 delta 为 0，部分 release 保留其他 holder 的保护。

PrefillBudget 的 getter 跟随真实 tree ledger 变化。锁住所有叶子时，请求回收 1 个槽位无可回收结果；释放一个分支后可回收整个尾部叶子，可能超过请求值。夹具处理真实 core 返回的 free 列表后，allocator free 的增加与 evictable 的减少相抵，预算总可用容量守恒。该守恒只验证计数合同，不验证物理页释放或可见性。

先 pin 再 split 的四组中，原 node handle 继续作为 receipt 锚点；新父节点继承已有引用覆盖，新分支的尾部仍可回收。另一 holder pin 后，释放原 receipt 不会释放共享祖先最后一个引用。错误 receipt 锚点在计数改变前触发断言；跳过 FULL 的 receipt 获取与释放不改变真实 holder 的 ledger。

与 legacy 直接 insert 不同，unified core 插入重叠前缀会发出重复槽位释放 action。本轮显式接收并处理这些 action，检查无重复 free、无非持有槽位 free。raw 尾部仍由请求夹具持有，不能把它们计成 cache 泄漏。

Eagle 的 raw token 数与逻辑 bigram 数不同。page 128 下 raw 512 仅可缓存 384，raw 32768 仅可缓存 32640；三请求最终各保留的未缓存尾部合计均为 384。关闭 Eagle 时，两种长度均整页可缓存。这是直接 insert 的 CPU 身份/索引合同，不代表完整请求生命周期只能缓存到该边界。

split 的 prefix/suffix 索引 tensor 以及 match 拼接结果具有独立 metadata 存储；改写返回的 match 索引未改写缓存索引。这里复制的是 KV 槽位编号，未证明 KV payload 被复制，也不证明 device Copy-on-Write 的实现。

## 完整导入失败与实验隔离

R007 仅尝试一次完整 UnifiedRadixCache 模块导入，结果为 UNAVAILABLE：`cache_controller → memory_pool → cache_move` 依赖 `sgl_kernel.copy_all_layer_kv_cache_cpu`，CPU dependency view 中没有 sgl_kernel。原始 traceback 保留在 unified-source-result.json。R008 没有重试该导入，也没有用源码类通过结果覆盖失败。

两次 CPU 容器均无 accelerator device、无网络、只读 root、资源上限与 capability drop；仅四个已哈希模型 metadata/tokenizer 文件只读挂载，无权重。任务容器均已删除，参考服务的 ID、启动时间与 restart count 前后一致，四个模型文件未变。

## 候选、成本与否定条件

本轮验证现有实现，未实施 CACHE-008 的重复 discard 省略，也未实施 CACHE-009 的插入游标改写。额外分支、状态缓存或批处理会增加维护成本；短路径、低共享或缓存冷热变化可能抵消收益。host/SWA/Mamba/session/linker、取消、PD 发送与计算并存需要独立覆盖，不能沿用 FULL-only 的结论。

最小下一步是把真实 allocator 的 page/free/available 合同接入上述树状态，再在已分配的 Ascend PD+MTP 服务中记录 receipt、split、free、transfer 完成及设备消费者结束事件。任何提前回收、漏回收、计数/身份不一致均否定候选。没有同条件 native trace 与端到端比较，不填加速比，不宣称逼近设备物理边界。

Loop 结论为 INCONCLUSIVE：源码类计数合同得到支持，原生资源与性能门槛尚未满足。详见 native-next-contract.json、source-index.json、zcode-audit.json 与两份 CPU 结果。
