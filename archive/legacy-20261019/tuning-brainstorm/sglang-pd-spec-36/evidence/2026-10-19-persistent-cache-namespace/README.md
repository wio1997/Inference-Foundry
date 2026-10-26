# M036：持久存储名称、清空确认与模型生命周期

本轮固定 SGLang `ce0a8fea5f4245839d174b2b8c55692be1810a7f`，哈希核验64个源码文件，其中新增收集10个存储/更新依赖。实际原 File/LRU 在任务独占目录完成28组295项 CPU/文件检查：参考278通过、17失败；controller 返回值候选修复9项清空确认失败，**8项模型名称别名反例仍失败**。继续向上执行原 Scheduler 消息处理器后，另10组38项出现5项错误成功响应；完整确认链候选使这38项全部通过。**候选只获清空确认链的局部 CPU 条件支持，完整 namespace/世代方案未接受、未部署。整体 E0、accepted baseline=null，无原生或性能结论。**

## 实际存储如何区分身份

原请求 extra_key/cache_salt 进入 storage seed，但后端还添加自己的模型/并行标识。File 将 model_name 的 `/` 转为 `-`，再按 MHA/MLA、TP、PP、CP 配置构造文件后缀。Mooncake 构造函数也规范化模型名，并与 extra_backend_tag 拼接。NPU MemCache 的原 runtime-fields/tag helper 使用 rank/并行后缀与显式 extra_backend_tag；这两个 helper 没有把 model_name 加进 tag。[File 后缀](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/hicache_storage.py#L383)、[Mooncake prefix 源码](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L427)、[NPU helper](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/storage/npu_memcache/npu_memcache_store.py#L395)。

因此上轮“外层隔离未知”现在有了后端差异证据：已有模型/tag/rank 隔离，不能说全部持久层均无 namespace；这些可配置字符串也不能自动充当权重、布局、KV dtype 或代码版本的独立世代。

本轮完整导入原 HiCacheStorage/File/MetadataCache 模块及原 LRUFileEvictor，实际运行 get/set/exists/clear 和文件创建、原子替换、重建实例读取。固定输入文件为512和32768个 token，**整个输入 ID 数组的 SHA256 只是提供给后端的外部 key 夹具**，未运行原生 page-hash 链。每个原始文件 payload 是128个 int64 CPU 值，不是设备 KV，也不是缩短后的服务 benchmark。MHA/MLA 配置 × metadata cache 开/关 × 两种输入，共8组配置，各做名称别名、正常不同名称、删除错误三种情形，共24组；另4组覆盖返回类型、attach、NPU helper 和权重更新回调。

## 真实文件中的条件反例

两个同目录、同拓扑、同 incoming key 的 File 实例，model_name 分别为 `org/A` 和 `org-A`，得到相同后缀。后者实际读到了前者的旧字节，共8项隔离合同失败。`org/A` 与 `org/B` 的对照不命中；MHA 不同 TP rank 分离，MLA rank 共享符合原后缀合同。[参考结果](file-reference-result.json)。

反例前提是两个名称代表不同语义模型，却共享该 File 根目录、key 和兼容 payload 大小。若部署合同明确这两个名字是同一模型别名，或使用独立目录/已隔离的 key，该反例不证明其部署有错。当前原 GLM-5.3 的后端、名称、目录和真实业务均 UNKNOWN。本轮不将 File 读到旧 CPU 值写成 Ascend 服务输出错误，也不把仅源码追踪的 Mooncake prefix 写成已运行远端读写。

原 File.set 对已有 key 返回成功并跳过重写；新实例能继续读到旧字节。此行为符合不可变内容键的做法，要求调用方在权重/布局等语义变化时改变有效 namespace；本身不表示 set 违约。File.clear 则遍历整个配置目录的所有普通文件，包含其他模型后缀。本轮只在新建的任务独占目录验证该范围，正常重试后全部测试文件已删除。没有对外部目录执行清空。Mooncake/NPU 的 clear 调用 remove_all 仅作源码记录，未实例化或调用原生 store。

## 清空失败如何被两层吞掉

在一个已知任务内文件的 os.remove 上注入 OSError，原 File.clear 捕获错误并返回 False，文件及旧 payload 实际保留。原 HybridCacheController.clear_storage_backend 不检查后端返回值，随后返回 True，原 StorageAttachment.clear 因而报告成功。8个文件错误情形和1个显式 False 返回控制均复现，共9项失败。[原 controller](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L379)。这是明确的故障注入，不是声称磁盘实际损坏。

[第一版离线补丁](candidates/L037-clear-ack/clear-ack.patch)让 controller 对显式 False 返回失败，同时保持 None 类型后端和 True 的成功语义。295项同条件检查中仅9项确认谓词改变，8项名称反例仍失败；原 File/LRU、IO 和 namespace 未改。

源码继续追到原 Scheduler.clear_hicache_storage_wrapped：它同样忽略 tree_cache.clear_storage_backend 的返回值，只要 hierarchical cache 开启就产生 success=True。[原消息处理器](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/managers/scheduler.py#L4891)。因此本轮增加了实际 File → patched controller → 原 Attachment → 原 Scheduler 的组合，使用原 BaseReq/ClearHiCacheReqInput/Output 和真实 msgspec msgpack 往返。10组38项中5项失败，覆盖文件错误、storage disabled、False 和被下层捕获的异常；下层已经返回 False，上层仍输出 True。[处理器参考](handler-reference-result.json)。

[完整确认链补丁](candidates/L037-clear-ack-chain/clear-ack-chain.patch)再把 tree_cache 的布尔结果传给 Scheduler 输出，10组38项全部通过，仅改变这5项谓词。成功清空、feature disabled、None/True 后端兼容和真实结构序列化保持通过。[候选结果](handler-candidate-result.json)、[两组逐项比较](correctness-comparison.json)。没有运行 HTTP、消息分发进程或 TP/DP 一致性；这不是完整 API/服务验收。

新增成本仅为返回值检查与分支，没有增加 IO 或扩大清空范围。补丁不提供并发写入屏障、原子跨副本删除或世代协议。Scheduler clear 这段源码也没有 attach/detach 所用的 is_fully_idle 检查；本轮没有并发测试，不能把成功布尔值当作全局失效证明。未叠加上轮 JSON Req 或 slot 身份补丁。

## attach 与模型更新边界

实际原 StorageAttachment 在已启用同种 backend 时只更新策略，忽略新的 served_model_name/extra config，不重新 attach；不同 backend 要先 detach，非法策略先拒绝。原 NPU helper 配置实验表明，仅改 model_name 不改变 tag/helper 后缀，显式改 extra_backend_tag 才改变其 key；constructor、memcache_hybrid、外部服务和 NPU 均未执行。

权重更新函数依据 flush_cache 开关调用内存 flush，并另行记录 weight_version。实际执行的只是这两个原回调调度方法，回调为明确 spy，没有加载权重。源码 Scheduler.flush_cache 重置内存树/池，Unified reset 调 controller.reset，后者停止再启动存储线程并清队列；这些已读路径没有调用 backend.clear，也没有在 record_weight_version_change 中更新 storage namespace。**不能据此声称所有更新入口或外部 backend 均无版本隔离**，也不能把一次内存 flush 或版本日志当作 L3 世代轮换。[flush](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/managers/scheduler.py#L5170)、[权重更新调度](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/managers/scheduler_components/weight_updater.py#L117)。

## 证据范围、记录与下一步

R001/R002 固定源码收集 PASS；R003 28组295项、17失败；R004同覆盖295项、8失败；R005独立的上层处理器10组38项、5失败；R006对应38项通过。没有 INVALID Run。两组比较分开报告，不把参考/候选重复相加。6条冻结命令由 ZCode 简单执行器各执行一次，CLI0 与内层0/0/1/1/1/0分开审阅，输入哈希均未变。[审计](zcode-audit.json)。本地 Torch2.8、msgspec0.19 与 Python3.12 为 CPU 环境，安装目标镜像未复核。

下一步需要冻结一个跨 raw-tree/slot/L3 的语义 namespace 合同：配置模型身份、权重/量化/KV dtype、布局/拓扑、encoder/adapter，以及 caller 不可控制的缓存世代分别由谁提供、验证和传播；新旧实例/目录/外部 tag 如何隔离、退役和回收。保留 L035 的384索引旧缓存反例、此轮8项名称反例和原生门槛，先在独占副本检查迁移，禁止将共享目录 clear/remove_all 当作默认修复。

任务44/48资源仍未确认，精确原模型 config 恢复哈希不匹配、实际 KV dtype、特性业务、PD 拓扑和当前 MTP 参数仍未闭合；历史 NEXTN3/1/4仅参考。必要的重算/存储/删除工作与性能下界均未校准。**L037 INCONCLUSIVE；ACK 局部条件通过，完整 namespace 和服务候选未接受，整体 ACTIVE/E0/null。**
