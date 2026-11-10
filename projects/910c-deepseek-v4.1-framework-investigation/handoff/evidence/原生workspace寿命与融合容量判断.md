# 原生 workspace：实际寿命与大融合容量

**范围更正**：三处审阅尾行标签超过EOF，已按实际文件长度更正；不存在行不计覆盖。[审阅v2](native-workspace-source-review-v2.json)与[原记录更正](native-workspace-source-review-correction.json)保留原Run证据。

2026-11-10，loop-033。**Goal ACTIVE，全链路未扫全；每个替换原语逐位一致。** 本轮没有实现替换原语，没有模型、NPU计算、服务请求、新profile或显式远端写入。

## 介绍

沿真实 `GetWorkspaceSize → Byte tensor → raw pointer → OpCommand → ACLNN → kernel workspace readers`，判断缩容量、跨层共享和新并行能消除什么交付。上轮DFC两个payload共bank合计960MiB和QLI源码容量差，只是入口线索，不作实际新释放显存。

## 到了哪里

### 版本和证据资格

171主机名核对成功，同自有容器/镜像。[安装/映射目录](native-workspace-catalog.json)读取到TorchNPU `2.10.0.post7`，声明Git `1d88b8608fd1e44545b5804026c7a57c44240bb4`；20进程映射同一当前磁盘 `libtorch_npu.so`，含原16个service worker。库SHA `7e5eb430fd0f1d438e9598cd89e609332d4d1eea1c458e8f42fd01060afa47a1`。

通过GitHub connector取该commit的24份原件；[审阅表](native-workspace-source-review.json)只承认所列15份文件的范围，其余为接口/定位副本。[9份安装header](native-workspace-backend-records.json)逐SHA匹配，包括allocator/config、stream、queue、graph、OpCommand、OpParamMaker、NPUDefine、OpCmdHelper。这支持“与安装包声明版本及接口一致的后端源码参考”，不证明.so全部实现或实时option对象等于源码。

16worker初始白名单均见 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；TASK_QUEUE_ENABLE、PER_STREAM_QUEUE、blocking、force-uncached和multi-stream选项未设。源码fallback分别为queue1、per-stream0、blocking0、reuse2；queue还可被运行时atomic覆盖，不能把初始环境当实时getter。

### 实际分配、排队和最后设备读者

| 交付 | 读到的实现 | 对融合范围的约束 |
|---|---|---|
| 非零workspace请求 | 实际macro `op_api_common.h:700–772`；`TensorFactories.cpp:169–221` 的NPU empty直接选CachingAllocator | 不是按函数名自动走专用WorkspaceAllocator。empty只在三个确定性flag共同为true时额外fill；没有无条件整workspace清零可删 |
| 分配/析构 | allocate用 `getCurrentNPUStreamNoWait`；deleter直接free到cache，空foreign-stream集合可立即进同流free pool | 选中路径不自动清空提交队列/等待设备。free计数、Tensor作用域、requested bytes都不是物理释放证据 |
| callback | macro捕获rawptr/size/executor/current raw stream；OpCommand直接执行或COMPILE_AND_EXECUTE入队；ExecuteParas复制std::function，再由ExecFunc执行 | 回调保留闭包，但没有补出一个不存在于捕获列表的workspace Tensor lease。新aux-stream/arena方案须明确保活和最后读者 |
| 同流缓存 | best-fit从同stream查块；allocated=false、event_count=0、stream_uses空才可合并 | 原同流已提交设备工作的先后可容许复用；不能推广为“任意线程在macro析构到enqueue之间插入分配也安全”。本轮未观测这种交错或证明现网故障 |
| 跨流 | recordStream集合在free时转events，完成后才能回收；capture可延后处理 | callback返回/host queue空不是设备全部完成。搬到新stream须保留真实producer/consumer事件与异常退役 |
| graph | capture按stream/model过滤到private pool；显式pool handle/refcount可共享；replay只ExecuteAsync记录地址 | captured节点不会每步重跑host empty/GetWorkspaceSize。host对象析构不撤销replay地址；不能先把captured scratch交给其它活图写 |
| pool退役 | reset源码先releasePool，再AclmdlRIDestroy；releasePool refcount到0通常只是标freeable，实际cache释放另走事件与release/unmap | 不能把refcount到0当最后replay设备完成；也不能从注释推成“已先destroy”。新并行仍需真实last-reader/graph lifetime证明 |
| DFC设备 | kernel入口到WorkspaceInfo、GMM1/activation/GMM2、combine、计数/soft flags，复用上轮同SHA范围 | C1/C2、A/activation本来共bank；payload末读和peer协议不能由host Tensor结束替代。减小offset需同时保存原行域、scale、旗标与原FP16/INT8链 |
| QLI设备 | 复用loop030 arch22主body/LD epilogue、输出及PA消费者范围 | 无FD payload仍有reset/init/prefetch/sync；index/candidate是后继活输出，不能与scratch一并覆盖。LibAPI/system前缀仍不能删 |

专用 `unsafe_empty_workspace(size,stream)` 是另一个实现：其grow分支还可能device synchronize再free。本macro没有调用它，不把专用allocator行为替换成当前事实。原macro scope释放rawptr的语法也**不单独构成use-after-free结论**；仍缺实际interleaving、当前binary和runtime对象证据。

### 源码容量算到哪里

DFC host实际请求为 `16MiB + max(cocWorkspace, initRoutingWorkspace)`。`cocWorkspace`包括两个payload共bank、两套scale、routing index/counts及flags；它不等于960MiB。保持原65536容量、M24/top6/24local experts时，该公式在coc占优条件约976.556MiB。

若证明**该次所有16sender行数均≤M**，不丢弃合法receiver条目的容量界可研究 `min(65536,16*M*topK)`，还须原clipping/alignment和dummy/graph/prefill资格。M24/top6的两个payload为33.75MiB，含coc其它区域和16MiB前缀的条件公式约49.824MiB；M3/top3/8local experts则payload2.109MiB、条件公式约18.131MiB。M3/top6/288只是条件示例，不声称是当前target实际shape。[逐项整数公式](native-workspace-frontier.json)保留原值。

这些数字均**不是实际GetWorkspaceSize结果或可释放显存**：inner routing可能占优；不同prefill/graph桶可能保留峰值；同流cache已复用；不同pool未必相同；VMM映射/物理handle缓存又是另一层。缩容量不自动减少实际已读写的有效行，更不自动产生HBM延迟收益。

### 旧profile能支持什么价格

只读原C8 rank0 DB并复核SHA；其中没有分配/释放或PYTORCH_API表，memory类只有memcpy枚举。整DB的CANN API里，InnerDFC GetWorkspaceSize共853次/17.398410ms，QLI主GetWorkspaceSize32次/.956220ms；2次MallocPhysical共.819940ms。**853不是选中首token的160次DFC**，API duration也不是allocator独占时间或返回字节数，因此不给请求净省、不乘16。该缺失仅针对已检查rank0 DB，不声称所有档案都无memory证据。

继续使用上轮匹配2560任务/16rank和8组typed成本的**大DFC**条件价：共同尾段全免的请求P90信用20.090%–22.690%；若相关且暴露阶段占25%、其中半省、每层新增5us，条件净信用2.458%–2.783%。半省/h5us要到1%需该阶段占共同尾段9.29%–10.49%。这使“容量支持的更大调度/融合”值得保留，但没有把容量减少兑换成该收益，也不与QLI/其它parent相加。

## 结果

| 归类 | 项目 | 结果/具体缺口 |
|---|---|---|
| 已停止 | 960MiB或QLI声明逐层/逐卡相加作为新释放VRAM | 已有bank别名、同流cache、private pool；缺独立物理峰值，不成立该计法 |
| 已停止 | captured decode按每native每步省malloc/free/GetWorkspaceSize | replay不执行这些host节点；该重复CPU信用0。eager/capture miss另作范围 |
| 已停止 | scope结束或callback返回即可覆盖全部workspace | 不满足真实设备最后读者/graph寿命；不能作alias资格 |
| 待确认 | 分桶/全sender有界DFC容量 + 大段共驻/megakernel | 缺各eager/padded/dummy/prefill的全组M/clip、实际ACLNN/inner scratch、全流/peer末读、pool地址与物理峰值、逐原语位一致和争用净价。原24/48不提供空闲4/8 |
| 待确认 | QLI scratch contraction及稳定地址/selected-panel计划 | 缺LibAPI与实取workspace/tiling、LD真实touch范围、graph桶union及活输出代次；保留大QLI/SMLA联合parent |
| 待确认 | host query/cache/分配合并 | 缺实际eager/capture/rebuild命中、executor与参数代次/独占暴露成本；不从整DB API总量直接外推请求 |
| 已确认可实现 | 本轮新增 | **0**。本轮是源码/寿命和有条件价格判断，不是新原语验收 |

下一处未闭合交付：SMLA metadata AICPU输出1024×INT32包→arch22分核/FD workspace/实际末读者。重点读producer分核和合法deterministic/空KV/ori_sparse条件如何限制selected-KV跨层共享与更大Attention融合；不能用小metadata预算剪掉cache→QK/PV parent。已知缺口留在地图，继续走真实路径。
