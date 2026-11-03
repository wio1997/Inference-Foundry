本轮完成剩余28处 registry、workspace、proxy 和诊断分支的定点数据流核对。累计 graph 文件55处静态 guard 均已本地分析，加上前轮 runner 的14处核心分支，共69处；不再把这28处作为未分析范围。原 M22 两文件142units、1101个 source 关联导航位置和16个语义范围保留，整体8项mandatory gates仍未关闭。

关键结论是复用键和持有关系不能简化成 token 数：GraphParams 的 events、workspaces、handles、attn_params 四张 map，分别属于 main、draft、draft prefill 三个命名空间。相同 num_tokens 不代表相同图、任务、层、参数或代际；独立的 Python 容器也不证明不同显存或 native handle。初始化只建立 dict、空列表与 None，不在这些 helper 中分配 GPU workspace，因此没有将其算作待消除的显存分配。

workspace updater 在 registry未创建时静默不操作；创建后可新增一个 workspace key，却不建立对应 events/handles/attn_params 或 graph entry。纯Python模型构造了这一不完整 bucket 反例，以及不同命名空间引用同一存储的反例；重复 initializer 会报错。动态 bucket、双 bank 或跨阶段共享需要完整初始化、配对及生命周期协议，不能只看 workspace key。weak_ref_workspaces 只替换非None workspace 引用，不清空其他表，也不证明内存可释放或存在设备copy收益。

又排除了两种重复收益计算。在同一个稳定 backend 的普通 FULL wrapper/runner 组合中，use_updatable_graph 为真时 legacy update helper立即返回；两种更新路径是互斥选择。在构造后 ENPU flag一致的条件下，wrapper 的显式 stream同步需要非ENPU，runner 的显式同步需要ENPU，也不同时选中。256种布尔赋值均验证该逻辑；它们不是实际调用计数，不说明其它 nested/native/event同步不存在，更没有得到同步删除收益。必要的 CPU参数改写 host完成、native参数epoch和GPU ready/last-reader要求均保留。

本轮没有新增候选或变体，仅为前轮3项 graph-generation 条件变体增加14条约束。proxy 委托当前 runnable，DEBUG影响缺属性的错误信息；文本错误识别也不说明实际发生过硬件或stream故障。没有提出升级或变更服务。

验证登记3项全部pass：同SHA的14个小函数/声明、28处分支及registry模型/256逻辑赋值/协议图、原范围和8gate一致性审计。本轮没有远端登录、模型/torch/NPU/服务/健康操作。前轮529次命名 metadata DIRECT/0GRAPH 的推导保留；实际 bank、任务/层/event/handle配对及native代际仍未获得运行时证明，16次图阶段仍未绑定，166鉴权问题仍待恢复。

下一步用保存的2请求12tokens/4请求24tokens、有效capture[8,16,24]、DP2/TP8/spec5与已有DP同步合同核对 worker 的图模式/全局padding选择，再仅补未解分支代码。不能只凭本rank行数推断全DP图bucket或按执行序号补16次未知阶段。之后继续capture/warmup、原worker邻接责任范围、实际graph参数bank/全部读者、同分母资源上界、完整进程transport及其余模块。

35模型候选、7框架候选、0整模块关闭；26理论比较INCOMPLETE、3136资源行、849缺PMU、16phase-unbound保持。Goal active，未以局部源码核对或报告成稿代替全范围完成。
