第15代完成一次真实 Q06 shadow-only 短诊断及失败边界审计。整体任务仍未完成，待主会话验收。

真实现场：EXPERIMENT_HOST/aarch64，原 Ascend910B1 物理4–7、TP4，原 SGLang/CANN9.0镜像和权重；bucket1、512/C1/out32。controller PID1546184/start_ticks515407354 非阻塞取得原 performance-workflow.lock 后才操作。新代独立 runtime/manifest 的源码 SHA 全部检查，原文件未改；原数学、调度和状态写入路径未干预。未运行正式矩阵或新 SDK 导出，未采新 native profile。

真实请求返回 HTTP200、prompt512、completion32、finish length、[DONE]/EOF；新观察器15个进程均有显式 EOF close 回执。4个真实TP rank按原probe torch.distributed.get_rank 显式映射（容器内PID171/175/179/183 -> rank0/1/2/3），各加载4个SHA-bound目标模块；观察器错误均0。

完整采集验收失败：每rank的16384条环形缓冲被空队列preplan和普通planner-return记录挤满；丢弃前缀分别12494/12492/12496/12488条。controller在第92行严格无丢失断言失败，不能把此诊断标为完整通过。此为观察器记录设计缺陷，不是模型/优化机制被推翻。全部原始输出保留，142文件/12707757字节已下载逐文件SHA验证。独立审计只接受连续留存后缀。

在每rank留存后缀中，逐字段核对33个pending对象/launch关联，32个具有完整进程/RID/pool epoch/slot/generation身份；末次一个关联在正常release后slot变null，按身份不匹配保留。每rank都有一次committed31/limit32、queue_depth1且完整身份匹配的preplan观察。pending-token下界、copy完成、GDN/cache内容和复用安全仍未知；decode计数不作token下界。当前API request ID到engine RID本代未审计，不按唯一请求/数量/时间补连。没有更改调度，没有优化采用或质量/greedy/性能结论。

清理：controller在失败前已执行15进程collector thread-disappearance断言，但未逐条保留原/proc返回stdout，因此这部分仅有源码/控制流间接证据。EOF cleanup/records/thread身份文件存在；最终独立docker top仅原sleep，container/image一致且StartedAt更新为2026-10-26T17:16:20.58077704Z。controller已退出、原performance锁无占用。本轮不终止原pipeline/CPU审计，不释放原8卡逻辑归属。

修正准备：next-candidate/shadow_runtime.py只聚合健康空队列preplan和普通planner-return计数，保留非空/未知队列及错误/丢失遥测。10万次空闲迭代保留constructor证据、0drop；四份真实留存后缀离线重放各16384→272语义条目并逐字段不变，0新drop。9个过滤用例、20个bootstrap/collector CPU用例及9个身份审计对抗用例通过。这些是CPU/离线结论，缺失前缀未恢复，候选未部署。源runtime/实际controller保持冻结，未同因重跑。

原native204单次新快照：90 rank verified_archived；pipeline submitted88/inflight4/pending112，计数差异保留。原4068537/7390/7391/18358身份与既有记录一致，后级继续等待；gen82已terminal、专属SDK容器停止、slot resumed。不是最终204归档SHA验收。

下一步：新代先看native204；未ready则复用本代真实源码/启动证明，建立新独立稀疏观察器plan（不得覆盖本代runtime），CPU验证绑定和每条/proc线程检查持久化，再在原锁下仅补一次无丢失shadow诊断。所有优化门控继续fallback。若204 ready，优先实际204 SHA闭合及gen8–15显式overlay/repackage/browserQA；私有main/index归档、R分配及root最终接受仍待办。
