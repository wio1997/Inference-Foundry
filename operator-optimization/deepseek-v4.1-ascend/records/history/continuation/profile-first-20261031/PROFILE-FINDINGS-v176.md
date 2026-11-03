整体继续 PROGRESS，未将全部目标标完成。原服务已恢复，health=200，启动脚本SHA一致；启动时间为2026-11-03T01:17:47.463945252Z。全部提交的控制器和CPU分析handle均已终止。

本次原生Execution Trace合并采集完成模型加载/图创建，并执行部分冻结C8迭代，随后触发单worker768MiB停止阈值。完整C8校验失败，没有16份FINAL，也没有完整输出对照；不把诊断时间或部分结果当性能基线。此前有效profile继续承担性能证据。版本复制时的诊断hash错误、一次额外我方容器启动重载和原服务恢复均有独立记录。第一版CPU提取在写出完整邻域后因64MiB检查失败，保留产物并仅用CPU完成汇总，不重跑NPU。

从未闭合顶层的ET文件中逐节点恢复EP0的821146个完整JSON节点、235种公开/运行时op名称，5362个绑定marker与侧录匹配；最后4个sample_tokens回调位于step71/72，保持未匹配。没有人工补文件尾，没有由ET ID推出物理分配或retirement顺序。全局storage-ID邻域只是过近似，诊断RF引用也不是基线必须保留的消费者。

关键算子证据有四项。第一，16worker各69次完整拒绝采样入口/返回，共1104调用，实际参数为all_greedy=True、all_random=False、no_penalties=True、max_num_logprobs=None、synthetic=False、entropy=False。支持这批调用的greedy专门化前提；仍保留score其他读者、argmax ties、correction、输出单元和RNG契约。准备态双True flags不能当实际随机需求，观察窗口也不外推为全Case所有分支。

第二，目标和草稿的topk_indices_buffer在capture/formal_start两次目录中指向相同原始StorageImpl与地址，INT32[8192,512]共16MiB。它们是共享状态，不能按独立bank安排写者，也不能重复计两份显存。是否可省写仍取决于发布、读取、回滚和代际，分配尺寸不是每次HBM流量。

第三，目标/草稿embedding和head四个BF16[16160,5120]参数分别绑定不同storage对象/地址。未读取数值，因此不声称数值不同；同shape同样不能支持值CSE。共享输入、多输出投影是另一种组合，仍需计全部必要权重与计算。

第四，C2mask有73个命名视图绑定同一个8192B持久backing，出现长度24、1704、3200、8176，并有73次source-level copy。BOOL[24]不是固定24B allocation，也不代表恒定值或代际。可以在公开语义边界融合最终写者，但不能因指针稳定而宣布write-once或删除所有写入。

已对27个声明源码模板、854节点、1346边完成7条融合/拆分/共享/跨模块/多消费者规则的一轮条件语义检查，无新增非等价候选族。保留各模板特有的数值舍入、HC fork、QR/QS区别、SMLA重复index、多输出、共享bank/RNG/通信/迭代状态约束。该轮覆盖声明图，尚不能仅凭数量或partialET宣布完整模型G0及有效callee全集已闭合。

此前1101静态点的算子效应、42候选共同组合约束、94条P5比较均直接复用，未重复读模型源码、做笛卡尔配对或追查物理fmax。当前EP0参数目录有49种module class、3108身份记录，注册module hooks均为0；这不等于所有native/其他观察者都不存在。

只保留一个收尾范围审计：把26模块360语义行、源码/调用帧清单、实际class/op目录及27模板逐项对齐，列出仍真正缺算子条件/消费者/别名/跨迭代答案的具体ID，核对G0和effectivecallee完整性。已有明确条件答案、历史epoch、物理上限未知或未实现提速不继续当通用阻塞项；确实未分析的范围也不能因报告成稿而关闭。没有默认的新NPU采集计划。
