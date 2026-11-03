本轮先复核 v39 manifest，再按 profile 清理108条“部分分析”链，而非扩大源码扫描。108链/66685TASK分为6类：compressed physical slot12链2276TASK，slot/page/C2 length64链47636TASK，batch snapshot6链221TASK，CP suffix9链13960TASK，RoPE10链2112TASK，DSpark first-pass7链480TASK。大部分使用与166阶段源码SHA一致的本地冻结快照，RoPE复用v34证据。远端只读取本地缺失的精确slot helper，并核对三个消费者/DeviceOperator文件SHA。

最重要的排除：已有 packed slot融合helper并非legacy直接替换。ratio1时它还施加valid_end掩码，legacy只看slot非负。例如slots=[3,5]、valid_end=1，legacy flat=[3,5]，helper=[3,-1]。这是两条源码域的区别，不证明现场出现错误值。复用需证明额外掩码冗余，或显式调整mask ABI。ratio2还要同时保留物理slot完成条件和逻辑位置奇数条件；positionsNone、generic ratio、连续输出stride和state-cache PAD分支也不能丢。坐标/flat可与slot producer重组，但保留原始slot和所有组的物理所有权。

已有共享也核实了：long/index同组slot key共享持久[N,2]/flat输出，C2 quotient/remainder本已按lengths:c2每batch缓存一次，CPglobal/local已有RoPE view复用。它们不再计为新增优化。可以分析一次producer同时出quotient/remainder、共享不可变查询几何、以及合适的producer-consumer输出ABI。

发现draft共享接口缺口：调用方构造shared_dsa_draft_cache，但只有SAS共享字典，没有common_v41_batch_metadata；DSA build_for_drafting839-842又丢弃kwargs直接调用build。只转发当前kwargs仍不足，需要明确的新batch/字段共享接口。每组sliding_window可能改seq/table，必须把真正相同的QSL/positions等字段与组相关长度、physical slots、operator banks分开。不能仅因B或request IDs相同就整批共享。

CP target与dummy的完整调用栈走相同worker共享字典结构，v13“cache hit后长度临时量无内容读者”的结构证明可覆盖这一路后续cache hits；首个local/global miss、QSL固定地址发布和组字段保留。未给dummy虚报400次实测删除数量。draft当前没有共享batch传入，fresh miss结果仍有消费者，不能把target的2200次死生产者结论直接套过去。

QLI消费者已定点定位：select_projected在物理query.shape[0]==0或max_cache_seq_len==0时先返回；否则legacy读取/wait metadata后调用QLIv3。packed格式有自己的None检查和metadata路径；packed producer仅在num_actual_tokens0且非metadata_in_graph时零化。逻辑actual0不等于capture物理shape0，已有图中的QLI节点可能仍读plan。plan省略只登记为“全部实际消费者绕过且图体无读者”的条件方案，保持frontier/events/bank和格式；实际图体/参数银行绑定移交M22。

DSpark seed/tail、已有copy-expand、seq-rejected+Q、uniform QSL和canonical CPU镜像的发布顺序也已绑清，不能覆盖仍被kernel读取的旧seq或把CPU上界替换权威GPU值。RoPEuse_cache=True仍刷新mutable runtime/spec bank；省重复查询需要输出代际/末读者，不能用缓存flag或指针直接判断命中。

本轮登记8项条件细化、27项约束。186条选中native链现均有局部字段/接口分析，但这只证明该selector的归属完整。还需将其与原M21 phase/branch/host/capture范围逐项对照，再做M22-M25及全模块有限完成审计；整体不完成，0整模块关闭。未实现/未测速不单独算未分析缺口。

保留两次本地错误：快照缺helper使准备中断；我随后误启动依赖该脚本的executor，它在本地读脚本阶段失败，未连接服务器。显式v2完成唯一一次远端CPU点读/核对。加本地点读与主审共5runs：3pass/2error。35模型候选、7框架候选；26理论比较INCOMPLETE、3136candidate资源行、849缺PMU、16phase-unbound图保持，无新模型/torch/NPU/服务/健康工作。Goal active。
