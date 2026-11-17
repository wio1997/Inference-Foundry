# 910B 原生低压缩比 target/draft 池组合

G226在固定A2镜像的八张910B1上，以完整发布的ServerArgs/RuntimeContext和checkpoint的text_config构造40层target、3层draft池。保留原生DSV4NPUTokenToKVPool的源层归属与FP32 request-scoped pending-pair状态工厂；为低压缩比工厂接上G196已验证的68 bytes/slot原生FP4 Indexer，避免额外分配旧C4专用INT8侧缓冲。

C1/C2的KV继续以BF16 PA_ND存储，逻辑行1024bytes。C2物理页为128行，一个FULL页只对应64行，因此传输item是65536bytes，不能把物理页131072bytes当作一个FULL页。无拷贝逻辑页视图保持原始data_ptr。C1 Indexer每FULL页对应两个64slot页，共8704bytes；C2对应一个64slot页，4352bytes。四个source层产生八个main传输区；所有consumer与其原生source共享同一块KV。

八卡分别完成C1、C2的非零FULL页1/2/3的KV原始字节切片检查，并通过完整池访问器读回原生FP4结果，与CPU原生fake_quant_fp4逐元素一致；共16个案例、16次图捕获、48次变更输入重放。确认三个C2 pending-pair池使用FP32且DSPARK6的请求环为8。target和draft实际分配了池；没有装载模型权重或执行attention，也没有启动跨机传输。

- [验收范围与实际区域字节数](../evidence/native_lowratio_pool_G226/acceptance.json)
- [rank0原始结果、实际source SHA与checkpoint配置SHA](../evidence/native_lowratio_pool_G226/evidence/rank0.json)
- [controller终态和八卡进程清理](../evidence/native_lowratio_pool_G226/state.json)
- [复用原生池的工厂、访问器与逻辑页描述](../code/native_lowratio_pool_G226/lowratio_pool.py)
- [完整上下文与字节、图检查入口](../code/native_lowratio_pool_G226/probe.py)
- [原始G222配置层级读取错误](../evidence/native_lowratio_pool_G222/evidence/rank0.json)

G222在从原始配置外层直接读取compress_ratios时失败，尚未分配池；G226改为读取text_config，并在使用前核对实际固定镜像的两个native pool源码SHA。原始失败与清理证据保留。

本次full_size=8192只用于池的地址/页契约验证，不是完整263296上下文容量验收或性能测量。选定工厂的低压缩比Indexer为68bytes/slot，因此旧G94把未使用的INT8侧缓冲计入198bytes/slot的候选不得直接部署；完整预算还需加入实际BF16存储和保留页。完整服务的C1/C2 source/compressor/indexer候选语义、attention后端、DSPARK接收回滚、跨机PD所有状态、实际目标/草稿完整图和四档性能仍未验收。

## 选定工厂的预算重放

随后按G226实际选定的BF16 PA_ND与原生FP4池重放原生计价方法，并对八卡原始分配的8个区域逐项核对整数预测，全部一致。保留页开销使用实际物理128行KV页和64slot Indexer页计算，不能按裸行系数省略，也不能继续计入没有分配的旧INT8侧缓冲。这里是现有硬件分配证据上的CPU算术核验，不是把客户端CPU结果当成NPU性能。

- [逐卡实际字节与整数预测、系数和padding](../evidence/native_lowratio_pool_G226/selected_geometry_CPU_replay.json)
- [尚未部署的限定范围原生预算overlay](../code/native_lowratio_pool_G226/kv_geometry_g254.py)

该overlay尚未安装到完整服务。完整SWA/request-state/draft预算和263296上下文的实际模型池分配仍须在完整引擎核对，不能由此次小池地址契约测试推定可用容量。
