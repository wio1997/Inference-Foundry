# 原生 FP4 Indexer 池：写入、读回与完整图

固定 A2 镜像的原生 Triton FP4 打包和写入核可在这批八张910B1上直接复用。G196通过原生 DeepSeekV4IndexerPool 分配68 bytes/slot的页缓存，保持每页先payload后UE8M0 scale的原生布局；set_index_fp4直接委派给原生方法。读回仅提前把反量化常量表驻留NPU，保留原生取字节和scale恢复计算。

八卡分别完成16、96、8192行测试，共24个数值案例、24次图捕获和72次变更输入及slot重放，全部BF16读回与固定镜像原生CPU fake_quant_fp4逐元素一致。另一层缓存和保留空槽均保持原值，终态确认全部八卡进程清理。

此前候选全部保留：G184在捕获期间发生常量表H2D，GLOBAL图禁止该操作；G188新增的packed-scale右移广播出现输出形状错误；G192复制原生方法时漏导入Optional，尚未进入算子便失败。G196恢复原生取字节路径、补齐类型导入，并复用原生写入核，实际通过。失败均不是完整模型性能测量，不从这些试验推定CANN通用FP4能力。

- [通过的rank0原始结果](../evidence/native_index_pool_G196/evidence/rank0.json)
- [通过的终态和八卡清理](../evidence/native_index_pool_G196/state.json)
- [原生写入委派与常量驻留改动](../code/native_index_pool_G196/fp4_index_pool.py)
- [页边界、另一层及空槽检查源码](../code/native_index_pool_G196/probe.py)
- [捕获内H2D失败](../evidence/native_index_pool_G184/evidence/rank0.json)
- [补偿实现的广播失败](../evidence/native_index_pool_G188/evidence/rank0.json)
- [探针类型导入失败](../evidence/native_index_pool_G192/evidence/rank0.json)

这里没有把NPU旧C4 Indexer的INT8侧缓冲当成V4.1 FP4语义，也没有更换驱动、CANN或二进制。这个池的实际数据格式是68 bytes/slot，后续完整KV预算需要按实际工厂分配核算。完整NPU模型池的force_fp4工厂接口、C1/C2 KV访问器、FULL页到物理页字节描述、pair状态与PD、Indexer候选层选择和完整模型验收仍待接入。本轮没有端到端性能、波动或收益结论。
