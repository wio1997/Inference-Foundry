# 已融合MC2内部依赖复核

实际采样出现`DispatchFFNCombine`，与当前`enable_fused_mc2=1`自定义分支对应，不能直接叫作CANN MegaMoE。源码入口：`ops/fused_moe/moe_comm_method.py:563-611`；底层已冻结来源为`csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp`。

## 源码已经支持的事实

- AIC分支执行GMM1、等待V2C flag、再执行GMM2；AIV分支执行DispatchAndCombine（213–234行）。这已是计算、激活/量化和通信协同的复合kernel。
- GMM1开始等待AIV完成cumsum；按expert循环中，还需等待相应expert的数据/flag（435–538行）。这些flag真实存在，但不能推断为所有数据必须先全量搬完。
- 当前实现按epilogueGranularity和末尾两个发布点通知SwiGLU；AIV侧等待C2V并执行SyncAll，然后发布V2C给GMM2（约875–970行）。这是当前分块/同步策略的事实，不是任意合法新实现都不可避免的整阶段屏障。
- 部分flag轮询用ReduceMin观察相关rank的完成状态；实现也依据expert行数调度核心与tile。跨rank到达不均、expert工作量不均、等待及小M tile效率均是合理竞争解释。
- 对小expert行数，GMM1源码设置权重L2 cache hint为disable。只有确认具体case满足条件、且调用该实例后，才能用于解释其缓存行为；不能用它推断其他独立Wq_b的HBM流量。

## 实测能说明与不能说明的范围

已有样本每rank该算子时间并集约622.8–754.7ms。其reported MAC流水ratio按task时长描述性加权约10.65%–13.52%；同一算子内部还包含访存、激活、路由、通信及等待，低MAC ratio不能直接解释成INT8指令慢。

目前能支持“有这些同步及数据到达依赖”，尚不能确认哪一个原因占主导、占多少微秒。需要把同一调用的参与rank、任务开始/完成、expert计数或内部发布点对应起来。不能将不同rank/不同shape的累计时间差直接算作可消除通信时间；也不能把所有非MAC周期视为浪费。

## 理论模型处理

理想model允许按已就绪的最小可证明数据块生产/消费，并先放宽通知和实现屏障成本。只有证明量化归约、必要数据到达、资源容量或因果链对声明域内所有合法实现都成立，才将其加入时间下界。当前epilogueGranularity、expert遍历次序与SyncAll只可作为受限实现域约束或诊断事实。

状态：源码机制存在为SUPPORTED；上述机制是否主导当前耗时为HYPOTHESIS。没有设计修复策略，没有修改任何内核。
