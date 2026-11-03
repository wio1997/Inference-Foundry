本轮按“算子能否融合、理论上限是什么”收敛取证，没有扩大为框架研究。

保存的profile中，Down、Q_B、IndexQ共22种BF16输出规约、9,671次执行均有Vector工作；GateUp的9种INT32规约、4,472次执行没有Vector工作。规约与PMU一一绑定，未改变原有模块归属和不确定性。

固定commit的[QuantBatchMatmulV3 per-token算子源码](https://gitcode.com/cann/ops-nn/blob/a61e016a71f422cb280af0c024975f13629ca100/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_opt.h)显示：Cube写出INT32，Vector读取并反量化，最后在UB内转成BF16再写出。这提供了“BF16转换后接普通Vector加法”的条件路径。公开分支尚未绑定当前CANN9.2二进制，profile也不能单独确定具体转换指令。

融合应保留Down的BF16舍入，再执行原加法及其舍入。有限FP32/BF16-RNE模型中，d=257、R=-256时，原顺序结果为0，直接在首次BF16转换前加R则为1；把257和-256两个INT32部分和先分别转BF16再累加，也会把正确的1变成0。这些是数学反例，没有执行设备测试。

在原INT32中转保留、S没有未吸收读者、R已就绪且结果Z须写GM的域内，普通Vector融合可省掉S的一次BF16写出与读回，预算为4×M×5120字节：M=3为61,440B，M=800为16,384,000B。Down局部边界含内部INT32中转时，逻辑值搬运预算由16n变为12n，读写各由8n变为6n。它们不等于HBM流量或已实现的提速。

[SetAtomicAdd文档](https://asc.gitcode.com/api/SIMD-API/basic_api/atomic_operations/SetAtomicAdd.html)给出A3的BF16及UB→GM支持，但不会自动初始化目的值，且2201存在设备级BF16饱和模式。原子方案仍需匹配原加法的数值语义、旧R读者、目标bank及完成顺序。先复制R到新Z再原子加，额外复制成本可能抵消上述逻辑流量节省。

[Fixpipe的A3类型表](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToGM.html)未列出INT32→BF16，不能把其他代际的能力当作当前INT8 Down的直接替换。该结论限定于公开API，不是硬件不可能性证明。

理论模型按同一资源的必要工作推导，并允许资源重叠。令n=M×5120、K=2304，在上述固定实现域内，T≥max(2Kn/C_INT8,6n/B_GM读,6n/B_GM写,12n/B_共享〔若证明共享容量〕,L_Vector,L_真实依赖)。实际容量上限缺失的项保留符号；公开752TOPS、1.6TB/s只作参考，不能据此声称实际MFU或可达提速。

本轮更新5项既有候选，增加F21/F23的目标bank与就绪关系，台账为47条两两关系、11组多候选关系，中央尚有548个两两关系待汇入或复核。342条模块分支、24个语义模块及原有profile缺口均保留，整体任务尚未完成。
