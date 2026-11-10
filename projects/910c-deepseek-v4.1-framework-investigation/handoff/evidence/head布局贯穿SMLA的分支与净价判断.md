# head布局贯穿SMLA的分支与净价判断

Goal ACTIVE；loop039只完成有边界的源码、标量计划与条件价格判断。全链路未扫全，每个替换原语逐位一致。本轮无新增原语准入。

## 介绍

沿较大的交付：原Wq_b全64head→query RoPE→CP局部query SMLA→inverse RoPE→transpose/A2A→TP O投影，考虑改为每卡8head、全局24query，直接交付原TP输入。保留原量化/反量化舍入，不把35MiB名义权重读范围差当HBM流量、显存容量或请求收益。

## 到了哪里

[源码](attention-head-layout-source-review.json)分开记录17个当前SHA匹配和实际已读行段；[原标量计划](attention-head-schedule-cpu-a3/result.json)执行25个显式输入，保留900个定义packet字和124个未写哨兵。首次driver编译错误、第二次验证器错误原件仍在a1/a2，均非原生正确性失败。CPU改正后通过，也不是NPU逐位准入。

压缩CSA在默认策略下mBase=G：原每CP卡3query、64head，FA3；新全局24query、8head，FA24，都无FD/S2split。每个query的head分片仍会重读KV。K512、SWA128、D512的均匀参考中，QK/PV直接读SWA，压缩KV另经GM merge，每卡逻辑GM交付从6.75MiB到54MiB，差47.25MiB；扣35MiB名义Qb读范围后仍多12.25MiB。此处是源逻辑交付，不是HBM交易量或8倍耗时；FA并行度也增加。原有效query-head行数192不变，但Cube M向16对齐，192变384个padding行。

前两目标层走独立dense-SWA kernel：两个代表rank各480目标样本中24个纯SWA、456个CSA。dense mBase256，原FA1、新FA4；mask union的源坐标为原130行、新每request133行。不能套CSA的8倍访问或2倍padding。该分支完整N-buffer搬运和原生代价仍待确认。

CP device长度减suffix的因果mask坐标在20帧、260个query坐标相同，原restore_tp_heads与新TP输出的786432个地址坐标一致；这些仅是整数坐标身份。MMAD/SoftmaxFlashV2数值未比较。head8会改变softmax动态行tiling，不能凭“head独立”判BF16逐位。原global cache builder还明确build_query_metadata=False，不能复用为globalquery packet。CPU seq_lens是上界，须保留device权威值。selected-ID源更新与层间复用已追到真实共享bank；新全局收集仍缺epoch、末读、空DP、capture和collective顺序证明。

## 结果

**待确认**：贯穿SMLA到TP O投影的较大布局候选仍值得保留，需common globalquery plan、KV/L1复用或足够并行/带宽余量、NZ切片owner、原生逐位和完整请求净价。只留下具体缺口，转扫更大的共同metadata交付。

**已确认可实现**：本轮0。原标量packet和坐标检验不计替换原语通过。

**已停止**：只改flag借用cache-only packet；将所有Attention层套用CSA流量倍数的估价。没有停止更大共同plan或保持原序的资源重排。

[条件成本](attention-head-layout-delivery.json)复用loop038不可变价格：Qb新耗时取旧1/4，新通信各按旧较大AG参考付费，每交付另付5us，SWA不省时，原条件净为窗口1.228%/1.299%。若仅以1%窗口作敏感性参考，新增SMLA退化及metadata总余量仅4.473/5.795us每目标交付；其它新增为0时，SMLA最多约增9.66%/12.48%。若同时SWA减半，余量为32.234/33.370us。上述都不是预测、完整请求验收线或实测收益，不能跨新旧profile窗口换分母。

下一段从已完成idle-attribution的prepare input出发：跨组协调结束到首metadata提交约31ms、每迭代243where、与Free重叠约27ms。先追同代CP区间、长度、suffix、mask到各资源组和真实消费者。31ms并非精确函数或全可删时间，也不重复认领旧CPU-plan/25次H2D或已融合slot mapping收益。

本轮远端仅只读源码/旧数据，本机执行标量C++和整数/价格分析；NPU、模型加载、推理请求、新profile、显式远端写入、服务动作均0。
