本轮核对 routed expert 与 shared expert 的输入量化共用条件，限定在算子输入、内部路由量化、分区和表示成本。

Profile中MC2共4472次执行、5种输入组合：target为TopK=6、draft为TopK=3。3200组图内MC2与shared DynamicQuant已按原生TASK连接，复用6400条旧TASK，其中2560组有阶段归属、640组仍阶段未知。当前shared量化在同一记录stream的MC2之后；把它变为共同生产者，需要改变提交和就绪关系，不能直接把该kernel耗时当作收益。

冻结host给内部路由量化传入40个AIV分区预算、UB=196352B、quantMode=1、scaleDim0=0。外层launch由平台CalcTschBlockDim单独设置；profile的blockNum=24、mixBlockNum=48不是内部量化预算。最初把两者直接对应的检查已纠正，未据此判断版本冲突。父算子源码与原166已保存副本的8个文件哈希一致；这仍不是加载ELF和内部tiling参数证明。

full-load和gather的无smooth全列分支，在每个core区间中对同一个token量化一次，然后复制到该token的多条路由。区间边界若切开token，就会重复量化。按冻结源码推导，3行/TopK6为18次行量化，3行/TopK3为9次，2行/TopK3为6次；816行/TopK6为836次，834行/TopK6为834次。五种组合按调用次数加权，源程序参考862040次，对比全部输入行各一次的799560次。这些是条件源码循环量，不是已实测的内部指令数，也不能乘除成耗时或加速倍数。

分区数和工作量并非单调：把同样的assignment分区预算改为48时，816行可到816次，而834行会到858次。按完整token的路由组对齐可以不重复行量化，但M=3、K=6时只有3个token分区，原方案则有18个assignment分区。总工作量下降与并行度变化必须同时评估；映射、最后写者、计数器、跨核flag及通信准备仍是合法性条件。

当前routing量化含BF16→FLOAT、absmax/127、FLOAT除法、FP16 CAST_TRUNC、INT8 CAST_ROUND。shared DynamicQuantV2是否具有同一舍入、scale、零与非有限值规则，尚未证明。当前MC2也接收BF16输入，不能把INT8指针直接塞入原ABI。条件方案包括：证明量化器一致后复用共同q/scale；或保留两个量化器，只共享被证明相同的输入/amax工作并支付双输出成本；或调整token分区。每种方案都保留参数与行generation、padding/mask、路由映射、peer记录和全部读者。

M=3、H=5120、K=6、18条有效assignment的声明GM值模型中，共同基线为338520B。复用已存在的shared q/scale后为246432B，差92088B；若两个量化器都保留并增加routing q/scale输出，则为261804B，差76716B。H+32路由记录的92736B写入仍在，receiver的dispatch/GMM/SwiGLU/combine、各自权重和网络协议也未删去。这不是HBM流量、线缆流量或真实收益。直接调用的816行可能包含800行之外的padding，不能按同形或相邻关系认定与shared输入全等。

在保留新鲜、互异、未压缩的逐assignment GM q+FLOATscale记录这一表示域中，必要写值至少为A·(H+4)，本例92232B；当前H+32记录多出的padding不是通用必要量。时间界保留T≥A·(H+4)/B_upper_GM_publication，容量上限未知，不能套用HBM带宽。若改为索引/共享记录等新表示，必须重建这个界。sender量化行次数与receiver专家GEMM assignment数保持分开。

已有REL73-08及三组联合合同得到细化，未增加独立候选或虚增关系覆盖。累计357条分支记录、24个语义模块；中央122条条件关系/25组联合合同，473对仍待汇入或复核。2073个regime/285598次执行的条件分析计数不变，9个regime/1352次执行仍缺关键证据；849条PMU缺失、16张阶段未知图、3136条模块归属歧义保留。全模块最低MFU和整体优化分析尚未完成。
