本轮从v28发现的F04覆盖缺口推进，先读保存profile。全部32份原生分片及80张图中，共4472次RmsNormCast：3200次图内，960次target原生，264次draft proposal，48次draft dummy。dummy仅是调用祖先，不等于启动或可删除。每次有BF16/FLOAT双输出；图内3200个FFN块各有对应FP32 router、HashTopK、BF16 routed/shared专家后继。没有按static definition去重，也没有把相同次数配成同一次调用。

原生帧把问题收敛到device_op72、model846/912、MoE285及router204。定点源码确认，现有RMS+Cast已经融合；不能再当作新实现。关键精度问题也在指定算子源码合同内回答：BF16分支先做FP32归一化/乘gamma，在115行舍入为BF16，再在122行将该舍入结果扩成FP32，128/129行分别写两路。因此FP32路不是未舍入中间值。新router消费者可研究边读BF16边精确扩展、避免独立FP32全局平面，但还必须保留舍入边界、FP32权重/累加、专家/shared分支和其他读者。不能把unrounded accumulator直接给router，也不能只凭topk排序不变删除norm缩放；当前sqrtsoftplus、bias、路由权重同样属于结果。

kernel、build及打包源码的SHA相同只支持源代码一致性，不证明每个执行已绑定到相同binary hash；本轮没有新NPU数值实验。进一步核对已验SHA的两个router helper发现，只有gate带weight_fp32时才选传入FP32路由平面，其他gate可能忽略它并走自身forward。因此再查原生完整栈，1272个router API全部明确命中408行直接F.linear分支，分为target960、draft proposal264、dummy48；没有仅凭参数出现就声称实际消费，graph部分仍分级为source-consistent。FP16另有先低精度cast再乘gamma的分支，未推广BF16算术合同。F04保留已有融合、FP32平面/路由联合边界、HC/共享quant更大子图和行/feature/有效行重组三类后续条件；没有直接drop输出、跨层复用norm值或精度放宽结论。

本轮又从3200个FFN块识别出新的跨层机会。每张图40层都出现Cast、Equal、Fill、SelectV2四步ID准备，80×40×4共12800条TASK；另有5088条原生TASK由router163/176完整caller确认。四步区间中位7500ns只是观察排程，不能相加当作收益。源代码处理的是INT64转换以及仅-1→0，其他负数图像sentinel必须保留。HashTopK还接收逐层的hash表、vision/text bias、scaling等；第三输出被忽略且out_flag=False，不能从profile仍显示全score shape便推断大矩阵写流量。

补强已有F20的ID复用变体（本轮内部标识F20-ID，不新增主编号）：路由ID转换/占位修正的层间复用与代际重组。target forward在循环前SP切分input_ids，并把该表达式传给各层。MC2方法对IDs做零填充和行切分，取决于每层num_tokens、replace-allreduce、TP rank/size和DP padded长度。对逐元素f(x)=(-1则0，否则INT64(x))，有效行上f(Px)=P(fx)，填充行上f(0)=0，所以它与这类零填充/行选择可交换。由此可以在稳定input代际内只算一次路由用f，再保留各层独立行对齐；不要求层间所有P都相同。这里只改分析图，没有实现共享bank。

保留三项明确反例：把其他负数一并clamp会改变图像token；不变地址里的[-1,7]更新为[9,7]后旧bank不再有效；若别的协议用-1填充而不调整填充值，则直接交换也不等价。F20-ID不修改原始IDs，不省略collective，不把target/draft银行混用，不按地址/shape跨迭代复用。每次forward/graph replay需要匹配新的输入generation、捕获地址/就绪协议，callee/异步别名不可变性仍是条件。

MC2行域也补齐了源码机制：prepare对BF16 hidden、FP32 router输入和false mask按DP统一长度padding，IDs使用相同行轴零填充/TP选择；SP finalize返回原始num_tokens前缀。因此M=800、P/TP=816时会补16行再截回800，解释了规范上的norm800/where或gating816/残差800关系。保存的这些形状与源码机制一致，但没有靠相同次数或最近时间造出实际same-call padded_num_tokens/mask证据。旧F23的行域问题从“机制未定位”缩到“实际上下文/有效行/代际绑定”仍待证。

首版op文件名定位把生成物也计入候选，触发数量保护；失败保留。修正版只列16个C++源码metadata，并保留其余路径；后续只读已定位的四个小op文件、直接consumer和MC2方法，没有扫整个SDK。一个MC2 subclass没有自己的__init__，原样记录，未当作没有prepare实例化。RMS/ID/kernel源码合同、source/build线索、profile与未知级别分开。

F04新增12项条件，F20-ID新增12项条件；模型候选仍为35项，框架仍7项。模块表和阶段表链接同步，候选主记录与索引更新。26个整模块仍未关闭，目标继续active。下一批优先转M14 Engram hash/cache、host/UVA gather-dequant与gate边界，用已有fullcaller和图外段继续补覆盖，不重复追查已回答的F04舍入问题。本轮没有模型导入、NPU负载、服务或健康状态变更，最后健康证据仍v12post。
