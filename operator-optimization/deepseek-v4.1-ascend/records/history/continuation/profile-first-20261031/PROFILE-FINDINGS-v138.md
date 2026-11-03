本轮补齐 F01–F12、F01–F13：HC 分支与稀疏索引发布、SWA/压缩注意力拆分后的共同调度。固定profile优先，随后只核对直接算子调用和已保存的native语义源码。整体目标仍未完成。

960组eager HC/SP前沿现在关联到SMLA、逆RoPE和head发送布局转换；另有192组QLI→索引整理及query量化、24次source20候选块TensorMove发布，共3480个原生TASK。这是将既有原始profile中的TASK精确关联到HC/SP前沿，不是新增设备测量或原生字段关闭。192次TopK self-copy源码API保留精确CID TASK列表，没有TASK不能据此认定成本为零。

为核对融合后的索引生命周期，又补了720次CP第206行shared.topk_indices切片API，归入192个生产者组。source2/8/14的TopK分别供后续5层复用，source20/24/28/32/36分别供后续3层复用。因此，在生产层把QLI、整理和当前SMLA合成一个kernel，也不能删除后续层仍要读取的稳定TopK发布。这里是源码分支和原生消费者的关联，实际bank地址、ID内容、世代和跨rank episode仍需证据。

source20还发布候选块，供source24、28、32、36的MODE2 QLI使用，共96个后续前沿。候选membership只依赖对应source20结果与逻辑行/可见域/epoch，可在资源允许时提前构造；它不必等待未来consumer HC生成新的Y。但未来consumer的query、QLI分数和位置TopK仍依赖当前层输入，不能被这个membership替代。QLI的INT8/DEQF16/ReLU/headreduce分数也不是SMLA的BF16 attention分数。QLI epilogue融合整理时，仍要保留有效前缀、时间序、-1后缀、重复出现次数、fillers/ties和所有候选/TopK读者。

SWA数据路径本身不依赖压缩索引，可以研究在QLI或压缩分支就绪前先执行。但当前SMLA已经在一个kernel中处理SWA+compressed KV、online softmax、sink与PV，调用方关闭LSE输出。把它拆成两份普通BF16 normalized输出后相加或拼接，不能恢复共同归一化。必须提供兼容的归一化质量和分子，例如声明域内的per-head(m,l,a)状态，并核对空分支、sink只计一次、原始出现多重集合、mask与实际gather顺序。

96个实数参考用例核对了状态合并，保留同一plane内重复ID对应的相同值；另有sink重复计算和错误去重的反例。这些证明信息与数学边界，不是native BF16等价性。原kernel的BF16指数/PV输入转换、online更新、tile顺序和最终BF16舍入仍需保留或另立数值合同。仅启用LSE或返回BF16 partial output，也不能自动恢复原先未舍入的信息。

合并点还受head维度限制：不同head具有不同归一化质量。反例中，两种partial状态有相同的O投影结果、相同的per-head质量，正确最终投影却分别为0.5和0.75，说明普通混合后的O投影partial已经丢失head贡献。通常应在head混合前合并，或继续携带足够的head级投影贡献并计价。head恢复通信只是特定行/head映射下的置换；若提前交换partial状态，还需新的dtype、布局、rank/epoch和collective/capture合同。原逆RoPE、O投影数值与归约边界不能因实数线性公式而自动移动。

HC与SWA提前执行必须按同一张图估算。1296组理想同服务模型中，64组出现协同、22组收益重叠、1210组在该具体用例中可相加。例如，单独提前HC的Y可能仍被外部QLI期限挡住；与SWA先执行组合后，之前没有显示的重叠才有价值。反过来，当前HC系数未就绪时，更早结束attention也可能不缩短Post入口等待。模型终点是HCPost输入就绪，省略共同Post服务，后续pre4/FFN没有计价。

再加入假设的state/launch/merge服务及HC系数竞争后，7776组模型同时出现较快、较慢和相同结果。这些数字仅说明独立收益不能相加、额外成本能改变结论，不是实际负载胜率或硬件最优时间。真实Cube/Vector/SFU/访存/链路容量、队列和资源竞争尚未绑定。当前完整4H系数、post/comb、下一FFN所需pre4与Post原始X残差仍然存在。

partial状态交付也有代价。在声明的FP32(m,l,a512)格式中，一个head有514个FLOAT值；M=800、64 heads时，一份值内容为105267200B，相比原BF16输出52428800B更大。它可能复用原online GM workspace，也可能需要新增bank或写读，不能把所有内容直接当作额外分配或HBM流量，更不能视为免费重叠。持久SWA kernel若一边占满QLI需要的资源、一边等待QLI发布，还会在无抢占的条件模型中形成等待环；必须通过已验证的资源保留、yield/抢占、launch边界或联合调度保证有限进展。这里没有断言当前设备发生死锁。

新增REL138-01/02，条件关系146→148、待核对449→447；HG93加入F12、F13，联合组仍26，全部子集与联合native合法性保持开放。35主候选、7历史框架记录、358语义行/26模块、4472次固定SP前沿关联和854节点/1346边/48提示的部分图保持。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍未补齐，整模块关闭0，全部模块物理MFU及最低时间仍未建立。
