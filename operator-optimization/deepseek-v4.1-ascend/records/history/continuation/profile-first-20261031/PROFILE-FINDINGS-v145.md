本轮补齐 F02–F12/F13/F14/F15/F16：输入归一化到索引选择、稀疏注意力、head 恢复、输出投影及 reduce-scatter 的条件组合。整体目标仍未完成。

960组正Query前沿复核10280个TASK，补充了2960个输出阶段角色：960个WO_A、960个WO_B、960个写回稳定输出buffer的TensorMove，以及40组head zero/copy。大行数WO_B为BF16 MatMulV3；不能仅凭源码“直接写输出”的注释认定copy已消失。使用新out producer删除这份私有接口，仍须证明out/stride/alias/capture和native精度合同。

2880个通信记录按完整记录中的group/opName形成360组八rank标签一致性记录，没有用跨rank时间或ordinal配对。标签一致仍不是正式的request/tensor身份、全局tag语义或时钟证明。959个A2A使用直接CID；另1个复用v16已有的精确CANN→dequeue→decoded flow→enqueue→最内层HcclAlltoAll证据，保持queue-derived等级，没有按时间邻近补配。

本轮修正了先前简化模型的行域口径。第三个大批次中：每个owner有M=834行；大多数Query为834，末卡有效Query为830；cache实际clip为T=6668；而all-gather接收区和WO_A/WO_B/稳定输出处理G=6672行。不能把T用于完整G buffer的写入量，也不能凭cache少4行就推断RS前有一次4行concat。本轮960个窗口未见RS zero/cat，实际投影输出已按G对齐；源代码的其它未对齐分支仍保留。

对声明的QA/KV接口图，当前全局Z参考量应为2H(G+T+Q) B；旧4TH+2QH式对应T裁剪carrier的简化域，少算完整接收buffer的2H(G−T)。在6672/6668这组中差40960 B。将同一G写入计入基线和rawY替代图后，双方相抵的2QH共享读取参考不变。这些都是逻辑值接口计数，不是HBM流量或物理时间。

Q830路径先zero一个M834的全head buffer，复制Q有效行，再做send布局变换。可研究在inverse-RoPE/pack producer中直接写valid rows与zero tail，避免中间padding/覆盖接口；但A2A仍须得到原协议要求的全部head块，原BF16舍入、坐标和有效位置都要保留。删除padding矩阵计算或G−T行还需关闭所有pad/state/error/NaN/额外读者，当前没有证明。

跨卡等待现已进入参考图：输入normalized/projected carrier等待所需owner及传输；head exchange等待所需query发送者；RS等待对应TP partial；HCPost仍等待当前系数与残差。3072个P8参考排程在假设服务下640组joint较快、2432组较慢，不能将单卡缩短相加当端到端提速。模型采用充分的全前沿，允许更细row/tile流水，但没有把现有whole-kernel边界说成普遍必要条件。

上一轮把Z当作消费端已就绪的假设也已显式展开：全部6144个C2参考Case加入max(owner_Z)+carrier，零附加释放时逐项复现旧结果；加入所选正代价后3032组终点变晚，3112组被其它期限遮住而不变。它们没有实际跨rank时钟、指针或带宽含义，但排除了从单个owner的M行就启动全局T行C2投影的模型误用。

另一个参考例说明细化RS到row owner可能影响下一层：当前full-frontier会让两个owner同时等待最慢行；若保留原BF16 partial/native SUM并引入合法tile handoff，较早行可先进入FFN，使假设的下一完整carrier期限从201变为103。此处只是排程反例，新的partial publication、队列、资源保留、capture和有限进度均未得到native证明。

共享候选membership也不能代替当前query的QLI分数，TopK更不提供attention各head的归一化质量。构造的正dot-score例中，取消BF16归一化舍入可翻转排序；它不表示冻结模型的实际QLI输入已发生该变化。SMLA的headwise softmax/sink/occurrences、LSE为空的接口、BF16/online顺序、cache/state读写、HC原始X/系数/Post/pre4均保留。

条件关系173→178，待核对422→417，联合组仍26。35主候选、7历史框架记录、358语义行/26模块、4472次SP关联及854节点/1346边/48提示的部分图保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍待补齐；整模块关闭0，全模块物理MFU与最低时间尚未建立。没有新增服务器或NPU运行。
