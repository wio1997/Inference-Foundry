# v167：用共享profile收窄MoE证据需求

32份已有NPU profile中，288个原生图完成范围均有40组相同的MoE/共享专家尾部，共11520组，模式不匹配和观测重叠均为0。R为DispatchFFNCombine，S为其后dynamicquant/GMM1/DequantSwigluQuant/GMM2四核，M为两路结果相加。S工作时长中位54.8us；R中位308.847us。96个R超过1ms的记录全部在各自图内的第一个MoE局部位置，两个窗口均出现，不可归因于一次性初始化，也未证明是接收负载、输入就绪还是DP偏斜。

固定已测服务时长和数值语义、只允许同一episode内R/S重叠且M等待两者、保留episode串行顺序及外部串行段长度的域内，每组必须至少max(R,S)+M。因此Gnew>=Gold−sum[episodeSpan−max(R,S)−M]。图时间比值上界中位1.0671、范围1.0480—1.0825；这是放松下界推出的条件比值，未展示可达调度，不是物理理论最小或请求收益。任何跨层、跨轮、提前释放或改变通信等待的方案均不在此固定服务模型内。独立复核保存在shared-tail-bound-independent-review.json。

复用v56已证算子事实：路由/dispatch/GMM1/激活/GMM2/combine本来就在一个原生边界内，零M本已跳过blockMmad，tileM128不证明实算128满行。计量应分rawreceiver n_e和容量前缀截断后executed m_e；直接两GMM为每assignment35389440MAC。MoeGatingTopKHash不自动保证同token专家ID不同。若同一次EP16交换所有rank均M<=3，总assignment<=288，384expert中至少96为空；这个空expert基数结论不需distinctID。单expert<=48需distinct，且本轮未将jointM3前提绑定到实际跨卡迭代。padding-only在向量/scale/投递均不变域内最多去掉0.5435%逻辑record字节，只有字节量动机降优先级，未证明wire或延迟收益。

直接按本机SDK格式解码旧32份raw MC2记录，共384条，得到EP16及两个TP8完整成员。usr_rank_id在TP8内相对编号0—7，不是globalrank；没有per-operation序号，新Level0 cohort没有该记录，不能同号跨批串接。没有重新导出巨量启动栈。

三个受限真NPU探针实际完成，应用数据分别968B、1160B、1160B，不加载模型、不重启主服务。第一个证明GPU索引history在同图两次replay保存到不同轮次，并由MSTX->CANNconnection->native execute/wait准确绑定。第二个证明图创建时NPU采集保留逐核launch连接，但异步提交线程让Python标记无法直接同线程嵌套到launch。第三个加入创建期CPU采集，正式CONNECTION_IDS将CPU源api集合20->{48}、21->{60,65}连回3个捕获内核的6次执行，准确得到2个Python创建点/调用栈，无时间近邻猜配。该路线只在此控制图验证，下一批真实推理须验证customkernel覆盖；不把它算作历史归属补齐或all-reader证明。探针原始输入hash不变，所有handle终止，三个history均符合预期；profiler stop警告保留，复核限于确实存在的预期记录。

下一步不再重复小图或逐对配对，使用一批创建+运行共享采集补实际receiver counts、同轮mask和首个MoE/普通后续MoE对照。只围绕MC2窄接口确认schema/ownership，不研究框架全仓。真实跨卡结论需要实际group/步骤见证，不能按时间或本地ordinal拼组。采集新reader不用于证明原buffer有旧reader。原始35+7候选、26模块358行、所有条件分支/跨算子/层/卡/迭代融合拆分重组与上限目标保留；未分析范围和physical capacity/numerics/reader问题仍在，整体PROGRESS。
