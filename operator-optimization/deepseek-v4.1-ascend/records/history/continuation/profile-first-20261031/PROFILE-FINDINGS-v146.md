本轮补齐 F02 与 F23/F26/F27/F29/F30/F31/F34/F35 的八组条件组合。整体目标仍未完成；F01、F02 各34个中央候选关系已在声明范围内审查，不表示候选所有变体或联合子集关闭。

原始profile绑定3528个唯一TASK，其中248个补充了直接源码/CID关系：24组目标末尾的快照、collapse和norm共144个TASK，以及104个草稿head norm。复用了312个草稿input norm、312个FFN norm、88个main-context norm、72组aux前沿和936组跨层前沿。它们保持原始角色与精度口径，没有晋升9类待核对原生输入或全局phase。

目标末尾的第三种大行数中，每个owner的输入norm有834行；raw MTP快照是6672×20480，最终norm是6672×5120，缓存域为6668行。草稿输入norm每卡2或3行，head norm是全局10或20行，main-context norm另有12/24/6400/6528/6668行域。即使宽度同为5120，也不能复用它们的值或统计；辅助上下文还读取进入配置层之前的残差均值，而非HC加权Y或归一化Z。

48个MTP通信保留了两种关联关系。24个hidden all-gather没有旧缓存中的native-API TASK链接，但原始API CONNECTION_IDS集合下存在通知和AICPU任务。24个premix all-gather有缓存链接。原始连接集合成员、精确COMM连接上的TASK和缓存native链接不能相互替代；没有缓存链接不代表没有通信、没有设备工作或零成本，也不能直接晋升为唯一Compute因果归属。

末层按需输出与输入norm剪枝之间存在KV/状态读者：本轮对N=4、两层、全部U/W/aux子集及三种attention窗口共12288个声明值图，使用独立祖先遍历核验反向需求。768个全状态写入W场景均仍需全部8行input norm。这里不是原模型全部分支的枚举；实际KV、未来ring/accept/rollback、额外观察者、错误/RNG/EPLB、队列/capture/有限进度仍须绑定。

raw MTP快照与后续collapse/norm也要分开处理。若实际最终读者只需要U，可以在保留完整raw4H快照的同时，只对U做后续collapse/norm；完整快照并不强制所有后续norm行。它同时阻止在未查清快照读者时声称上游完整4H状态已死亡。当前快照写满G的事实也不自动证明下游每一行都被读取。

相邻层可研究FFNPost→下一HC collapse→RMS的producer/handoff组合；aux均值可与HC共享所选raw4H遍历；context与query准备可按实际cache范围流水。它们均须保留BF16 Post/Y/Z、当前coeff/pre4、raw残差和不同参数/位置/状态代次。草稿原始collapse仍可能供confidence读取，head norm不能替代它；未来keep或accepted结果也不能提前提供当前decoder的输入。

工作量按同一图计数。对所选独立物化的终端collapse/RMS图，FLOAT输入、乘积、求和与BF16中间接口并集为76GH B；raw MTP拷贝的另一接口为16GH B。它们是逻辑值接口，不是HBM流量、分配峰值或实际提速。按需删除Y行后，只能对剩余行计算融合接口收益，不能再叠加一份全行融合上限。24组当前六阶段TASK耗时中位和为2621802 ns，所选包络中位数2628102 ns；这些数值也不是可消除的关键路径时间。

条件关系178→186，待核对417→409，联合组仍26；35模型候选、7历史框架记录、358语义行/26模块及原部分图均保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图、3136项归属歧义仍开放，整模块关闭0，全模块物理MFU和最低时间尚未建立。本轮没有服务器、NPU、模型或replay运行。
