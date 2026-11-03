本轮把 HcPre 分支、输入归一化和前移投影的 SP carrier 放入同一张图，并补齐前一层 Post、Engram、辅助读者及 premix 世代的约束。没有执行设备任务或修改目标模型。

原始键关联了3200个目标图前沿和312个draft前沿。目标图保留2560个root已绑定、640个phase未绑定层及80个SWA/C1归属歧义；draft保留264次正面KV投影和48次正面metadata=None清零。HC、RMS、SP、attention Post、后继FFN Pre及前层Pre/Post的TASK/Compute/API/CID已检查。它们是固定源码与图控制域的关联，不是历史tensor指针、premix数值或跨卡同次迭代绑定。全部4472次attention SP中，另960次eager target尚未在本轮做该关联，仍列为下一步。

外部premix存在时，Y使用旧premix和本层HC输入，当前系数仍要交付post/comb给attention Post、pre4给随后FFN Pre。这项分支独立性已在v58/v93提出，本轮不重复当成发现。新增组合要考虑：Y与owner投影何时可交付每张卡的KV/QA，collective和输出路径何时返回，以及系数是否赶得上各行Post读取。原profile只记录整个HcPre，不能从两个kernel之间的空隙恢复内部C/Y服务或可用重叠时间。

源码还保留没有外部premix的路径：ProcessY改用当前计算的pre4。此时不能删除当前pre4→Y依赖，但也不能添加“等所有post/comb结束”的通用barrier；后两者仍有自己的Post截止时间。当前8个HcPre regime均有外部FLOAT premix输入；None路径只作为固定源码分支审查，未证明它在本次负载执行。

9349组有限抽象DAG分别计算原图、仅拆Y、仅前移投影、两者组合的完成时间。模型明确假设固定服务、无资源竞争及选定的完整carrier交付协议；既出现收益重复，也出现组合弥补单项无效的情况，说明不能把两个单项的差值直接相加。实际新C/Y/投影服务会受Vector、Cube、访存、队列、pack、QA路由和collective实现影响；512维载荷不自动等于更快。更细的行/tile需求图也可在部分行ready时推进，完整carrier模型不是通用整算子等待约束。

若Y和归一化后的Z确实只在新融合内部使用，保持Y的BF16舍入、完整5120维RMS和Z的BF16舍入后，Y/Z各自write/read的逻辑接口预算合计8MH。Q_a所需raw行、rstd、C1/C2/Indexer及其它读者没有闭合时，不能领取完整预算。需要按行scratch、重算、Vector→Cube布局、可见性和新队列方案计价。当前full4H系数、HF32/FLOAT投影、cast/Gram/RMS/sigmoid/Sinkhorn及原HC输入残差仍有需求；这里没有证明HBM必要字节、片上可行性、分配VRAM或物理上限。

跨层3120条Post→Pre关联中，160条对应block1/14的Engram写入分支，240条对应block37/38/39的辅助读者，2720条在所选schema中没有中间compute记录。没有记录不等于没有其它读者。新Y、当前系数和Post residual必须使用Engram更新后的相应行；可以研究按行交接，不能一概提前到Engram写之前。辅助均值不修改X，因此不应凭空强制Y等待它计算完，但它要求X内容和bank仍存活。

前层FFN Pre的pre4与前层FFN Post的新残差是不同输入世代。把当前新pre4当作旧premix，会改变Y；早到的KV载荷也不能授权复用X缓冲区，因为当前系数、Post residual、aux及排队读者还可能访问。原kernel的整体完成事件没有提供新Y部分发布合同，不能仅改变launch顺序就声称实现早交付。Post→Y/norm/owner-KV可研究producer融合，但完整HC状态仍供这些读者，不能同时删除全部残差存储。

中央新增REL133-01/02/03，分别对应F01/F02/F03与F32的因果、接口和跨层世代关系；它们共享一个HC基线，不代表三份独立收益。关系139→142、待核对456→453；HG93补入F32，联合组仍26，全部子集/运行时合法性未关闭。35主候选、7历史框架记录、358语义行/26模块及854节点/1346边/48提示的部分图不变。9类/1352次原生字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义均保留，整模块关闭0。整体任务未完成。
