本轮补齐 F01–F17：HC 分支与 C2 投影、pool 和跨迭代 ring 状态如何共同重组。只使用固定 profile、定点源码和条件模型，没有运行设备或修改模型。整体目标仍未完成。

72组 eager C2 前沿现在各关联16类原生消费者，共1152个TASK，覆盖输入FLOAT转换、SWA norm/RoPE、清零、pool、ring更新、pool norm、indexer旧latent读取及量化/FP16 scale/cache写入和long RoPE/写入。关联依据是同一forward内的源码栈和精确CID。当前均执行pool→ring更新→norm；这证明当前顺序，不能证明norm在所有合法重组中都依赖ring更新。原9类/1352次原生字段缺口并未因此消失。

v24/v62已经确认原request配对、历史偶数位置、pool旧读与ring覆盖顺序，以及token-aligned输出；v88已经分析过SWA BF16与两份C2 FLOAT投影的类型化通信。本轮将这些前提和HC当前系数/Y分支组合，核对分片后是否还能保持同一计算与状态合同。

直接按SP分片各自调用pool会出错。原request起点s、长度n决定逻辑配对；只有原request首个奇数起点的配对需要读旧ring。若token0、1分别在两张卡上，原计算应使用本批两行；第二张卡将自身起点1当成新request起点，就会改读ring中的旧token0。有限标量反例中，本批值1、3应得到2，旧ring为9时却得到6。正确的owner方案必须保留原request、pair、current/old选择和residual算术分支，补传本批跨卡伙伴。即便值相同，把普通分支改成残余分支也没有原生舍入等价性证明。

16752个整数域用例覆盖原起点、长度、分片切点、物理行基址及有效/无效页，独立核对完成行、历史读取、跨owner伙伴和原末尾32行集合。逐片各维护min(片长,32)还可能增加原本不会写的行；若改为并行更新，需要另外核对覆盖次序和相同slot的冲突。这些是源码helper域的条件证明，实际request元数据、页世代、接受/回滚和跨rank episode仍未绑定。

可比较三类交付方案：

| 方案 | C2值内容及接收端责任 |
|---|---|
| 传原FLOAT投影 | 每sender行K+score共4096B；接收端继续原global pool、norm和ring维护 |
| owner先pool，保留现有复制ring | BF16 pool结果1024E，加原FP32 K+score尾部4096U；补传跨owner本批伙伴和必要旧状态，接收端恢复原token-aligned输出及ring世代 |
| owner先pool，重做状态归属 | 需要新的历史读取路由、页迁移、回滚和capture协议，不能继续假定当前每rank本地ring读取合同成立 |

其中E是所有有效原request完成的pair数，U是各有效原request末尾min(n,32)的维护行数；跨owner本批伙伴X另有4096X的逻辑内容，远端真历史也需单独计入。每个需要的状态副本、header、对齐、metadata、额外raw读者和collective拓扑均未包含在这些内容公式里。SWA另加1024T，必要的Q_a另加2560T。SWA可选择交付投影后由接收端norm，或移动norm并核对对应参数/数值合同。不能将这些公式称为最小wire字节、HBM流量或实测速度。

只传当前BF16 pool结果不足以恢复未来状态：两组FP32输入[1,3]和[0,4]在等score时都得到当前pool=2，但保留的偶数位置0不同。若helper允许后续从位置1重新开始，下一次pool会不同。这个有限helper域反例不声称实际负载必然采用该回滚；它说明不能在尚未关闭未来观察者时删除FP32状态。F34的只维护必要偶数位置、16槽表示等方案需使用其自己的观察者及ABI证明，不能同时领取“现有完整ring”与“已裁剪状态”的收益。

固定发送长度必须按实际sender分布设计。示例取T=6400、8卡、两个各3200行的偶数起点request；它只是参考元数据。每sender完成400个pair，但64行ring尾部集中在sender3和7，各32行，不能用平均8行作cap。保留完整状态时，C2 pool+tail的参考固定内容为540672B/sender，加SWA为1359872B，再加Q_a为3407872B。实际profile虽有相应shape，不能由shape推断示例起点、长度或尾部分布。

pool旧ring读取必须先于覆盖同一范围的更新，pool的BF16输出必须先于norm；在输入输出和副作用不重叠的条件下，ring更新不必阻塞当前norm。可将当前pool/SWA交付和未来状态尾部交付拆开调度，但尾部必须赶上未来历史读取及投影/metadata/bank回收期限，必要时显式保留尾部快照并计入copy与队列成本。1536个独立加权DAG用例中，给定假设服务下合包当前路径较快144例、拆分较快1104例、相同288例；未来ring完成另行计价。这些不是实测分布或最优方案。真实Cube/Vector/访存/链路竞争、ordered communicator和HC系数期限会改变结果。

HC原始4H/FLOAT-HF32系数输入与C2的归一化BF16 Y再widen输入不同，不能因FLOAT存储就视为一个同输入宽GEMM。保留当前系数、post/comb、下一FFN的pre4和Post原始X残差。保持FLOAT投影及pool算术、BF16 pool→512 norm边界、完整token-aligned清零/输出位置、indexer旧latent→long-RoPE的逐范围顺序，以及INT8 key和FP16 scale舍入。原生new-M数值等价性、权重与norm副本、所有读者、alias/epoch和实际容量上界仍需证据。

新增REL136-01，条件关系143→144、待核对452→451；HG93加入F17，HG73-C2与HG88加入F01，联合组仍26，全部子集及联合原生合法性保持开放。4472次固定profile SP前沿关联保持完成。35主候选、7历史框架记录、358语义行/26模块、854节点/1346边/48提示的部分图，以及849个PMU缺口、16个phase未绑定图、3136项归属歧义均保留。整模块关闭0，全部模块的物理MFU和最低时间仍未建立。
