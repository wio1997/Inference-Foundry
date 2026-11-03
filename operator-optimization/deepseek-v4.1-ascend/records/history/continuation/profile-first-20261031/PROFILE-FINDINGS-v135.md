本轮补齐 F01–F33：HC 系数分支与 C1/SWA 投影、通信和旧值读者如何共同重组。只分析固定 profile 与源码，没有运行设备或修改模型。

从上一轮24个eager C1前沿出发，新增关联了每组12类原生消费者，共288个TASK：两个512维norm，SWA/index/long三种RoPE，indexer的512→128投影和128维norm，INT8量化、key写入、FP16 scale转换/写入以及long KV写入。所有关联均由同一forward内的源码栈和精确CID确认，不能替代历史tensor地址、权重副本或新M数值等价性。旧80个captured graph的SWA/C1歧义继续保留。

v27已确认的独立矩阵、两个512 norm及旧latent读写顺序是本轮前提。SWA与C1各自构造权重，不证明加载后的值相等或不等；共享输入分块不需要两矩阵相等，删除一份计算则需要另外的值/epoch/精度证明。HC系数读原始4H/FLOAT-HF32状态，C1/SWA读归一化后的BF16 Y，不能无条件合成同输入宽GEMM。原HC系数、post/comb、给后续FFN的pre4和Post残差需求仍保留。

| 交付切点 | 每sender行的值内容 | 主要接收端工作 |
|---|---:|---|
| 两份GEMM输出，norm留接收端 | 2048B | 两个512 norm及后续分支 |
| 两份norm后的latent | 2048B | index投影/norm/RoPE/量化与long-RoPE |
| long-RoPE结果与index norm结果 | 2304B | index-RoPE/量化及cache写入 |
| long-RoPE结果与编码index终点 | 2178B | key/scale/long cache写入 |

各方案均含512维BF16 SWA平面，SWA角色的RoPE/写入按声明位置保留；若还需传1280维Q_a，每行另加2560B。这些只是命名平面的值内容，不含header、对齐、slot映射或其它raw读者。只移动GEMM时可让接收端保留自身norm参数；移动norm或index编码则增加相应gamma、quantizer、position、nativeM与舍入条件，不能把更强条件误当作所有投影前移的限制。

编码终点方案比latent多130B/行，试图用owner的Wk/norm/RoPE/INT8quant/FP16 scale计算换掉接收端重复工作。它要求实际读者只需要这些终点，key/scale/long的映射和世代一致，原量化及FP16舍入精确保留，数据在当前QLI/SMLA需要时就绪。不能等接受结果后才交付当前所需cache；也不能将本轮fallback的编码布局直接套到packed、folded、empty或!owns_k分支。新接口的数值等价性和提速仍未验证。

3456组有限抽象DAG比较了有序分开发送、合包和owner编码交付。在给定假设服务中，合包较快548组，拆分较快284组，其余2624组相同。它们只说明较晚的C1/编码分支可能挡住SWA交付，省一次collective也可能有价值；不是实测比例或硬件最优时间。实际Cube/Vector/访存、队列、pack和HC系数竞争会改变服务，Post还要等本层系数。表中模型的终点是attention HCPost，后续pre4/FFN服务没有计入。

indexer的Wk必须读取未旋转C1 latent的相应值，再允许in-place long-RoPE覆盖同一范围；也可付出copy或改为out-of-place。只需要保留真实行/范围的旧值读者，不应添加“整个index发布完毕才能long-RoPE”的通用barrier。packed1024输出的两份512视图还带row stride1024，不能继承当前连续512缓冲区的合同；若需要转成两个连续输入，适配的逻辑值读写为8ND。两个512 norm、所有GEMM→BF16及index/FP16边界继续保留，不能改成一个1024 norm或未舍入输入。

新增REL135-01，条件关系142→143、待核对453→452；HG93补入F33、HG74-C1补入F01，联合组仍26，全部子集和运行时联合合法性未关闭。4472次固定profile SP前沿关联保持完成；35主候选、7历史框架记录、358语义行/26模块及854节点/1346边/48提示的部分图不变。9类/1352次原生字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义均保留，整模块关闭0，整体目标仍未完成。
