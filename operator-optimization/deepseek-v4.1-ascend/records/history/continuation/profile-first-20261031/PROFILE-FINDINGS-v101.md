本轮补的是此前只有接口合同、尚未展开算子步骤的随机拒绝采样PyTorch回退分支。先复用已有profile证据：72条greedy链可见；原始保存的精确random/recovery名称查询为0。本轮因此是条件源码分析，没有把回退分支写成当前负载已执行，也没有新采集、设备测试或目标代码修改。

新增G8的5张局部算子图，覆盖恢复采样的token→request映射、普通恢复值计算、普通随机验证、block随机验证，以及可选blockwise恢复。共70个源图节点、88条边；这些是source边界行数，不是新增原生算子。源码只取对应算子函数、调用guard及导入的随机数生成器，没有展开框架研究。

最重要的结论是后端不能按同名算子直接互换。Triton block逐步计算pi=min(pi·p/q,1)，PyTorch回退先把每个p/q截到1，再做cumprod。取3个位置的p/q=[1/2,2,1]，uniform=[9/10,4/5,4/5]，Triton后续位置可重新通过，最终接受到第2个位置并加bonus；PyTorch前缀质量一直为1/2，没有合法位置，会读恢复行0。这是有限正数上的公式差别，不是浮点小误差，也没有断言当前C8实际触发。

普通验证还有比较精度边界：导入的uniform生成器返回FLOAT64；Triton调用显式转FLOAT32，PyTorch回退直接接收原值。p/q=1/2、uniform64=1/2+2^-26时，原值比较拒绝，转FLOAT32后uniform变为1/2，比较接受。融合需要保留所选后端的路径。

可选blockwise恢复也不是普通max(P−D,0)。它使用max(p_before_position·P−D,0)；一个同Q反例的两个恢复ID分别为0和1。读取的root调用没有传use_block_verify，默认仍False，因此不能仅因后续启用block验证，就把这个可选恢复公式套进去。

有三类具体优化机会。第一，当前恢复映射用T×B区间比较、BOOL→INT32、argmax/any/where找request。对于非负、有序cu，可以改为右偏upper_bound；零长度请求使相同前缀出现重复值，左偏搜索会错。也可以按互不重叠的request区间写入owner/position，或由融合kernel的request/position网格直接寻址。容量尾部t≥cu[-1]原本映射request0，通用helper不能无条件跳过。5456组有限用例覆盖了零长度和容量尾部；无序cu给出了反例，必须声明或验证适用域。

第二，普通PyTorch恢复有private clone/残差、Q gather、两份Q safe中间值、除法结果和异常guard结果等可融合接口。按6个FLOAT32临时平面、各写读一次的独立物化参考，预算为48·T·V字节；例如B=2、S=5、V=129280时是62054400B。这个数不是实测HBM、分配峰值或必需流量，编译器也可能已消掉部分平面。新tile内核可按原顺序算残差/排除draft列、保留Q异常guard后直接归约argmax，避免物化这些私有平面，但不能删除原P的接受判断、raw/logprob/entropy等其他读者。

新padding还必须单独处理：PyTorch只有实际V列。若全部有效列因Q=0/Inf被置为−1e10，而新kernel把额外lane填0，padding会错误获胜；应按对应有限argmax合同排除无效lane。NaN、dtype、first-tie、原除法舍入和NGRAM/概率残差分支仍需保持。不能借用另一个upstream kernel的reciprocal-multiply或tail行为来代替Ascend当前实现。

第三，已有v33的decision→selected recovery合同可用于这些回退body；它不是本轮新发现。需要真正改消费者的读取集合：当前PyTorch验证先gather所有recovered值，之后才按first/last决策选用。保留原uniform/Q默认及seeded draw顺序的基础上，可以只算最后需要的恢复行，并与临时平面融合按一个基线计费；不能叠加两份覆盖同一临时量的收益。普通验证的私有entropy可以按有效非greedy访问前缀按需计算，block验证则一般仍需全部相关位置。减少词表工作也可能增加决策阶段和损失并行度，未声称实测更快。

block递推还存在一个理论重组方向：精确实数下f_r(x)=min(rx,1)可用(a,b)表示min(ax,b)，函数复合(a2a1,min(a2b1,b2))满足结合律，可做并行prefix scan。但它不是当前FLOAT32递推的直接替换。有限正数P=[2^-100,2^-100,1]、D=[1,1,2^-100]时，原顺序在第二步下溢为0，后续仍0；重新结合可以保留2^-100并改变最后接受判断。125组实数复合用例与该IEEE反例分别保存。源码S步循环也不能因此宣称是所有合法算法的硬件下界；精确组合舍入函数是另一种有成本的表示域。

可选blockwise helper的max(n).item()还可以登记静态S≥max(n)的新接口候选，代价是更多无效prefix槽位；必须保持有效T/cu/position、零长度和错误行为。它没有被计作当前root正在支付的同步开销。

理论部分保留SYMBOLIC_CEILING：在无缓存winner或额外结构的独立有限score比较模型中，K个完整词表行最坏需要KV次score观察、K(V−1)次比较决定；这不是融合后强制GM/HBM读取次数或每列Div次数。实际LOG/EXP/DIV/归约与存储层容量仍需同口径上界，不能借BF16矩阵峰值代算。

台账新增3条M17语义记录，并细化既有F31/F35关系，没有新增中央pair或原生执行。当前为362条分支记录、24个语义模块；局部源图813节点行/1297边行，46条结构提示；中央131条条件关系、25组联合合同，464对待汇入或复核。2073个regime/285598次执行条件分析，9个regime/1352次待关键字段，849 PMU/16阶段图/3136归属歧义未变。整体融合覆盖、最低模块MFU和物理理论上限仍未完成。
