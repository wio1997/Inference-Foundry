本轮转向注意力侧，核对稀疏索引重复是否能作为融合/重组收益。复用已有640条QLI→prepare→SMLA顺序前沿，共1920个TASK，保留512条root绑定、128条phase未绑定；这些profile没有索引值、重复率、每query gather次数或新方案ready事件。因此本轮只对影响判断的已有冻结算子代码局部取证，不把shape当成重复数据证据，没有服务器、Torch、NPU或模型运行。

prepare_indexer_indices并不去重。它按slot过滤selected>=0且selected<visible，替换为INT32_MAX sentinel，经可逆FLOAT bit-key排序后恢复ID，最后sentinel变-1；长度是valid slot的数量。例如selected=[1,1,-1,5]、position=3、ratio=1时输出[1,1,-1,-1]，长度2而不是distinct数1。结论限定position+1不溢出、0<=visible<=INT32_MAX的正常整数域；不能跨越位宽/sentinel边界移植。7500个multiset参考用例保持每个有效ID的出现次数，另复核了10006个边界/样例key值的有序可逆编码，未执行Triton或native sort。

QLI源码的单个S2 tile使用ArithProgression加cuBaseS2Idx，局部有效lane ID确实不同。但全局有效TopK唯一性还需要每query的S2/LD分区不重叠、sort/merge保留record以及offset/range一致，不能单凭TopK输出shape证明。固定V41 ordinary caller没有传output_idx_offset，注册schema默认None；另一个通用非零offset分支会对包括-1在内的输出slot统一加offset，可能把无效填充变成正ID。该分支没有被断言在当前case生效，也不是实测bug；它说明“所有调用都唯一/填充总为负”不能脱离调用域推广。

SMLA所读的是索引出现项。源码先按sparseBlockSize展开每个slot，再检查真实position，PA分支经request block table取地址。对正blockSize，-1只展开负位置，已有代码会跳过对应有效KV拷贝；不能把重复的-1 padding当成重复有效key载荷，重新领取整份gather/QK/PV收益。CopyInKv还会根据物理offset选择成对拷贝的最小起点/绝对stride，重叠或范围不适合时回退两次SingleKv；新gather/reuse必须保留实际发出的行序与数值契约，不能只凭逻辑索引已按时间排序就假设物理累加顺序相同。

若一个query/head确有重复有效项，直接删除会改变softmax权重总量。例如同一score0/value1出现两次，另有sink score0且sink value0，原输出为2/3，删为一项后为1/2。精确实数域可以按组保留multiplicity c：denominator=exp(sink-m)+sum(c*exp(score-m))，numerator=sum(c*exp(score-m)*value)，sink仍只出现一次；但前提是组内完整有效score、value、mask/bias/position和bank epoch都一致，不能只比较裸指针。

实数式也不是无条件native bitwise替换。声明的source-style BF16 exponential/PV参考中，exp值257/512舍入为0.5，三份先舍入再参与PV得到1.5；先合并exp再BF16舍入得到1.5078125。带一份exp1的背景/sink时，参考最终BF16输出分别为0.59765625和0.6015625。这个有限反例没有测native exp/log或MMA，只是否定“加log(c)/乘c总能无损替换”这一通用推断。保留原occurrence顺序、复用一次KV读取后重复原算术，是另一候选域；它也须证明读者、舍入与并发写入契约，不能自动把所有QK/PV工作删掉。若真实producer给出唯一有效ID证书，则去重是恒等操作，没有这份重复算术收益；当前归档没有证明其全局native证书或实际重复率。

DSpark非因果、use_logical_indices=True的另一个来源给出了跨query的结构性重复：每request先构造start=max(seq_len-query_len-window,0)、visible=seq_len-start的连续逻辑窗口，再按该request query_len重复同一索引行和长度。1680个含零query请求、不同prefix/window的独立参考核对了恢复顺序和长度。它支持一份request描述符供多个query读取的条件候选，也给v60既有KV-union敏感IO公式提供了新的source限定实例；不是再加一份独立IO收益。native可能已经通过tiling或cache复用了KV，profile缺少相应证据，未认领新的HBM或耗时减少。

同一个索引窗口不表示query相同。相同keys=[0,1]、values=[0,1]、无sink时，q=0与q=log(3)的实数输出分别1/2和3/4。每query的QK/softmax/PV、位置变换、mask与输出仍有需求。新descriptor接口必须支持request owner、cache/page/head/layout/compression/mask generation，并保证index/length同步发布、capture固定地址、旧reader及queued writer退役；现有persistent bank是基线，不能直接拿8B替换当前native期望的整行buffer。

9个明确布局参考分开计入descriptor和INT32 query-owner。例B=1、T=3、index width256，当前活跃indices+lengths内容3084B；一份request index row+length+owner为1040B，区间(start,length)+owner为20B。差额2044B/3064B仅是选定新ABI的逻辑内容参考，不是实际分配、HBM必要字节或可达提速。原_max_tokens持久容量、其它读者、对齐/metadata和新地址生成开销仍需计入。理论状态保持SYMBOLIC_CEILING，数值时间下界/吞吐上界/实测上界比为未知，没有从descriptor字节推导MFU。

把三个以前未展开到primitive的selected source分支加入G9：prepare multiset、DSpark logical-window broadcast和SMLA PA occurrence gather，共41个source节点、49条边。部分全图台账从813/1297更新为854/1346；source行不等于唯一native算子，连通2..4节点集合只是结构枚举，未证明整张图或所有条件分支闭合。两个新结构提示F12/F31与F13/F31已有中央REL73-28/REL73-27，结构提示46→48，中央条件关系仍135、联合组26、待核对460。现有REL83-08与上述两条关系和HG83补入了语义证据。

新增M09-OP-12、M10-OP-15两个语义families，台账356→358行、仍26模块；没有因局部参考用例或新增source图而关闭整模块。原9类/1352次原生字段、849缺PMU、16 phase未绑定图及3136含糊归属均不变。实际ID/mask/page epochs、全局QLI分区与native算术、完整读者/生命周期、物理容量上界及同口径最低利用率仍缺关键证据，整体目标继续。v120 Pad授权已完成消耗，本轮没有新的设备运行。
