本轮围绕F35的直方图 → penalty → 采样归约子图分析融合、拆分和跨迭代重组。32份既有profile投影中没有匹配的penalty/bincount名称标签；这只是检索结果，不能证明分支未执行或no_penalties实际值。若no_penalties=True，源码本来就提前返回，该子图当前收益为0；temperature0/all_greedy本身不等于无penalty。

Triton调用中，prompt侧只有“出现过”参与repetition；output侧的count转FLOAT参与frequency，output出现位参与presence。不能通用地把output count退化为BOOL，也不能把prompt次数加入frequency。8种系数组合给出充分统计表示（不宣称所有输入或实现都必须按该表示存储/执行）：rho!=1需要联合出现位U=prompt OR output；frequency!=0需要原生cast后的计数值；仅presence需要output出现位。frequency计数可导出output出现位。仅repetition分支可以合并prompt/output出现位；全中性系数不再依赖历史，但删除零项/恒等运算本身仍须处理signed-zero/FTZ/NaN及原生精度合同。

在合法token0..V（V为padding）、输出-1先转V、每bin不溢出INT32的声明域，request j的第d条target draft行具有C[j,d,v]=C_committed[j,v]+count(spec[j,:d],v)。零draft请求不产生target-history行；不能用固定B×K推断所有row owner。可条件性共享每请求prompt出现位和已提交output基表，再加draft前缀差分。直接按每v扫描前缀会新增sum(V*n*(n-1)/2)次相等判断；小直方图/列表/bitmap也有构建、寻址、同步和发布成本，不能把复用看成零成本。

跨迭代bank只提交验证后实际输出的token，包含实际发出的recovered/bonus token；不能持久化全部draft提案。回滚/截断要按重数删除或重建，槽位复用要切换request/prompt epoch。只有union bitset时，重复token删除和prompt保留需要额外重数/来源或重建。新持久bank延长生命周期，不能直接把每调用数组的差额当作峰值内存节省。

该整数重组不允许重排penalty的FLOAT运算。两个参考反例中，BF16值0.10009765625和FLOAT rho≈1.7采用“先舍入倒数再乘”与直接除法结果不同；BF16值2^25、frequency=presence=count=1时，按序两次FLOAT减1得到33554432，先合并成减2得到33554430。它们是声明IEEE参考策略的反例，未证明当前原生编译器采用哪个reciprocal/FMA/FTZ策略。

条件性融合点包括：把output count>0放进已有count消费者，省掉独立BOOL plane；prompt producer按所需出现位/联合位特化；把penalty结果直接交给选定mask/MinTokens/有限forcing/argmax图。后者仍保留v108的恢复谓词与A/B/H状态，不能省掉有读者的完整处理后logits、raw、entropy、random/recovery或logprob结果，全部mutation/error/参数/捕获epoch仍需绑定。

显式Triton逻辑基线（I32/BOOL/FLOAT为4/1/4B）中，两路zero/count→mask和penalty为32TV字节，加独立argmax为36TV。内联output_mask消除6TV接口预算，成为30TV；再在全部读者允许时移除penalty→argmax私有FLOAT写读8TV，成为22TV，另计新谓词/归约/元数据。该数字未计历史读取、原子更新、contiguous/gather/H2D/padding、参数、bitmap构建或必需输出，不能当作HBM、实测时间或可独立叠加的F35收益。

源码私有辅助plane值大小为10TV；每请求I32基计数加packed prompt位图的参考大小为4BV+4B*ceil(V/32)，另有draft差分与状态成本。这些是所选表示的值大小，不能当作实际峰值/保留内存。B/T=2/10、4/10、4/20只沿用保存的shape/capacity参考，不证明实际有效draft数。

Triton缺失分支不能直接继承上述合同。只读取回一个缺失的utility文件，并核对两份caller SHA：fallback使用INT64的[T,V+1]计数，再返回[:,:V]视图，并把repetition交给独立custom op；其辅助count/mask值大小参考为18TV+16T，不能套用Triton的32/36/30/22TV流量预算。Ascend的HAS_TRITONFalse直接调用Sampler.apply_penalties，未做主路径的repeat_indices gathers；若参数仍是请求B行，B=T也不证明owner相同。例如draft数[0,2]的两条target行都属请求1，而未展开请求行是[0,1]。

对这条fallback的形状推导，以[官方scatter契约](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.scatter_add_.html)及[广播/原地运算契约](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html)为条件：scatter索引不自动扩展请求行，普通广播也不等于按draft owner展开。文档版本不证明目标Torch/Ascend实现；实际fallback可达性、是否有已展开输入、custom repetition的原生dispatch/精度仍未绑定。本轮没有执行或修复该分支，未宣称当前负载存在运行错误。

理论上限为SYMBOLIC_CEILING：V>=3、1<=L<=2^20、rho1/frequency1/presence0、无旧计数/摘要/缓存、精确word-query、任意BF16可表示logits的声明域。历史全为token2、logits=[0.5,0,-1024,...]时ID0胜出；任何未读历史位置改为token0会使ID1胜出，改变任何未读logit也能改变ID，且可保持BF16表示。因此这个域的最坏输入需要读取V个logit和L个INT64历史word。已物化FLOAT输入为4V+8L字节，保留BF16并精确widen的另一输入图为2V+8L。选定一个共同存储边界且容量上界为U时，T>=(s*V+8L)/U（s=4或2）。不把旧FLOAT界套到BF16图；有warm bank、隐式producer、压缩或预先摘要时须重建界。U尚未绑定，没有数值物理时间、MFU或提速。

CPU参考覆盖81120个请求上下文、206856条精确prefix计数核对、64组含零draft的ragged布局、8种参数角色、4个错误整数改写和2个FLOAT策略反例。它们检验局部公式与证据一致性，不证明实际输入/flags或原生kernel。未新增NPU工作、模型执行、候选关系或原profile归属；原9类/1352次关键字段、全模块最低MFU与整体理论物理上限仍未完成。
