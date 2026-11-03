本轮完成一条此前未进入中央关系台账的F30–F31条件分析：DSpark置信度/验证预算与metadata构造、发布。它不是新增主候选，也没有宣称动态裁剪在原负载启用。先只读查询32份固定SHA的缓存profile，保留183813条相关Torch API记录及原始键/源码栈；精确CID数组关联得到104个seed publication、520个ID publication和1232个next-ID/count范围TASK，其中1144个有COMPUTE定义、88个没有。对confidence、scheduler、keep-length retrieval及confidence head的源码位置查询均为0命中；这不是配置为False或没有消费者的证明。没有服务器、模型、Torch/CANN/NPU或replay运行。

profile不能单独决定值语义，随后仅复用3份固定hash的proposer/utils/worker局部源码及既有metadata协议证据。选中padded路径中，当前target sampled IDs先产生next-ID和valid-count，再进行拒绝修正、draft输入和metadata准备；merged draft先运行decoder，再完成Markov序列，最后才可调用confidence scheduler。因而本轮晚出的confidence keep长度不能反过来裁掉已经需要的当前draft metadata或noncausal query/KV行。当前target接受状态、当前draft、下一次target验证的proposal长度、下一次验证后的接受状态是不同代的数据。

全局TopK的结果也不能直接当成待验证token位置。当前实现把TopK索引换算成request IDs、scatter-add得到keep-count，取出结果时按每个request截取token前缀。例如一行4个survival都为1，min_k=1、额外选中原位置3，是一个允许的平分选择；代码最终keep=2，发布位置0、1，而不是0、3。枚举65028个数学候选子集，其中14182个符合TopK值顺序，8938个选中位置集不同于最终发布前缀。这些是源契约下的反例，不是native TopK实际tie策略或当前数值的恢复。

metadata必须基于最终前缀和请求映射。B=2、D=3、budget=2可得到keep=[3,1]，对应纯decode query长度[4,2]、QSL=[0,4,6]；用统一[2,2]代替会得到[0,3,6]，总token数同为6，owner行归属却不同。将请求顺序反转也会改变QSL。相同总量、shape或图bucket不能证明可以复用原metadata。实际token-list宽度、consumer cap、zip长度、request ID/generation、调度后row map、CP/DP/group及全部plan输入均需匹配。零draft提前返回、scheduler None/关闭、宽度变化、非parallel continuation分别保留条件，未推定旧keep bank已更新。

confidence keep与验证后的valid-count也不可互换。同一个待验证k可以因后续拒绝/丢弃得到0..k+1个有效结果；它不能预测下一轮接受状态。当前next-ID helper还按字面谓词x!=-1且x<V计数，再取count-1位置：有洞的[7,-1,9]给出count2、next=-1，不能改成rightmost-valid的9；小于-1的负数也仍可能通过原谓词。候选必须保留原INT32 next-ID、INT64 count、backup、discard及独立GPU/CPU发布消费者。

可研究的跨边界组合是：在动态分支确实启用、完整request/width/epoch契约成立时，让最终keep-count生产者同时生成兼容的下一次验证prefix-scan/owner输入，或向metadata producer传播精确需求。它不等于把opaque SMLA/QLI callback直接塞进confidence kernel。既有异步submit、input-ready/reuse/frontier、capture固定地址、零query写入和旧reader退役仍须满足；只需真实值/range依赖，不新增一个无证据的全局barrier。周期预算重算要先得到本轮正确budget，其他步可使用已知budget，不能一律等同处理。

2834组B0..4、D0..5的布局参考用例核对了前缀、QSL和owner。明确选择的新ABI若保留INT32 k[B]、QSL[B+1]和可选owner[Q]，逻辑内容为4B+4(B+1)+4Q；这不是实际分配、HBM必要流量或现存device scan成本。当前host int(k)/prefix retrieval仍是观察者，尚未证明可以删除它，也没有凭新ABI认领4B读取、host同步或关键路径收益。原FP32 confidence/sigmoid/cumprod、globalTopK ties、periodic state、RNG及score消费者约束均保留；单独的embedding复用和F31 summary融合是既有候选，收益不重复累加。

中央新增REL129-01并补充既有HG73-TAIL和HG85-VERIFY-CONTEXT-DRAFT，条件关系135→136，待核对460→459，联合组仍26。M20-OP-10及M21-S10等既有分支获得跨代证据，语义行仍358/26；局部13节点/15边的代际DAG未当成新增native算子或全图补齐，部分全图仍854/1346/48提示。6465项前置核验通过，全部本地查询已结束。原9类/1352次字段缺口、849缺PMU、16个phase未绑定图、3136项含糊归属均不变，整模块关闭0，物理MFU及最低时间下界未建立，整体目标继续。
