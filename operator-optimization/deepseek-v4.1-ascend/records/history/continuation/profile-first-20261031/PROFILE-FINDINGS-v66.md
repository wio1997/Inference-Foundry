本轮把通信当作算子与数据所有权问题分析，新增17类、11,528次计算表通信节点的条件分析，并检查全部14,720次HCCL调用。32份原始数据库均重新核对SHA；全程使用本地保存数据及5份已绑定原166版本的算子源码。

14,720次调用中，11,528次可以通过数据库内connection、完全相同的时间区间和算子类型对应到计算节点。其余3,192次保留为独立通信调用；通信调度定义也有3,192条，但数量相同不证明一一对应。557,688条内部记录同样不能当作消息数量或额外数据量累加。

通信已按13种语义用途、42种类型/dtype/卡数/count组合建模，加入原始group标签后为61种组合。AllGather的count是每卡发送元素数，ReduceScatter是每卡接收元素数，AlltoAll是每个peer的元素数，AllReduce才是每卡完整归约数组。固定24行图中，AlltoAll count=12288、BF16，每卡完整输入为196,608B，发送给其他7卡的内容为172,032B；不能把count×2=24,576B当成整卡通信量。[HCCL接口定义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC3alpha003/apiref/hcclapiref/CANN%208.0.RC3.alpha003%20%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E6%8E%A5%E5%8F%A3%E5%8F%82%E8%80%83%2001.pdf)、[ReduceScatter定义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha001/API/hcclapiref/hcclcpp_07_0024.html)。接口参数含义由固定代码布局和profile count交叉验证；这些较早文档不证明当前9.2的算法或物理容量。

本轮较重要的重组是把第一段输出投影WA移到token所属卡，再交换结果。

| 条件方案 | 每卡每层代价与收益 |
|---|---|
| 当前先恢复TP heads，再WA→WB→ReduceScatter | WA权重8.389MB，WB权重10.486MB；24行时恢复heads远端发送172,032B |
| token所属卡先算全部8组WA，再恢复TP布局 | 交换维度4096→1024，发送43,008B；WA权重增至67.109MB，额外58.720MB |
| token所属卡计算全部8组WA/WB并完成等价归约 | 有条件免去输出AlltoAll与ReduceScatter；权重额外132.121MB，必须保留每组BF16中间结果和原归约语义 |

在交换行数与投影行数相同的条件下，前移WA不增加每卡常规矩阵乘操作数，输出AlltoAll＋ReduceScatter的逻辑发送量变为原来的2/3。但它把每组权重复制到所有卡，新G8小行数计算的tiling、冷权重读取和BF16精度均需保留为约束。40层推广会额外占用约2.349GB/卡；两级都前移约5.285GB/卡。若20行补齐到24行、6668补齐到6672行，直接在补齐行上计算还会增加工作，必须单独核算有效行裁剪、补零和行映射。

两级都前移也不能把分组结果随意用一次FP32求和替代当前HCCL归约。BF16值[512,-512,1]的两种成对归约顺序可以得到1与0；这说明需要确认原生累加和舍入政策，并不代表已测出当前HCCL的政策。

另有分块流水与调用合并机会：attention/逆RoPE直接写发送布局、AlltoAll到WA的面板流水、WB与ReduceScatter及本地HcPost交接、HC和FP32 premix的混合字节gather、以及三次aux checkpoint gather延后合并。它们保留原始内容、消费者、舍入和发布顺序。HC/premix从两次gather合成一次不会减少payload；P=8时，普通打包与拆包反而增加18q字节逻辑读写。三次aux合并还会丢失提前通信机会并需要保存三份局部值，不能只计算少两次调用。

embedding的AllReduce改成ReduceScatter，在对齐且所有读者都接受行分工时，ring参考发送量可以减半；当前标准embedding明确提供完整首层序列。任何仍需完整结果的读者若补回AllGather，总通信量就恢复原值，补齐时还可能更多。词表最优项摘要、上下文feature分片、Engram按行/头交换则与此前分析的概率、熵、RNG、完整bank和UVA读者约束一同保留。

所有通信字节与轮次均声明为ring或pairwise参考算法；记录中的RING-RING-NHR标签不证明实际只有一条flat ring。当前缺少绑定到这次执行的物理链路、注入带宽、cut容量、时延和HBM/L2流量，故本轮不编造通信微秒下界或端到端加速。AIV通信还使用Vector资源，会与norm、quant和RoPE竞争，计算与通信也不能默认免费重叠。TP8的HCCL记录不覆盖MC2内部EP16通信及CPU/Gloo元数据交换。

共记录20项条件方案或排除结论，建立18个通信语义分支，并向保留的输出投影、末端状态和aux上下文分支增加4项；其他21个模块矩阵记录保持一致。全观察台账现为1,015类、201,431次执行已有显式条件分析，尚有1,067类、85,519次待核对。真实物理容量、所有适用模型分支和3个MemSet待定项仍保持开放，整体Goal未完成。
