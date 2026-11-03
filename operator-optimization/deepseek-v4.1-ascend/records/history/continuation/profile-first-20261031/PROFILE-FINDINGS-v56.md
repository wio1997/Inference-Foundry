本轮直接分析DispatchFFNCombine内部融合、接收端工作量和通信下界。读取8个定点operator/kernel/tiling依赖、2个历史哈希匹配的完整EP计数资料；没有模型、设备或新采集。

DispatchFFNCombine已经融合routing量化、跨卡dispatch、GMM1、SwiGLU/再量化、GMM2、dequant-return和weighted unpermute。profile旁边的DequantSwigluQuant是共享专家支路，不能重复算成routed FFN待融合激活。

内部仍经GM：GMM1的FP16结果→FP32激活/scale→INT8结果→GMM2 FP16结果→BF16返回。外部BF16 dtype不能替代内部FP16边界。C1/C2复用同一GM bank，input/activation也复用；细化流水必须保护旧读者和行/stride/scale边界。每个receiver assignment涉及的三处逻辑中间接口往返为8I+2I+4H=43,520B；这些不是实测HBM字节。

按H=5120、I=2304计算，每个活跃expert的两组INT8权重共35,389,440B，另有77,824B packed scale；两次GEMM每接收一个token-assignment需70,778,880整数运算（每MAC计2）。源码对空专家不调用blockMmad，对n_e≤128的权重请求绕过L2。因此不能把分配的24个expert权重全部当成每次读取。

流水第一切点固定为本地expert数减3，即target21+3、draft5+3。保存的80个正式target计数cohort中，前21个expert包含接收行的中位93.069%；按实际工作量选择切点值得研究。它是计数份额，不是耗时份额。

2560个root关联图层：MC2设备耗时中位304.513μs，共享支路合并前span中位57.301μs，MC2开始到最终add结束span中位384.4475μs。即使不改变工作、假设输入独立就绪且资源无限，能重叠掉的共享支路也至多受该premerge span限制；最终add不能被并行到两支完成之前。实际两支共享24Cube/48Vector及带宽，不能把57.301μs直接当收益。

通信模型区分assignment与唯一token-destination。当前dispatch读取H+32=5152B的INT8/scale记录，BF16返回数据为2H=10240B。可研究同一token发往同一rank时只发送一次、在本rank fanout；但现有计数矩阵只有边缘统计，只能给唯一pair数量区间，不能给实际去重率。提前在receiver做weighted partial combine还会改变舍入和求和顺序，并需要传递weights，不能作为无条件等价融合。

本轮形成5个条件方案：更细的Cube/Vector中间交接、按接收工作量切流水、按目的rank合并token传输、受数值条件约束的提前combine、与MC2兼容的共享专家重叠。它们细化现有F21/F22/F23，没有新增主候选或实测加速。

理论形式为T≥max_r(6HI·A_r/P_INT8，活跃权重/内存带宽，必要跨rank字节/链路容量，依赖下界)。87个计数cohort与后来的profile不是同一轮，参考硬件容量也未冒充当前实测，不能相除报利用率。下一步继续共享专家及其它热点算子的定量融合/上限分析，整体尚未完成。
