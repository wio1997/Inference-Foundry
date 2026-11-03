本轮围绕DSpark上下文预计算与query路径的缓存发布关系。先检查旧台账：v34/v35已提出按层延迟准备和同一WKV权重拼行，并已提示context-before-query写入顺序。本轮不把这些旧机会重新计为发现，而是补齐拒绝后槽位冲突的条件公式，以及它对上游context主投影工作量的影响。

复核32份已保存任务投影后，得到104组严格的源码顺序模式：每组3个context KV计算/写入周期随后是draft repeat，其中88组之前有context主投影链，16组为dummy调用域。312个已有context周期的任务键、开始/结束时间和stream均逐项核对。这只证明保存的执行顺序，不能证明实际request、槽位地址或可回收关键路径时间；原始执行归属不提升。

定点源码显示，metadata kernel把全部context positions和slot mapping复制到输出；num_rejected并不裁剪这个复制。query的线性cache位置从seq_len-rejected开始，RoPE位置则从最后有效context位置推导，二者不能互代。context和query的标准writer指向同层swa_cache_layer的kv_cache[0]；实际加载对象、地址、字节写入范围和group/page epoch仍须运行证据。初始化时sample_from_anchor选择配置K或1+K个query，公式使用实际query_count，不把它固定为当前scheduled K+1。

在每请求context逻辑位置为[s-n,s)、拒绝数0<=r<=n、query位置为[s-r,s-r+q)，并且slot映射一致、页映射无别名、跨请求所有权/COW正确的域内，交集为[s-r,min(s,s-r+q))，重叠数恰为min(r,q)。DCP owner/local压缩映射有显式逆式，因此跨owner相加仍得到该数量；单rank不能直接套用总数。实际metadata若有无效槽、页别名或代际不一致，须计算物理slot/字节集合，不能直接套公式。本轮25,344个CPU映射参考用例，以及独立3,240个区间用例和6,000个DCP逆映射用例通过，均不属于原生运行验证。

具体反例中，context位置[7,8,9]、r=2、query位置[8,9,10,11,12]在8/9冲突。让旧context写入晚于query，会覆盖新的query值。只让即时attention读到正确值仍不够：晚到的context写入还可能破坏持久cache最终状态。保留两条物理写路径时，应保留冲突字节上的context→query顺序及各reader的RAW依赖；这不要求所有无关槽或后续层的context工作全局串行。no-localQ也不等于无query KV写入，旧profile的cache-only rank分支继续保留全局KV工作。

新的工作量机会是沿读者需求反向裁剪。若某context写入在被保证完整query覆盖前没有任何读者或状态/错误观察，其私有KV投影、512D norm和RoPE工作才可能随该死写入删除。更上游的F27主投影/5120D norm是否可删，取决于所有层剩余context需求与其他bank读者的并集；只在一层被覆盖不够。metadata/ID计算不读context activation，可研究从当前包含bank copy的入口中拆出以提前获得需求掩码，但不能直接搬动整个带seed、metadata和bank副作用的方法。dummy、无context、无完整overwrite保证的分支不据此裁剪。

在普通完整GEMM、无预计算输出复用、精度边界相同的声明域中，所选图的WKV参考工作为2*5120*512*各层剩余context/query行数之和，main参考工作为2*15360*5120*全体剩余main需求行数。两组纯参考例子中，context/query行数12/10与24/20、给定拒绝数组后分别有3/8行覆盖：单层cache显式值写入差额为3072/8192B，WKV差额为15,728,640/41,943,040 OPS。这些不是原profile实际拒绝数、HBM流量、物理时间下界或提速。冲突掩码、逆行映射、压缩/拼行、发布、并行竞争和丢失的复用都要计入同一图，不能与旧rowpack、cache写入和上游裁剪重复计收益。

新增条件关系REL113-01（F27/F28），细化已有HG85组以及F29/F31合同；它表达query最终写入需求向context主投影的反向传播。现在133条条件关系、25组、462对仍待核对；35项候选、353条语义记录/26暂定模块保持不变。9类/1352次原生执行、849缺PMU、16未绑定phase图和3136含糊归属均不变，0个整模块关闭。原生Pad小探针仍等待单独明确回复，本轮没有远程操作、模型、NPU计算、捕获、replay或实现修改。整体目标继续进行。
