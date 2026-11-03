本轮从保存的采样profile范围出发，定点分析默认Ascend的top-k/top-p过滤、概率除指数噪声及argmax，并核对直接上游对照函数。三个文件均匹配已有冻结源码清单；当前profile仍未提供随机分支、k/p、batch-invariant或seeded参数的逐调用取值，没有新增服务器或NPU执行。

默认Ascend实现先做一次softmax，再将概率升序排序。top-k和top-p阈值都来自这份未变的原始概率；top-k写logits并不会更新用于top-p的概率。这个softmax和sort本来已共享，本轮没有将它们再记为新融合。

上游PyTorch实现不同：先按logits排序并截断top-k，再对截断结果softmax计算top-p，最后按排序位置mask。权重[1,2,3,4]、k=2、p=1/2的实数例子中，默认Ascend保留后两项，上游只保留最后一项；四项同概率、p=1/2时，Ascend的概率阈值比较保留全部同值项，上游位置mask只保留两项。同名函数不能直接替换数值合同。硬件能力选择在当前Ascend源码中的两个分支，实际上都进入同一个PyTorch过滤体；这项局部选择已厘清。batch-invariant会委托上游，Triton过滤器体仍需独立核对。

有明确可分析的融合范围。保留相同原P、sort、CDF和两个阈值后，两次masked_fill(-inf)等价于按两个条件OR进行一次写入。阈值有序且非NaN时，可进一步比较P<max(cK,cP)，减少一次整矩阵比较。无效元数据的原验证、gather错误顺序及中间读者条件仍需保留，不能趁此换成不同top-k/top-p语义。

top-p计数也可重组：若保留的既有CDF确实非递减且无NaN，原sum(CDF<=1-p)并将最后mask置False，等于min(upper_bound(CDF,同一阈值),V−1)。V=129280时，二分最多17次值探查/行。它保留原CDF的数值，不要求先改写cumsum；如果融合新的scan生产者，原生归约顺序和边界值还须证明。依赖读取的延迟也可能抵消比较数量下降。

top-k-only可研究在相同已舍入概率上找顺序统计量，替代完整排序并保留全部边界同值项；不能直接使用精确保留k项的logit TopK。所有k=V且p缺省，可在元数据与验证条件成立时作为no-op分支。p=1却不总能同样处理：FP32 softmax出现零概率时，当前实现可能把对应有限但极小的logit改为-inf；[-200,0]的数值模型即展示这一边界。

默认随机采样在global_stream生成指数Q，当前流等待并record_stream后，对probs原地除Q再argmax。保持同一P/Q、逐值除法和比较规则，可将Div与argmax流式融合，省去ratio中间平面。直接公开调用random_sample(P)有P被修改的语义，须保留写回；在forward内部P私有的范围，可不落地这个平面。processed_logits仍需过滤后的logits别名；processed_logprobs是独立log_softmax结果，不能用log(已舍入softmax)代替，后者在下溢时会变为-inf。

不能仅凭实数公因子消去最后一次softmax。两个保留权重为[2,3]时，原归一化总量6与截断后总量5产生的FP32 P不同；本轮固定正值Q数值例子中，复用原P得到并列第一，而重新归一化的正确P使第二项胜出。该例子是IEEE单步舍入模型，不冒充CANN输出或可复现的原生RNG种子。

RNG合同与拒绝恢复路径也不能混用：这里全部请求都有generator时跳过默认整银行draw，否则先draw全部B×V，再按字典顺序覆盖seeded行；v33的拒绝恢复wrapper则总会先draw默认Q。generator对象可能共用状态，不能仅凭相同seed重排。Q为0/Inf时，这里的源码没有恢复kernel的sentinel防护，也不能将那个防护直接移入。

mixed bonus采样最终按temperature<1e-5选择greedy或random ID。保留所有原Q draw和其他score工作后，私有P的Div/argmax可以只处理最终使用random ID的行。该谓词与拒绝helper的temperature==0不能在未证明调用者温度域时合并；小正温度是否能经上游正规化到达这里仍未认定。已存在的all_greedy提前返回也不重复计收益。

成本模型覆盖B=2/4、V=129280的20个声明参数/活动行场景及8种RNG draw情形。只看两段互不重叠的接口：同时有k/p时，两次logit读写参考16BV→8BV；P/Q已存在的race参考16BV→私有融合8AV，generic写回方案为12BV。softmax、sort/merge、CDF、返回分值、Q生成/存储、stream等待和新增launch均另计。这不是实测HBM流量或耗时。在声明的Sort32实现域内，完整首遍排序服务还在，elementwise融合不能凭空消掉它；换选择算法另建成本域。

检查通过7650组精确概率掩码用例、6375组保留CDF值的搜索用例、729组固定操作数race用例，并记录8项数值/域反例。均为源语义与数学验证，未导入Torch或进行NPU测试。

G7补上29行默认过滤/race节点、40行边；当前G1/G5/G6/G7部分图708/1159行已与覆盖入口绑定。计数不是唯一算子或全分支覆盖率。累计353条分支记录、24个语义模块；115条条件关系、24组联合合同，中央仍有480对待汇入或复核。2073个regime/285598次执行已作条件分析，9个regime/1352次执行仍待补；849条PMU缺失、16张阶段未绑定图、3136条模块归属歧义不变。整体目标继续，没有整模块关闭。
