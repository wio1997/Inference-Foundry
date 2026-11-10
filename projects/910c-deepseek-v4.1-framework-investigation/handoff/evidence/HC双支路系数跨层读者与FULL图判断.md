# HC双支路系数跨层读者与FULL图判断

介绍：把 FFN premix 提前交付扩大到 attention＋FFN 两个 HCPre，追当前系数尾部、跨层读者与 FULL 图的实际框架合同。保留原系数计算、每个中间 BF16 舍入与原语逐位一致要求。

到了哪里：15当前源码 identity 全部匹配；按精确行段读完相关系数尾部、target/DSpark premix初始化、Engram更新、aux快照、final/MTP读者、图包装及事件更新入口。每rank516对 attention/FFN 源一致链，目标480对、草稿36对；144双parent价格情景与16 PMU压力情景。CPU AST核对两入口，6个有限正常DAG/2004必需顺序对、3个破坏协议的反例；不是NPU并发、捕获或全分支穷尽证明。

结果：修正首层资格。target `model.py:1552–1553`、DSpark `dspark.py:260–261` 都先创建 FP32零premix，再令首项1.0。随后decoder attention用传入premix，FFN用同block的attn_pre，返回ffn_pre交下一层。因此这两个真实模型入口的首层也有premix。旧价格中“no-premix firstattention”只能保留为一般None ABI回退，不能描述本配置的首层；历史原件不回改，本结论覆盖该解释。

系数没有变死：原stage2输出newpre/post，随后softmax＋epsilon及固定次数行/列Sinkhorn得到comb。同phase HcPost读post/comb；下一phase/decoder读newpre。提前collapse只能改交付时序，不能跳过仍活跃的系数GEMM/归约或省掉208MiB原workspace。保留全部原HCPre甚至重复的旧Y计算，是本次保守价格模型；原算术未被“数学等价”替换。

保守汇合点可以统一：每次原HcPost之前等待本次整个系数作业完成，就支配了后续aux mean快照、Engram更新、下一premix、guarded MTP fullstate拷贝/两次AG和最终collapse。aux快照仍在Engram之前，不能移动。Engram更新后的四路残差才是下一attention的输入，辅助流要等这次真实写入。final `hc_collapse` 的Torch `.sum` 不可仅凭公式复用HCPre的四项左折叠。

FULL图条件不是开关：图包装实际捕获runnable及pool，NONE或模式不符时直接执行；现有图更新ExternalEvent在等待/重置后放行host metadata更新，不表示HC系数完成。已有Engram同图producer有SP/并行保护，冻结Engram overlap为false；MC2又关闭shared-expert overlap，不能借其事件/流假定HC自动合法。新方案需独立本地系数流，main前缀完成后fork，保持oldpremix、残差、全部输出及workspace到真正完成，再于本phase Post前join；FULL里fork/join必须进入同一图并守住地址/代次。跳过或异常退出也必须join已提交工作或保留owner，不能等未record事件。

两个parent分别重新计算fork/join。目标HCPre总时长34.719/35.068ms，为3.6927%/3.7722%窗口。各支路e5/h5、无新增下游争用时，条件净省2.6717%/2.7395%；若每支路下游多慢40us则−1.4126%/−1.3910%。这些是两个配平镜像负载短窗的条件价，不是原请求收益，也不能与QKV/head/O投影/frame百分比相加。

用已有PMU再收紧预期：EP0目标960次都有AIC/AIV字段；EP8有903次，57次N/A必须留未知。把原AIC duration仅当作新增争用的压力代理，e5/h5下已覆盖调用仅约0.442%/0.420%窗口；取两engine记录的max则约0.102%/0.095%，e10/h5转负。它们不是实际新争用、独占phase时间或硬件下界；EP8未覆盖的净值未外推，不能用代理直接停止整个方案。首次脚本把N/A强转float失败，完整错误脚本/原因已保留，修正为None并逐调用列缺口，未将其填0。

待确认：真实原语位、实际FULL事件/捕获/allocator与workspace、每支路提前成本及争用，普通/大prefill/草稿/zero-query与完整请求曝光。缺失的真实争用会改变去留，后续只需定点补这类参数；不能继续按零争用乐观价认定值得开发。已停止：误判首层None、借用metadata/Engram事件作系数done、缺producer fence、缺异常/跳过cleanup和删活跃系数输出的具体方案。已确认可实现：本轮新原语0。

下一段沿新到达的真实Engram更新入口，查全局lookup rows→replicated WKV→rotated gate→下一HC输入的大交付，先绑定完整布局/权重集和既有整段价。全链路仍未扫全；本轮只读源码/旧数据、本机AST和有限DAG，无NPU、框架进口、模型请求、新profile、服务动作或显式远端写入。

证据：`hc-coefficient-consumer-{dependencies,source-review,source-excerpts,frontier}.json`、`hc-both-delivery-price.json`、`hc-fork-join-cpu.json`、`hc-pmu-a1-invalid.json`。
