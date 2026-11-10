# 同代CP共同metadata的实际生产与价格判断

Goal ACTIVE；loop040继续，源码/价格首段完成。每个替换原语逐位一致，全链路未扫全。本轮新增准入0。

## 介绍

从runner输入准备的最终交付出发，范围是同代CP区间、qsl、CPU长度与device长度/suffix/mask，经过各资源组装、异步metadata发布，交给compressor、QLI和SMLA。复用已完成的阶段profile，不开新采样。

## 到了哪里

[当前9源身份与已读行段](common-metadata-frontier.json)：runner3822–3883每代新建batch字典，每KV组独立物理字典；CP112–167先构造global，再调用legacy helper、qsl H2D、device suffix/clamp/where，之后才进base的batch命中。qsl来自CPU，因此legacy helper1180选择Torch fallback。说明“base已共享”不等于“CP包装层已共享”。

base已共享batch、C2 lengths、RoPE和相同key的native task，不能重新计为收益。另一方面，SMLA/QLI闭包仍捕获各次common.query_start_loc和max attrs；只共享返回对象会漏掉这些输入。组别的block table/slot、C2 ring、skip与zero-query任务均保留。已有fused-groups slot映射也保留。

接续期间另一流程已发布同次采样的详细Host账单，保留其2f0d0c3提交及所有原件：输入准备Free中155.347ms仍无cpu_op覆盖，长度/条件/掩码48.590ms、视图27.866ms、小张量创建26.902ms、复制/转换/pin23.386ms。aten::where自身去嵌套墙钟10.349ms，其中8.359ms与Free重叠，均为11轮/卡平均；不是每轮10ms，也不是精确CP函数。进一步支持共同host组装的大范围，不能把31ms全部叫where开销或把各子表再相加。

V1 executor在submit后等待inputs_ready及buffer_reusable；release在主流记复用事件，没有join全部producer。因空query/空cache可跳过QLI消费者，原语完成不能从“所有读者都等过”推出。新common frame如果更早覆写主流输入，需要同时覆盖producer完成和consumer末读；只增加新producer的复用等待不够。旧尾部观测不作普遍安全证明。

[现成profile价格](common-metadata-delivery-price.json)来自同次16卡11闭合循环，EP0/8协调结束→首metadata提交31.790/30.984ms每轮，与Free交集27.717/26.925ms、243where。边界并非精确Python函数，243次未逐条绑定；profile overhead、库调用、回压和其它工作都可能在内。

## 结果

**待确认，值得继续**：同代共同plan加每资源组装/发布。先保留首producer原计算、私有CPU快照和各builder稳定device发布；后续共同device生产/合并发布另验精度和寿命。前者与历史CPU-plan能力重叠，不能重复加旧25次H2D/length-elision信用；新收益只认同一最终方案的增量。

假设仅25%的提交前墙钟属于可共享工作，其中50%真正暴露，再付0.5ms新增非重叠成本，条件净为3.474/3.373ms每轮，11轮为38.211/37.103ms；相对各DP同窗约4.553%/4.420%。72个情景含负收益，均非预测或已兑现。实际归因、曝光、成本仍未知，因此没有完整请求收益；不能借旧227waves、原8条不同prompt或15.792秒无profile参考直接放大。

**已确认可实现**：本轮0。源码commonness和价格空间不算逐原语对拍。

**已停止**：按shape跨代缓存整份metadata、跨物理资源共享整个返回对象、用CPU上界替device长度。包含完整epoch/动态attrs/物理组装与fence的新协议仍保留。

下一步具体缺口：同代输入的全部公开alias/mutation与闭包attrs；canonical frame的producer_done+consumer_done栅栏；空query/emptyDP/error/abort/FULL事件集合与顺序；逐原语qsl/length/slot/native packet及consumer输入；当前stock逐caller归因和无profile净价。继续这些大段缺口，不为单where内核单独补NPU参数。

本轮远端只读磁盘源码，本机仅stdlib分析旧数据；NPU、模型加载、请求、新profile、显式远端写入和服务动作均0。
