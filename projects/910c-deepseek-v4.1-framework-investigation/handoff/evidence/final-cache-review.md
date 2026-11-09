# 末层物理cache与原行映射判断

介绍：继续body39的整段producer，保留body38全行输出给入口39的aux。这里判断少算、少搬的合法边界，不恢复已撤回的两层删除。Goal ACTIVE；loop-016仅完成限定源码/价格判断而PIVOTED，下一段loop-017继续文本输入、embedding与DSpark context。**每个替换原语逐位一致；本轮没有候选设备准入。**

到了哪里：16份实际运行源码已核，15份匹配本地，`v1/kv_cache_interface.py`补取当前件。只读171，无远端写入、NPU、模型加载、推理或新profile。原始源码路径/SHA见[身份](final-cache-execution-identity.json)，逐段前沿和56窗口复算见[结构化判断](final-cache-frontier.json)。serve脚本SHA仍为c297…，显式block-size128；没有把未显式设置prefix当成有效默认值证明。

结果：两种直接配方已停止；保留完整原cache生产、按需求生产query/O/FFN仍待确认，值得进一步判断。源数据支持物理边界、原行号和有成本情景，不支持数学等价或整网输出相同即可放行。

## 物理写入到未来读取

介绍：`SWA128`是attention窗口，不是cache的模128环。要先追坐标和发布资格，才能删投影/写入。

到了哪里：

- `cache_config.py:147–227,248–274`建立四个layer-outermost物理槽和全局block-ID池。执行原拓扑helper的标记对象重放得到body39属于slot3、framework group11；与source20 longKV、其它SWA共享backing槽，但活跃group使用各自分配的物理block ID，不能按同tensor就认领同块复用。标记容量是占位单位，没有作为字节证据。
- `model_runner_v1.py:5389–5462`用真实padded page stride建立view；`dsa_v41.py:227–242,297–308`以`[block,row]`调用A3 ScatterNdUpdateSk，`[-1,-1]`为skip。常规DCP1 slot是`table[req,position//128]*128+position%128`，再由builder拆成二维坐标。非负null block ID本身不等于PAD。
- `_preprocess`的query和cache域分开。CP rank无query仍调用`_update_caches`，对全局cache行投影。SparseFlashMla读取同group的block table、device seq_lens和query_start_loc；body39仍读KV20/topK36，无自己的longKV或index写入。
- `kv_cache_manager.py:525–546`先以`total_computed_tokens-num_in_flight_tokens`回收；`SlidingWindowManager`只释放窗口之前的**完整块**，防止在途和拒绝回滚读旧行。已提交3328时skip3201，但按128页只可释放到3200，保留3200–3327；这是这个边界的例子，不能倒推本次大chunk只需写尾128。
- prefix命中使用连续窗口尾页，eagle还要匹配并丢弃一个peek块；缓存发布又有dense/sparse retention、replay/shared-prefix边界。合法后续请求可在256处恢复并计算query256，需要原缓存129–255和本次新行256。若只写旧请求尾3200–3327，却照常发布前面页的hash，便可能命中未生产的值。此为合法分支反例，**不是声称原C8发生了该命中**。CoW和所有权仍必须沿原协议保留。

结果：**“只写最后128行、分配/缓存发布不变”已停止**。没有把所有cache裁剪可能性穷尽：改变发布资格、重算协议或页布局可形成新设计，但要支付额外成本和重新证明所有读者。当前保留全原writer的方案不依赖未知prefix有效设置，故不为该参数重跑服务。

## 原行到query、topK和返回残差

介绍：选中行减少不能改变其query位置、topK读者或TP/SP返回所有者。

到了哪里：原CP令`P=ceil(N/8)`，全局行`g`的owner为`g//P`，旧local行为`g-owner*P`。单请求N3328末行g3327因此在rank7、旧local415。原`shared.topk_indices[:qr.shape[0]]`只适用于原连续CP切片；压成1行后`[:1]`会读旧local0。所选topK必须用旧local415等显式索引。原`restore_tp_heads`按等长rank-major all-to-all交换所有head，后面补FlashComm suffix再O投影/RS；压实后需要全局行ID、零query rank参与和原SP residual/pre_mix归属。

七个real eager DP窗口的计数与原head行数全部吻合：prefill请求只取末行，已有decode/spec请求每项6行都活着。混合波不能将六行验证改成一行，也不能把scheduler字典顺序当成实际InputBatch/QSL顺序。后者仍需从具体元数据建立映射。

每个原request/CP交集若选连续尾k行，保留权威device seq_len L，则逻辑位置仍为`L-k+j`。有洞选择不能仅改qsl；保留全行或提供显式位置/新metadata。即使逻辑集合相同，SMLA的actS1、M分块、S2窗口/partial划分也可能改变softmax归约。Q GEMM、HC、EP分组和GMM也同样不能按小M“自然相等”。保留原M/tile索引、原BF16舍入和原累加树的tile-mask只是一条实现方向，尚无设备证据。

结果：**“新topK前缀+原CP restore直接复用”已停止**。完整writer+显式旧行映射+原数值几何的联合范围**待确认**；原失败compact FFN不能再作为这条路线的实现。

## 保留原cache成本后的价格

介绍：不再将cache全部免费计入query/O/FFN方案。

到了哪里：56个原C8 eager rank窗口逐个定位body39的HcPre、输入RMS、Wkv、KV RMS、KV RoPE和SWA scatter，顺序和唯一匹配均检查。保留六个原语的旧时长，AG/等待/存活query计算/返回通信/新实现仍按最有利放松。按原8rank时钟约束重算DP frontier，再对各波取独立最有利DP并传播到八个TTFT。

结果：保留这些原语时长后，P90有利组件包络32.269702ms，**2.163059%**；更宽的整body39全免34.739908ms/2.328639%仍是另一种更松情景，没有相互覆盖。新增每eager wave0.5ms时，rho=.5得到条件净14.134851ms/**0.947468%**；rho=0为负，所有负值保留。rho表示真正可省且暴露部分，不是已经测得50%。这里包括decode-only和mixed eager波，所以仍不能把活着的验证行算作实可删。

这只是保留原六个primitive耗时的原schedule敏感性，**不是硬件上界、预计收益或对改变cache producer的大方案的剪枝**，也不能与前次terminal预算相加。它未达到可开发准入：缺实际多请求行映射、保原几何的逐位实现、端到端可暴露量、新成本及六普通负载匹配值。当前候选保留这些具体缺口，继续未扫producer；不因一个假设情景低于1%就停止整个联合parent。1%不是用户新设硬门槛。

## 下一条交付

介绍：返回剩余文本输入/embedding与DSpark context的大交付，不能用当前候选有名字代替全链路扫描完成。

到了哪里：loop-017从实际MM wrapper预计算文本embedding、raw input IDs/positions、target hidden/三aux进入DSpark context的消费者和所有权开始。先找整段少算与重复交付，再匹配旧profile；保留既有小writer低值结论，不重复定价同一预算。

结果：全图仍未完成；本轮新NPU0、服务请求0、新profile0、远端写入0，服务和环境未变。
