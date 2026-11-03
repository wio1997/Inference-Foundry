# Profile 主导复核 v7：MoE、draft 初始化、context KV 与通信

本轮补查了 F21、F28、F29，并细化 F30 的实际调用来源。**原有32条候选现在都有分范围的 profile 复核；这不是整模型机会枚举完毕，也不是模块或分支闭合。整体目标仍未完成。**

更正 v6 末尾的待办描述：F29 是 DSpark context KV、RoPE、缓存及索引复用；logits/Markov 属于 F30。候选主台账的名称没有改变。

## 实际执行与阶段分离

对固定 EP3 原始库，按 DSpark 提交栈关联2671条可连接的 API、3307条计算记录，保留全部匹配。进一步只查65个已知调用的完整父栈，得到：

| 调用种类 | dummy 祖先已确认 | proposal 祖先已确认 |
|---|---:|---:|
| 初始 repeat | 1 | 12 |
| context KV 投影 | 3 | 36 |
| base LM head | 1 | 12 |

这证明调用上下文，**不证明它们是正式客户端阶段、独立请求或某个 graph replay**。不能把全部13组都当请求阶段，也不能把 dummy 的工作直接算作在线优化收益。

补查165条 HCCL API→CONNECTION_IDS→COMM 连接，全部各匹配一个通信记录、rankSize=8：embedding AllReduce13，decoder 的 AllGather39/ReduceScatter39/AlltoAll36，draft collapse 输出 gather13，base logits gather13，上下文投影 gather12。模型计算任务子集中仅有1条 base-head AllGather，而通信表实际关联到13条；缺少 COMPUTE_TASK_INFO 关联不表示通信没有执行。此查询仍只覆盖一个历史 rank，不是全部参与者和消息身份的证明。

## F21：已有复合 MoE 的内部重组

1280个保留 target 窗口各有一个 DispatchFFNCombine：导出输入M为643或1023，局部专家24、top-k6；draft关联到39个复合任务，M为2或3、局部专家8、top-k3。8次选定锚核对包含6个不同任务，均绑定到 `moe_comm_method.py:591`；在固定源码中，这是 non-Mega、enable_fused_mc2=1 的 native 分支，输出计数使用预分配缓冲。

因此，dispatch→GMM1→激活/量化→GMM2→combine 应研究**已融合内核的内部排程**。既存合同已显示按专家组的 dispatch/GMM1 重叠；需要研究的包括激活和GMM2交付、cumsum通知、计数发布、缓冲最后使用和分组协议。不能把现有复合调用重新计为尚未融合，也不能仅见零行就删除同步。

另从166既有同调用快照复核4736条记录、296个完整cohort。正式C8部分单独包含target80组、draft7组：

- target 的每接收专家行数观测中位7、最大7836；各cohort接收rank最大/平均行数比的中位约2.66。
- draft 的896个接收专家观测中808个为0；7个cohort的接收rank最大/平均比约4.53–5.07。
- 这些是计数分布；不是延迟、独立试验、硬件利用率或已定位的慢因。旧169输入M643/1023与新166正式样本M400/1023也不同，禁止把两次数据拼成同调用。

固定“每个分配独立执行两次稠密GMM、无稀疏/结果复用/替代算法”的模型下，每分配工作为 `5120×4608 + 2304×5120 = 35,389,440 MAC`。对应条件时间下界可写为 `max_r(n_r×35,389,440/P_r)`，其中P必须是匹配INT8及资源域的能力上限。当前没有166同调用阶段时间和对应硬件容量证据，故保留符号界，不用旧169时长计算效率或提速。原 task-based 捕获的 pcSampling 为 off，静态源码不能补出内部阶段耗时。

## F28：先分片再复制四路，是更明确的结构候选

实际记录显示 draft 先 repeat，再 padding/SP：20行复制成四路、pad到24行、每rank首个HCpre为3行；另一组是15→16→2。每个按 repeat 顺序关联的组包含6次HCpre和6次HCpost，对应三层路径；排序分组本身仍不是张量身份。

定点 helper 代码明确：sp_shard 只在dim0补零再取本rank分片；sp_padding_mask只依赖全局token数、按相同规则分片并将padding行置True。可据此研究：先保持全局行数的mask准备，再对单通道embedding分片，最后只复制本rank四路。它保持上述token/channel值映射，不需要改变HC的浮点归约。还须保留input_ids对齐、premix、初始residual、功能性custom-op的clone/alias规则和图地址合同；target已经先分片再repeat，不能把draft的机会重复计给target。

仅作逻辑张量大小比较：20行原repeat输出819200B，改为本rank3行repeat则122880B；15行对应614400B与81920B。**这些不是实测HBM流量、峰值显存或节省时间。** 首层y旁路、权重折叠仍是另一类数值候选，不能由实数恒等跳过HF32/FP32/BF16及特殊值约束。

还登记了跨卡变体：当前标准embedding将mask后的词表分区贡献汇总为完整行，随后又做SP分片。若reduction与SP分组、rank排列和精度合同兼容，可研究耦合ReduceScatter直接生成本rank行。不能全局替换其他需要完整embedding的调用，也不能与已有PCP交换重复记收益；实际group所有权和collective数值等价仍待证。

## F29：已绑定完整的 context KV 物理链

39条投影链的物理context行数为20、8180、4760、24，每条各匹配一个norm、native RoPE和cache scatter。链为：

`BF16投影512维 → 全512维RMSNorm → 64维尾部RotaryPositionEmbeddingV2 → cat(448,64) → INT32索引对ScatterNdUpdateSk写BF16 cache`。

这里的RoPE实际输入x/cos/sin均为BF16，不能套用target的InplacePartialRotaryMul约束。定点源码还显示RoPE对原view做copy回写；融合若改变该物化，需保留可观察的别名效果或证明中间值仅为私有。

候选包括norm/RoPE后直接形成cache行、跨层共享同一版本的cos/sin查询、独立层投影打包或逐层准备流水。39链共156条context Index记录，每链4条，说明查询确实物理执行；共享仍需位置/注册表/层配置和生命周期一致。各层权重、512维norm与cache不同，不能合成1536维norm或共享不同层KV值；投影打包与提前交付的机会也不能叠加。

per-layer slot为None/空时，store在投影之后才返回，存在带条件的提前跳过研究点；须先证明无必需副作用和其他消费者，并保留dummy/profile的准备用途。负slot、重复slot、block_size/page stride、context/query cache写入次序及noncausal读取集合仍要保持。观察到scatter不证明其中一定有有效slot或实际写入。

## F30与整体覆盖仍待完成的部分

已绑定13次base head、65次Markov embedding和65次Markov bias矩阵乘及相关TP通信。模型帧筛选不包含所有proposer加法、argmax、rejection和confidence分支；它们仍需与既有gap证据及消费者合同核对。

本轮只为明确的profile缺口补查6个文件片段，未做全库审计、未新增设备采集或推理任务。两次分析查询停止及修正分别保留：一次source-chain数量触发预算；一次解析器猜错repeat行号，改为按实际任务和栈筛选。

详见 [moe-draft-summary-v7.json](moe-draft-summary-v7.json)、[moe-draft-reconciliation-v7.json](moe-draft-reconciliation-v7.json) 和 [module-coverage-gaps-v7.json](module-coverage-gaps-v7.json)。下一步转向26模块的阶段、条件分支和跨迭代覆盖审计，核实模块宇宙是否完整，补齐当前台账没有候选链接的框架、前缀/KV、调度、输入准备和响应路径。空链接不能解释成无优化，也不能以32候选复核代替这些工作。

**whole_modules_closed=0，overall_complete=false。**
