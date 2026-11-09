# DFC返回、裁剪及残差消费者（全MoE仍未宣布穷尽）

## 介绍

沿冻结SP MC2的prepare/finalize追原GMM2→两种Combine→peer返回区→Unpermute→共享支加法→FFN HcPost，再到下一层残差。新增8份源码范围的当前容器SHA均匹配冻结副本；[范围与pin](moe-return-source-pins.json)。代码决定范围之后，才与原空expert发布记录和profile相遇。

## 到了哪里

| 原交付/合法分支 | 实际消费者和不可丢合同 | 融合机会/状态 |
|---|---|---|
| SP `replace_allreduce=True` | prepare按原local行数切mask、补齐BF16输入和router logits，ID单独对齐；finalize仅返回`out[:num_tokens]`，没有TP AG | 下游只读原local前缀。可裁unpermute输出尾行；**待确认阶段净价及原kernel改动** |
| 非SP/未替换分支 | prepare补到DP长度再TP切分，finalize新buffer AG后trim；不能原地复用原输入，shared仍读它 | 当前冻结SP不执行此AG，不能领取删AG信用。保留合法分支合同 |
| Routed GMM1/2 | `dispatch_ffn_combine.h:219`的CType为FP16，固定Tile128×256×512；shared Up另为INT32 | 消去GM交付可以保留原FP16舍入；取消FP16中间值的配方未获准 |
| expert前21与末3 | 固定24专家时epilogueGranularity21；AIV在前段/后段各等待并释放AIC。空expert仍有发布/消费/最终drain | 相同空expert credit压缩变体已有逐位通过但native价值失败，**已停止原配方**；更大排程仍须配对改生产者和消费者，保留原expert ID、21边界、空段、容量截断和结束协议 |
| CombineV2：M≤682 | 原20列块、AIV子核按16行块，等待对应GMM2 flags，逐slice写原source peer位置 | 列块/返回排序可变，但不能删其他列块读者或按sender空跳接收端 |
| CombineV1：M≥683 | 逐expert等待，遍历16个source peer、按counts及capacity裁剪；每row原FP16→FP32→乘pertoken scale→BF16 RINT | 重排可保原BF16值，提前概率乘法或累加若去掉此舍入不准入 |
| peer offsetD→Unpermute | 返回值BF16转FP32，按原TopK顺序逐项Muls/Add，最后BF16 RINT；expandedRowIdx/probs为实际读者 | 按EP分组预求和会改变原加法顺序，不能直接放行。保留原顺序的deferred consumer **待确认** |
| Unpermute→shared+routed→HcPost | runner routed系数为1（1.5已在router）；两个BF16先加成BF16，再送HcPost。四路残差仍需持久化 | 可以在UB/寄存器保留两次原BF16舍入，消去中间GM交付；不是代数展开后一次舍入 |
| ResetTokenPerExpert→CrossRankSync→Unpermute | counts重置和结束代际屏障先发生，随后仍读offsetD；下一代counts带0x800000 ready标记 | **纠正**：结束屏障不意味着整个arena无读者。offsetD要等unpermute末读，跨层/跨迭代融合必须保留分区生命周期和下一代ready |

HcPost的A3分支是`hc_post_d_split.h`，不是DAV_C310寄存器分支：先`x*post`，再按残差路0..3分别Mul/Add，最后BF16 RINT。融合保留这个顺序，不能照搬另一架构的FMA顺序。该附加源码为旧取回件，尚未新增当前容器kernel binary/tiling证明。

SP tail裁剪不能把`M*6`输入route索引界直接换成有效输出行数×6：前缀输出的route可能位于完整排序域高位。应保留原索引域/hidden tile，只另加output row limit。当前八组原下游行域和可删store字节在[摘要](moe-return-frontier-summary.json)。例如DP0/carrier816只消费3/816行，可少写8,325,120B/层/rank；DP0/carrier818消费419/818行，可少写4,085,760B。尚未假定尾行必读六个expert payload，未把字节换算成实际时延。

独立核对已有captured M3 typed joins：16rank、144000行，2518行sender=0而receiver>0；754行receiver全空，最大receiver98、capacity65536。证明不能按sender空跳DFC。它不是C8 eager或当前生产频率。空expert发布与历史相同源码记录及[后续native结果](empty-expert-native-reuse.json)相遇：原组复制已跳过0 rows，仍有每expert SyncAll/credit和AIC drain。旧112窗1%排序参考中位需39层各净省约14.25µs。随后EP16 M3/E24/K6的8个串行原语、前后control native对照已完成：逐位/逐expert counts及eager/graph一致，但两种稀疏case分别比前后control慢0.30/0.94µs、慢2.36/快1.20µs；dense对照只快1.02/2.08µs。原样空credit变体按已有价值失败停止，不重做实验。状态名EXACT_OUTPUT_NATIVE_GATE_FAIL是价值门槛失败，不能误写成位差。

## 结果

- **待确认：deferred Unpermute→merge→HcPost大producer。** 可以让原DFC完成原Combine/reset/屏障后，保活peerD、expandedRowIdx workspace和probs；shared完成后再调用融合消费者，避免DFC持20/40等待aux。消费者逐row执行原TopK顺序、BF16 cast、shared加法及BF16 cast、原HcPost Mul/Add/cast。每有效行最多可消去两份BF16中间值的写回+再读，共40,960B；HcPost四路残差读写和expert返回不可算可删。需新结构化生命周期接口/事件、原语逐位门槛与阶段净价，不把现有deferred机制名当成已接通。
- **待确认：输出前缀裁剪和跨卡unique-row交付。** 已有具体消费者和控制合同；whole DFC inclusive时间不能替代各阶段耗时。前缀裁剪与整段consumer融合互斥字节不相加，任何新控制排程须解释相对已失败空credit变体的新机制，不能重复领取等待。
- **已停止：原样空expert credit压缩。** 复用已匹配source的EP16 native价值失败，不把原控制证明或空expert频率当性能收益；保留完整返回producer及其它合法控制方案。
- **已停止：单独删除shared DQ以争取C8首token1%的路线。** 连原DQ全部免费、忽略新成本/EP等待，条件信用最高0.2864%。不为这一小范围新增量化复用NPU试验；更大的routing/shared producer仍保留实际两套量化逐位缺口。
- **仍待确认：merge+FFN HcPost单独/联合producer价值。** 二者全部免费的条件信用最高1.3537%，无新成本尚需兑现73.87%才到1%；实际工作大部分必须保留，不能将这个数字作预期收益。若不含unpermute的stock小融合无法接近净门槛，就停止该小范围；不剪掉整个DFC返回producer。

以上预算均为v168带observer诊断cohort的原请求/载体调度参考，忽略EP等待/新成本；不减observer重建生产基线。新返回段分析没有新增NPU运行。[可复算脚本](../../tools/analyze_moe_return_frontier.py)。

下一条独立前沿：SMLA内真正的选中cache读取、QK/PV/归约与C1/C2/SWA行域。MoE的GMM内部、量化共享和所有合法shape仍有具名缺口，不按本表有记录宣称整个MoE或全Goal穷尽。
