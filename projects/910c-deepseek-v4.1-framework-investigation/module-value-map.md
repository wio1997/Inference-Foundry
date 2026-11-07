# 按模型执行顺序筛选融合算子

在 171 的固定 DeepSeek V4.1 Flash W8A8、DSpark、Engram、CANN 9.2 和 910C 环境中，先执行输入准备，随后经过 40 个目标层。在层 1、14 插入 Engram。每层依次执行 Attention 和 FFN/MoE，再进入末层读出、校验/采样、草稿提案及下一次迭代。表中收益按最早涉及的模块归属一次。TTFT 表示首 token 延迟。表内历史八请求的 TTFT P90 为 1491.85 ms，1% 对应 14.92 ms。不同服务对照使用各自的前后参照。

| 执行位置 | profile 中值钱的工作 | 对应算子、适用分支与取舍 |
|---|---|---|
| [M00 输入与元数据](modules/M00.md) | 单独输入非通信设备任务在已测 decode 周期内最高占 0.127%。同一批 26 个 builder 重复准备 CP 局部长度和 CPU 输入计划，提交与同步也耗时。 | **保留批内公共生产者。** 已有有效长度时跳过重复的后缀处理、减法、截断和空行选择，保留 query offset 和未命中分支。同轮 CP 单改整批缩短 6.10% / 4.89%。在 CP 加 slot 基线上，同批复用 CPU 计划又缩短 10.36% / 9.56%，后者属主机端工作共享。显式批字典、实际输入内容、TP 范围、causal 和代际都要匹配。[同场比较](baselines/module-fusion-estimates.json) |
| [M01 条件 Engram](modules/M01.md) | 两个 Engram 层在已测 decode 周期的 WKV 合计中位 0.553 ms，gate 为 0.317 ms。随 hidden 变化的 gate 仍要执行。 | 单独合并 gate 与 lookup 即使免除两者也至多占 0.743% 本地周期。跨层缓存只在输入内容、位置、权重及历史相符时可共享；即使常量免费，中位仍需消除约 75.4% 的可缓存部分才能省 1% 本地周期。自然命中率未知，保留为条件项。 |
| [M02 Attention 前置 mHC](modules/M02.md) | 2,560 次 prefill HcPre 中，2,176 次直连上层 FFNPost。其余有首层初始化、Engram 改写或 DSpark 均值读者。 | 普通层可让 FFNPost 交付残差和下一层 Y，将系数留给 Attention Post。即使把普通层原 HcPre 全部移走，历史 P90 可省 24.44 ms，需兑现 61% 才达 1%。旧实现前后参照结果不一致。仅重叠 Q 投影与 TP8 AllGather 的条件上限为 9.01 ms，单独降级。[跨层预算](baselines/attention-service-budget.json) |
| [M03 Attention](modules/M03.md) | QLI 在 prefill / decode 已测本地窗口的中位占比约 4.84% / 4.44%。重复生成的候选块表示可由四层共用。 | **长上下文保留四层 QLI 共享转换。** 各层保留自己的打分和选 token，原基线整批实测缩短 1.50% / 1.48%。在已有批量 slot 的基线上，增量尚未稳定。普通对话和短代码在已测命中次数下约省 4.98 / 1.01 ms。只消除索引交给 SMLA 的小转换优先级低。[长短负载与来源](fusion-candidates.md)  跨层保留压缩 KV 的单次读流量按连续拷贝参考约省 4.02–5.77 ms 首 token P90；稀疏速率低于约 465 GB/s 时可能改变优先级。[KV 页估算](baselines/sparse-kv-retention.json)  两路 Q 共用量化输入在 C8 首 token P90 约省 0.43–0.58 ms，独立降级。[同一输入的读者](baselines/activation-quant.json) |
| [M04 Attention 输出及 Post](modules/M04.md) | OA → OB → ReduceScatter → HcPost 的 2,240 条链中有 1,600 次 OB 输出拷贝。只消除拷贝和接口间隙，各请求首 token 预算为 0.23%–0.39%。 | 投影直接写通信所需布局是低优先级独立项。OA 与 OB 权重直接预乘会增加 MAC 并改变中间 BF16 舍入。优先保留下行跨 M05 的残差共用。[投影与下一读者](baselines/projection-joint-arrival.json)  直接从 Attention 写 TP8 发送布局，计入 25% 原转置成本后，C8 整批约省 0.49%–0.51%，独立降级。[发送端](baselines/attention-send.json) |
| [M05 FFN 前置 mHC 与归一化](modules/M05.md) | 2,560 条逐卡链上，HcPre 两阶段各读一次四路 BF16 残差。内部还将每行 81,920 B 的 FP32 工作区写回并供 Cube 读取。 | **残差与 Y 接口融合单独降级。** 若在片上交付两次残差读取和 Y 的写读，每行省 102,400 B，按已有同尺寸搬运的两种计价，C8 首 token P90 约省 4.92–9.83 ms（0.33%–0.66%）。路由同时消费宽输出时约 6.88–13.76 ms。连内部工作区也能省掉才有 12.78–25.56 ms 的情景，等价搬运路径尚无证据。须保留残差、三份系数、两种归一化输出和舍入。[张量及请求估算](baselines/hc-tensor-traffic.json)  让归一化再产出共享专家 INT8 输入，移动工作后的估算为慢 2.01 至快 3.09 ms。[三种输出](baselines/activation-quant.json)  让上游输出两份残差仅净省一份 BF16 读。只保存扩宽平面时，Y 与 FFNPost 两个后续读者抵消节省。[工作区与系数精度](baselines/hc-workspace-delivery.json) |
| [M06 Router 与专家](modules/M06.md) | DispatchFFNCombine 占已测 prefill 本地时间中位 44.65%，其中还有必要的专家计算和通信等待。共享专家的两次矩阵乘之间确实物化了 INT32 中间结果。 | Gate/Up → DequantSwigluQuant → Down 两处一起省去写读，C8 首 token P90 估算省 0.31–3.97 ms；按激活完全隐藏且两侧间隔可省的有利情景，约 8.83 ms。两处联合降级。归一化共用见 M05，加法并入 HcPost 见 M07。[逐形状与请求估算](baselines/shared-mlp-fusion.json) |
| [M07 FFN 后置与跨层状态](modules/M07.md) | 4,480 次 Post 已绑定。独立直连 Post 边界即使全省，最高占本地周期 0.538%。普通跨层、Engram 与辅助状态分支的读者不同。 | ForeachAddList 合入 HcPost 时需先保留 BF16 加法舍入。旧 M3 实现的整批时间比前后参照多 0.31% / 0.19%，这一实现降级。跨到下一层 HcPre 统一在 M02 计价；末层按行裁剪统一在 M08 计价。 |
| [M08 末层读出](modules/M08.md) | 四批实际有 26,318 个有效专家发送行，当前目标词表投影只选 62 行。其它 rank 的专家服务、raw 四路状态和 prompt 分数可能仍需这些行。 | 按行裁剪属于历史联合改写，下面的整段金额包含工作量变化。 在保留原前处理与共享计算时，净消除 25% / 50% 的所计时间，P90 约省 5.24 / 10.48 ms，整段免费为 20.95 ms。可归给算子融合的边界为 Cast → Mul → ReduceSum → Cast → RMSNorm，省一半仅 2.78–3.89 ms。出现过中间值差异，需保留系数及归约顺序。[逐请求成本](baselines/profile-fusion-estimates.json)  六个普通请求中，首轮仅裁剪末层归约与 RMS，估算每请求约省 0.26–0.61 ms。单请求校验时六个连续行在八卡原分片中仍需 24 个等长通信槽；路由、raw 状态及 prompt logprobs 分支单列。[普通请求与分片](baselines/ordinary-terminal-rows.json) |
| [M09 校验、采样与发布](modules/M09.md) | 目标 greedy 路径中，seed、计数与 padded 准备整段全省，最高占本地周期 0.083%。 | 将比较、计数和下步掩码直接交给 padded 输入只作附带机会。固定前序迭代，仅把当前批结果移到草稿三层前发布，历史 P90 最多约省 6.40 ms。随机/混合采样仍保留 RNG 和 raw logits 读者，当前数据不能给这些分支定价。[请求发布约束](baselines/terminal-sparsity.json) |
| [M10 DSpark 草稿](modules/M10.md) | decode 草稿整段中位 4.00 ms，约占本地周期 7.31%。这包括必须保留的专家、投影和注意力。 | 三路辅助状态 Index → ConcatD → 草稿输入投影可少写中间张量。64 个逐卡窗口仅 40 个有三次取行，16 个仅拼接，8 个两者均无。净省一半，P90 约改善 0.54–0.88 ms。三层专家保留时，单独免除已识别计算也仅约 6.66 ms。该打包链降级。[估算](baselines/profile-fusion-estimates.json) · [跨卡边界](baselines/draft-arrival-budget.json) |
| [M11 下一迭代交付](modules/M11.md) | 相邻迭代的长间隙含重叠 CPU、设备工作和等待。实际反复执行的 slot 坐标与扁平映射写入能直接定位。 | **先把同一输入的二维坐标和一维 slot 合成双输出，再把符合条件的 26 项写入批量提交。** 在同轮 CP 单项基线上，双输出增加 5.43% 整批耗时降幅。更后面的 CP＋双输出 slot＋CPU 计划基线上，26→1 增量为 3.39% / 2.22%。大形状、压缩比、无效行和未匹配的图缓冲区须单列。[双输出](baselines/module-fusion-estimates.json) · [批量调用](modules/M11.md) |

## 这轮调整了什么

前期把高耗时模块中的各种优化都当成融合线索，导致继续研究专家放在哪张卡、按冷热如何复制权重。此前也对单次小转换过早读源码和做变体。现在先登记实际算子边界和合并可消掉的工作，再用 12 个模块已有 profile 按请求计价，1% 作为本轮深查参考。

1. 用户等的是哪段时间？把同一机制在该请求走过的分支、层和迭代上汇总。并行的卡只计关键到达，分别报告首 token、整批或单请求完成时间。
2. 改写具体移除了哪次计算、搬运或等待？从生产者开始一次列清共享读者、原始输出和舍入要求。让两个分支在各自内部融合、给不同读者保留多个输出、只融合公共段，都在这一步比较。分支记录缺失时保留条件。
3. 全部可变工作都省掉能给请求省多少时间？若已低于该负载的筛查门槛，整组退出；若分支或跨层组合还能移除额外工作，重新按整个组合定价。对仍有价值的项用实测同形状时间或搬运速率估算，注明假设。只有会改变排序的依赖缺口才值得查代码。

例如，末层逻辑行从 24 减至 6，能省多少通信取决于分片。只要原八卡等长包的最长分片仍为 3，每次就还要传 24 个槽。反过来，单个 slot 写入很小，但同一批可把 26 次提交合为一次，应当按整批定价。[普通请求分片](baselines/ordinary-terminal-rows.json) · [slot 联合提交](modules/M11.md)

## 取舍与缺口

slot 双输出、26 项批写有同轮实测。四层 QLI 共享转换须注明已有批写基线上的增量不稳定。末层的行裁剪、专家负载均衡和 CPU 计划属于历史比较；其中的组合收益仅在拆出相同调度下的融合增量后计入候选。

12 个执行模块已有首轮价值判断。M06 共享专家两处中间写读已按 C8 与六个普通请求估算。下一项为 M07 跨层 Post → Pre 融合，引用既有直连与旁路证据。[完整覆盖台账](coverage.json)
