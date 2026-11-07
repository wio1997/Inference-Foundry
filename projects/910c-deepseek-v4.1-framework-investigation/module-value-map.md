# 按模型执行顺序筛选融合算子

在 171 的固定 DeepSeek V4.1 Flash W8A8、DSpark、Engram、CANN 9.2 和 910C 环境中，先执行输入准备，随后经过 40 个目标层。在层 1、14 插入 Engram。每层依次执行 Attention 和 FFN/MoE，再进入末层读出、校验/采样、草稿提案及下一次迭代。表中收益按最早涉及的模块归属一次。TTFT 表示首 token 延迟。表内历史八请求的 TTFT P90 为 1491.85 ms，1% 对应 14.92 ms。不同服务对照使用各自的前后参照。

| 执行位置 | profile 中值钱的工作 | 对应算子、适用分支与取舍 |
|---|---|---|
| [M00 输入与元数据](modules/M00.md) | 单独输入非通信设备任务在已测 decode 周期内最高占 0.127%。同一批 26 个 builder 重复准备 CP 局部长度和 CPU 输入计划，提交与同步也耗时。 | **保留批内公共生产者。** 已有有效长度时跳过重复的后缀处理、减法、截断和空行选择，保留 query offset 和未命中分支。同轮 CP 单改整批缩短 6.10% / 4.89%。在 CP 加 slot 基线上，同批复用 CPU 计划又缩短 10.36% / 9.56%，后者属主机端工作共享。显式批字典、实际输入内容、TP 范围、causal 和代际都要匹配。[同场比较](baselines/module-fusion-estimates.json) |
| [M01 条件 Engram](modules/M01.md) | 两个 Engram 层在已测 decode 周期的 WKV 合计中位 0.553 ms，gate 为 0.317 ms。随 hidden 变化的 gate 仍要执行。 | 单独合并 gate 与 lookup 即使免除两者也至多占 0.743% 本地周期。跨层缓存只在输入内容、位置、权重及历史相符时可共享；即使常量免费，中位仍需消除约 75.4% 的可缓存部分才能省 1% 本地周期。自然命中率未知，保留为条件项。 |
| [M02 Attention 前置 mHC](modules/M02.md) | 2,560 次 prefill HcPre 中，2,176 次直连上层 FFNPost。其余有首层初始化、Engram 改写或 DSpark 均值读者。 | 普通层可让 FFNPost 交付残差和下一层 Y，将系数留给 Attention Post。即使把普通层原 HcPre 全部移走，历史 P90 可省 24.44 ms，需兑现 61% 才达 1%。旧实现前后参照结果不一致。仅重叠 Q 投影与 TP8 AllGather 的条件上限为 9.01 ms，单独降级。[跨层预算](baselines/attention-service-budget.json) |
| [M03 Attention](modules/M03.md) | QLI 在 prefill / decode 已测本地窗口的中位占比约 4.84% / 4.44%。重复生成的候选块表示可由四层共用。 | **长上下文保留四层 QLI 共享转换。** 各层保留自己的打分和选 token，原基线整批实测缩短 1.50% / 1.48%。在已有批量 slot 的基线上，增量尚未稳定。普通对话和短代码在已测命中次数下约省 4.98 / 1.01 ms。只消除索引交给 SMLA 的小转换优先级低。[长短负载与来源](fusion-candidates.md) |
| [M04 Attention 输出及 Post](modules/M04.md) | OA → OB → ReduceScatter → HcPost 的 2,240 条链中有 1,600 次 OB 输出拷贝。只消除拷贝和接口间隙，各请求首 token 预算为 0.23%–0.39%。 | 投影直接写通信所需布局是低优先级独立项。OA 与 OB 权重直接预乘会增加 MAC 并改变中间 BF16 舍入。优先保留下行跨 M05 的残差共用。[投影与下一读者](baselines/projection-joint-arrival.json) |
| [M05 FFN 前置 mHC 与归一化](modules/M05.md) | 四批、2,560 条逐卡链均出现 Attention HcPost → FFN HcPre → RMSNormCast。单独 norm 全省只值 5.39 ms，路由填充和清零只值 0.21 ms。 | **候选为三段联合生成原残差与归一化输出，保留 FFNPost 后读的系数及 BF16 舍入。** 若净消除三段设备时间的 25% / 50%，逐层按两组 DP 到达重算，历史 P90 最多约省 12.12 / 24.24 ms（0.81% / 1.62%）。仅少读一次残差按同字节拷贝估算约 3.93 ms。旧版延后系数没有稳定服务收益，更宽的融合需能消除额外实质工作。[三段估算](baselines/module-fusion-estimates.json) · [原排程](baselines/ffn-schedules.json) |
| [M06 Router 与专家](modules/M06.md) | DispatchFFNCombine 在 prefill 本地时间中位占 44.65%，decode 为 25.10%，其中包含通信等待。真正可变部分包括六路返回位置、归约和共享专家合并。 | 对齐 120 层的原生时差、逐卡返回字节与真实首读者，在每接收端有效 25 GB/s、仅能重叠四段共享计算时，历史 P90 估算缩短 4.71–6.97 ms（0.32%–0.47%）。这版重新放置归约降级。末层有按行减少专家与返回工作的不同机会，见 M08。普通六批 eager 行数均未匹配该定价。[返回模型及小形状](baselines/ordered-return-endpoints.json) |
| [M07 FFN 后置与跨层状态](modules/M07.md) | 4,480 次 Post 已绑定。独立直连 Post 边界即使全省，最高占本地周期 0.538%。普通跨层、Engram 与辅助状态分支的读者不同。 | ForeachAddList 合入 HcPost 时需先保留 BF16 加法舍入。旧 M3 实现的整批时间比前后参照多 0.31% / 0.19%，这一实现降级。跨到下一层 HcPre 统一在 M02 计价；末层按行裁剪统一在 M08 计价。 |
| [M08 末层读出](modules/M08.md) | 四批实际有 26,318 个有效专家发送行，当前目标词表投影只选 62 行。其它 rank 的专家服务、raw 四路状态和 prompt 分数可能仍需这些行。 | **有条件保留末层专家返回 → 加共享结果 → mHC Post → 按行词表读出。** 在保留原前处理与共享计算时，净消除 25% / 50% 的所计时间，P90 约省 5.24 / 10.48 ms，整段免费为 20.95 ms。只融合 Cast → Mul → ReduceSum → Cast → RMSNorm，省一半仅 2.78–3.89 ms。出现过中间值差异，需保留系数及归约顺序。[逐请求成本](baselines/profile-fusion-estimates.json) |
| [M09 校验、采样与发布](modules/M09.md) | 目标 greedy 路径中，seed、计数与 padded 准备整段全省，最高占本地周期 0.083%。 | 将比较、计数和下步掩码直接交给 padded 输入只作附带机会。固定前序迭代，仅把当前批结果移到草稿三层前发布，历史 P90 最多约省 6.40 ms。随机/混合采样仍保留 RNG 和 raw logits 读者，当前数据不能给这些分支定价。[请求发布约束](baselines/terminal-sparsity.json) |
| [M10 DSpark 草稿](modules/M10.md) | decode 草稿整段中位 4.00 ms，约占本地周期 7.31%。这包括必须保留的专家、投影和注意力。 | 三路辅助状态 Index → ConcatD → 草稿输入投影可少写中间张量。64 个逐卡窗口仅 40 个有三次取行，16 个仅拼接，8 个两者均无。净省一半，P90 约改善 0.54–0.88 ms。三层专家保留时，单独免除已识别计算也仅约 6.66 ms。该打包链降级。[估算](baselines/profile-fusion-estimates.json) · [跨卡边界](baselines/draft-arrival-budget.json) |
| [M11 下一迭代交付](modules/M11.md) | 相邻迭代的长间隙含重叠 CPU、设备工作和等待。实际反复执行的 slot 坐标与扁平映射写入能直接定位。 | **先把同一输入的二维坐标和一维 slot 合成双输出，再把符合条件的 26 项写入批量提交。** 在同轮 CP 单项基线上，双输出增加 5.43% 整批耗时降幅。更后面的 CP＋双输出 slot＋CPU 计划基线上，26→1 增量为 3.39% / 2.22%。大形状、压缩比、无效行和未匹配的图缓冲区须单列。[双输出](baselines/module-fusion-estimates.json) · [批量调用](modules/M11.md) |

## 取舍与缺口

已能据实排优先级的设备融合有 slot 双输出、26 项批写和四层 QLI 共享转换。M04–M05 的三段交付及 M06–M08 的末层按需读出保留为结构候选。它们各自要兑现的可省比例、已有负面对照和数值条件见表；两者预算不能与各自包含的小项叠加。CPU 计划共享单列为主机端工作，避免把主机耗时降幅算给 NPU 算子。

一次相同形状的命中只说明该分支可能适用。普通请求的工程结论沿用实际 eager 行数、图回放和读者计数；没有匹配定价的分支保留条件。表中 12 个模块均已首筛。差额较大的未决问题包括 M04–M05 除残差单次读外还能实际移除多少工作、普通负载末层究竟有多少行可安全省掉，以及改变 tile 就绪顺序后的 MoE 返回预算。[完整覆盖台账](coverage.json)
