# 末层 attention 选行与 FFN 逐位门槛

## 介绍

以冻结 910C、CANN 9.2.0、TP8/DP2/EP16/PP1、原始 C8 和每个替换原语逐位一致为范围，沿最后一层实际 Q、缓存、跨卡和 FFN 消费划出可删工作。原生 FFN 失败件按 SHA 复用，价格按原 C8 的真实调用重新分段。

## 到了哪里

| 交付 | 实际路径和选行约束 |
| --- | --- |
| 末层输入及其他读者 | DSpark 取零基 37/38/39 层入口 aux，38 层仍保留全行。39 层无 Engram，读 long-KV20 和 topK36，写自己完整 SWA。冻结请求已排除 prompt-logprobs 读者。真正 MTP、pooling、PP 中间态、dump 或其他完整输出读者须走原分支。 |
| 原行需求 | NPU runner 先定真实 InputBatch/QSL。无草稿每请求取 QSL 末行，有草稿按 `num_draft_tokens+1` 取有序后缀。保留尚未验证的全部候选，含原实现会采样后丢弃的 chunk 情况。7 个实际 DP 批次中，5 个有长 prefill 可裁，两个只做验证的 6/12 行全活。八卡原 CP query 数与原总数一致。 |
| 原 Q 与物理缓存分流 | CP `_prepare_inputs_and_caches` 交局部 query 和全局 cache 视图，空本地 Q 仍走 writer。保留原 HcPre、输入 RMS、SP all-gather、全量 Wkv/RMS/RoPE/scatter 及第 39 层物理页发布。当前和以后迭代都能读这些页。原 Q_a、RMS、量化、Q_b 与 RoPE 可另审选行，缩 M 可能改 W8A8 tiling 与舍入。 |
| C1 SMLA 的原坐标 | 每个请求与原 CP 切片的交集内，所需行是连续后缀。沿原 device 长度 `L` 保持 `L−q+old_s1 = L−k+new_s1`，以旧 CP-local 行取 TopK36，保留 block table、causal 位置和压缩面长度。默认 G64 的 C1 每 M 块一 query，归档 arch22 的原 scalar loop/CalcParams 在 6 组 CPU 控制中保住 56 个 retained tile 的窗口、稀疏有效长度、首末 S2 和 M/V 几何。中间行、错用 topK 新前缀、改变 S2 切分均有反例。AICPU 分核、真实 flags、MMA、softmax 和写回仍需设备逐位对拍。 |
| CP 返回与 O/FFN | `restore_tp_heads` 先按目的 TP head shard 排列，再按源 CP rank 接收。60 个与归档尺寸相容的请求顺序覆盖 56,000 个行/head 坐标。原 3328 行单请求的末行在 CP7 旧 local415。其余用实际 QSL/indices 取地址。固定 owner 的等长包在五个可裁批次的最坏容量为 8、8、56、152、104 行，原来是 3328、6528、6536、3352、6544 行。这里是逻辑槽数，各 rank 仍须参加。O 投影缩 M、SP reduce-scatter 的 BF16 归约顺序和 FFN residual 位置各需验证。 |
| 路由与两个专家分支 | FFN HcPre/RMSCast → FP32 gate/TopK → W8A8 DFC → 本地 routed finalize → 原 shared Quant/GateUp/ActQuant/Down → BF16 add → HCPost。PP1/SP 的 MC2 finalize 在选中分支按原 local 行数裁片。原 dense 系数影子保住所测 routed 值，full expert 和 HCPost 仍各有 1 元素位差，最终读出相同。对照把首次观测差异缩到 routed 交付之后，原影子缺分开的 shared Down 和 BF16 add 输入。 |

[56 个 rank 窗口、原始通信、60 个请求顺序和端点复算](last-layer-attention-price.json) · [原 CSA 标量控制与反例](last-layer-smla-controls.json)。

## 结果

- **已停止 · 独立第 39 层 attention 选行。** 原 C8 四波按固定原时序宽松串行累计，SMLA 全免的 P90 TTFT 信用仅 **2.595 ms（0.174%）**。Q/O、原生算子和返回 A2A/RS 全免为 **9.306 ms（0.624%）**。保留六个原 cache 前置/写入原语及输入 AG，连其余等待和空隙都全免为 **8.638 ms（0.579%）**。两者采用不同的时钟和重叠约束，各自低于原独立 1% 排队参考，停止同预算内的 Q 选行、SMLA 排程与紧凑 A2A 小变体。
- **待确认 · 保留完整 cache/输入 AG 的 attention→FFN→head input。** 同 C8 联合全免包络 **31.097 ms（2.084%）**，每波新增 0.5 ms 时，至少要真正去掉并暴露包络的 **54.4%** 才到 1%。选行提前到 attention 会改变死行的后续 router 输入，原专家分组和计数随之失去来源。保持旧 M/tile、完整 EP16 参与及被保留值的算术是实际开发条件。
- **待确认 · 先做更可隔离的 FFN 逐位鉴别。** 用原 dense HcPre、归一化和路由权重，只对比选中 routed 输出，再逐一比 shared Down、原 BF16 add 及 HCPost。若 shared 变形导致位差，保留原 dense shared。若 add 的数值几何也需原样，保留原 add。原 EP16 join 模型下这两项保留情景的全免 P90 分别 **20.954/20.822 ms**，距 1% 的全部新增工作上限约 **1.509/1.476 ms/波**。需真实 rank/路由/布局及各原语位比较。若整个原 expert join 都得保留，原固定边界的全免仅 **14.853 ms**，停止同预算的剩余末端变体。
- **已确认可实现 · 本轮没有新增替换准入。** 六普通负载没有同边界的原生逐位和端点数据。原 C8 有利包络与 EP16 固定 join 模型不能相加。

下一段走最后一层 FFN 的原 EP16 路由、同输入 shared Down、add 与 HCPost 的实际完成边界，先用归档找出能隔离首个位差的最小对照。只在结果会改变联合方案去留时补原生证据。[结构化结果](last-layer-row-review.json)。
