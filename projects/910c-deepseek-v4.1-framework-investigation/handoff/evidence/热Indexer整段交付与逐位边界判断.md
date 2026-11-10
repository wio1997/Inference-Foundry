# 热 Indexer 整段交付与逐位边界判断

## 介绍

继续启动 helper 触及的真实热调用，融合范围先取 query 量化 → QLI scoring/membership → top512 → chronological indices/lengths → 多层 SMLA。判断整段能少算、少搬什么，保留原 aux-stream 重叠、原算术舍入、候选与状态代次，再拆局部。每个替换原语必须逐位一致。

## 到了哪里

原 C8 全16 rank 的既有 first-token native 表按实际 HC layer 窗匹配，共368组 quant→QLI→prepare。三段输入/输出几何与50个独立签名成本条件逐项相符；数据源 SHA 在 `indexer-delivery-public-summary.json`。当前运行树 Python caller 以及6个 C++/arch22 native 源已 pin；只审阅 `indexer-native-source-review.json` 所列范围，不当 native 全部实现、host tiler 或当前加载二进制已核完。

实际 index sources 为 2/8/14/20/24/28/32/36。四个全局 carrier 416/816/817/818 不等于每卡行量：记录中为3、413/419、416、814/818、815/817、816。所有368组 topk512 行数≤1024，sort输入还带 rows×1 的 lengths，已走融合长度。全局6540行不能被误用来推断这里存在独立求和。无 local query 的 peer 仍可能做其它真实模型/EP工作，不能整卡跳过。

同旧 cohort 的 per-layer DP 到达关系给有利 envelope；quant 在 aux stream、QLI/sort 在 main，先将三者 duration 全相加，故仍偏乐观，不能当实测暴露或因果收益。冻结 N8 collector 的 P90 是最大值。

| 范围/假设 | 条件 P90 省时 | 条件 P90 比例 | 最大单请求比例 |
|---|---:|---:|---:|
| QLI＋量化/排序全免费 | 89.693940ms | 6.012243% | 6.137633% |
| 上述删50%，每个 index source 新付5us | 44.686970ms | 2.995397% | 3.057868% |
| 仅量化＋排序全免费 | 8.870134ms | .594571% | .606971% |
| 仅量化＋排序删50%，每source新付5us | 4.275067ms | .286561% | .292537% |

这不是“6%融合收益”或完整请求/整批前后对；未计已知所有新资源成本的表是筛查情景，不与其它候选相加。六 ordinary 和整批 decode 不由 first-token表外推。此前 byte membership overlay/批slot组合是另一个历史基线，输出文本相同也不能升级成当前要求的每原语逐位准入。

值得深入的整段已继续追入 native：Python V3 在 `torch_binding.cpp:3160–3172` 绑定 `qli_v2::QuantLightningIndexerCandidate`，adapter再调用 `aclnnVllmAscendQuantLightningIndexerV2`，所以 saved task 标 V2 不等于走错 Python V3。adapter mode1 输出2048候选块，mode2要求 candidate 输入，mode3普通；原key/scale stride0显式传入。arch22入口、workspace两级score交付与 crosscore flags 已读局部，source identity 仍不等于服务加载库/精确tiling key身份。

发现的数值与融合限制：

- query quantizer 把 absmax/127 的 scale 先舍入 FP16，再转 FP32、clamp、div_rn、nearbyint。native Vec0还把 weights×qscale 以 half 相乘，Cube的 INT8 QK 经 Fixp DEQF16＋ReLU 到 half，再用 half 权重做 FP32 head reduction，最后乘 key scale。取消scale、重排乘法、从INT32直接算score都会跨过原来的舍入界。
- mode2 的 membership 在完整 QK/head-score之后才应用。BuildCandidateMask 会对同一候选行先 sort、逐tile做 CountGE/距离比较/展开；这是候选 producer→多个 native consumer 可共享表示的具体位置，不是只有热 sort helper。
- 原 mode2 用 `score + (NEG_HUGE-score)*isOut`，仍保留候选外真实位置号作为不足top512时的有效填充。把候选外全部删成 -1 会改变原语结果；即便 isOut=0，非有限值的乘零也不能当通用恒等式。source 注释不是数学/设备逐位证明。
- 原 chronological top512 bank 已在八个 index source 各产一次，被后续层共享。把排序融合进每个 SMLA 可能重复工作。末输出有常规路径和 LD归约路径；只改其中一个 CopyOut 不能覆盖所有合法分支。
- 原 pipeline 每个基本块有 Vec0→Cube→Vec1 的双buffer flag握手，ProcessDecode还有 SyncAll/LD。不能把源码中的等待全当可删空隙，也不能拿独立小kernel的代价替换整段新方案成本。

## 结果

**已停止**：同独立1% first-token预算下，只融合 query 量化、chronological sort/lengths，以及“消除6540行额外sum”的配方。前者连全免费 envelope 都低于1%；后者在已匹配368调用中本来就不存在。保留这些组件参与较大联合方案。

**已停止**：无条件取消half scale/DEQF16中间舍入、候选外变-1、按集合相同删chronological排序，以及以每层重算替代既有group共享的直接配方。不是新NPU失败记录，而是源码语义/原语验收条件不满足。

**待确认，值得继续深入**：整个 QLI native scoring/membership/表示交付与 SMLA 联合范围。原 membership 目前在算后mask，producer representation可被多个消费者共用；保留原数值链的key聚集/重排可能减少算术与搬运，但必须处理原候选外 filler、tie/非有限值、原key顺序、常规/LD两种末输出、persistent bank代次及新增资源成本。已有全段价格足以保留此范围，不能用小helper的.595%剪掉它。

当前具体待扫：host tiler/分核与 SplitCore；ProcessCandBlockTopk 的剩余 pin/tie/merge；ProcessLD/完整最终索引输出；PA key/scale收集的真实物理顺序；当前优先库/加载库与key身份。随后按实际同域候选包核定可省算术比例与预处理/共享成本，只有会改变去留的参数才补证。普通负载和跨迭代/跨请求cache资格保留独立缺口。Loop030继续，未接受替换原语、服务性能或基线；全链路仍未完成。
