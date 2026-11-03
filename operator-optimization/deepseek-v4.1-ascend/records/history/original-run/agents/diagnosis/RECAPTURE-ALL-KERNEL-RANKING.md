# 新采集全算子：严格可比组的范围内排序

本机流式读取固定 SHA 的已提取 JSONL，没有重读远端大 trace 或运行设备。footer 验证 16/16 rank 完成、无 rank error；16 条 rank 记录各自仅出现一次，rank 行 SHA 与 CSV/metadata 输入 SHA 已保存，并逐 rank 核对 trace SHA 对应既存 scope 摘要。CSV/metadata 的 SHA 是提取时观察值，不能反推采集前文件已被冻结。1,280/1,280 个目标 layer scope 与 sidecar 的 EP、PID、call_id、层号和 digest 匹配；留存唯一匹配设备任务 53,918 条。

提取记录中有 96,922 个目标 scope 内 flow start 事件、39,160 个未匹配 CSV、3,840 个设备事件缺名称或任务身份、2 个不唯一 start 被排除。这些未匹配不代表没有工作。分析依据原始 operator name、CSV Type、物理输入/输出 shape、dtype、format 与 core 类型组成的精确签名，要求同一 digest、同一层的全部 8 EP 均有该签名且每 EP 任务数相等；不把不同原名、不同形状的算子合并。即便物理签名相同，有效 token、mask、专家路由行数等实际工作量仍可能不同。

**2026-10-30 选定跨流区间补充：**复算 [16rank-selected-device-intervals.json](16rank-selected-device-intervals.json) 的 32 个单卡单摘要记录，各 40 层 MC2 区间并集与其他已选任务区间并集的交集除以该组 MC2 并集，组间最小/中位/最大为 0.1944% / 71.6603% / 91.8148%。`recheck_16rank_selected_device_intervals.py` 复算原始并集；`recheck_mc2_overlap_streams.py` 固定输入和汇总 SHA，再检查每组 40 个 MC2 同 stream 与其他选中任务的半开区间相交情况：同 stream 交叠对数为 0，其他 stream 交叠对数 1529；这些 pair 可互相重叠，不能视作 1529 次独立事件。结果仅覆盖同设备已选、经 CSV 关联的记录；未匹配通信与同步、跨轨时钟、组间工作量、实际数据依赖和请求关键路径仍未知。交集不是已被隐藏的通信时间或可实现收益。

| 分组覆盖 | 数量 |
|---|---:|
| 物理签名组总数 | 6,862 |
| 物理字段完整、8 EP 均有且每卡任务数相同 | 4,930 |
| 缺少一张或多张卡的相同签名 | 1,412 |
| 物理 shape/dtype/format 字段不完整 | 520 |
| 已有 8 EP 但任务数不相等 | 0 |

在 4,930 个物理完整组内，AIC 周期均正且 ratio 有效的有 **1,596** 组；3,334 组因为至少一个周期为零或字段无效而排除，其中涉及零周期的有 3,294 组、字段缺失/无效的有 40 组。AIV 分别有 **3,966** 个合格组、964 个指标不合格组；其中涉及零周期 924、字段缺失/无效 40。这些原因可能重叠，不能当成互斥分类相加。AIC 和 AIV 分别检查和排名，不将不同核及不同分母的值混排。

## AIC 的组内最低候选

在同一 digest 的合格组内，按 `sum(aic_cycles × 显示的 aic_mac_ratio)/sum(aic_cycles)` 对各**原始算子名及 CSV Type**取最低组，列出前三类。这里的层号是该类型最低组对应层；并非整个模型的最低值。

| digest | 第一 | 第二 | 第三 |
|---|---|---|---|
| 1b826883 | DispatchFFNCombine，层 14，0.4891% | HcPre，层 1，28.5715% | aclnnMatmul_MatMulCommon_MatMulV2，层 14，35.5651% |
| 7ddf7ae9 | DispatchFFNCombine，层 14，1.6268% | aclnnMatmul_MatMulCommon_MatMulV2，层 20，7.5370% | VllmAscendQuantLightningIndexerV2，层 32，12.9887% |
| 8c18c293 | DispatchFFNCombine，层 0，0.1372% | aclnnMatmul_MatMulCommon_MatMulV2，层 14，6.8381% | VllmAscendQuantLightningIndexerV2，层 24，12.9767% |
| db852bff | DispatchFFNCombine，层 15，2.7863% | HcPre，层 33，28.8379% | aclnnMatmul_MatMulCommon_MatMulV2，层 14，36.2380% |

更长的原始名字例如 `aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3` 保留原文，不能靠“含 MatMul”将不同算子族合并。完整前十及物理签名见 JSON。

## AIV 单独观察

四个 digest 的 AIV 合格组中，`aclnnInplaceCopy_TensorMoveAiCore_TensorMove` 的最低显示比例均为 0.0%。这只是导出显示值，不证明实际零活动；比例导出精度、舍入及分母的具体版本语义仍待确认。以 digest 1b826883 为例，接下来的最低候选为 `aclnnInplaceCopy_TransposeAiCore_Transpose` 显示 0.1000%，`DispatchFFNCombine` 层 14 为 0.1250%。其余各 digest 前十单独存入 JSON；不与 AIC 数字排在同一榜单。

## MC2 与 SparseFlashMla：同 digest、同层配对

比较只接受两种算子**各自**都满足其 8 EP 同签名/同任务数及 AIC 有效的组，且两者来自同一 digest、同一层和同一 EP 集合；不要求不同算子互相具有相同形状。下表中位数是每层周期加权显示比例的中位数，不是时间、吞吐量或全卡利用率。

| digest | 配对层数 | MC2 中位数 | SparseFlashMla 中位数 |
|---|---:|---:|---:|
| 7ddf7ae9 | 40 | 3.6764% | 32.4730% |
| 8c18c293 | 40 | 2.5799% | 32.6490% |
| 1b826883 | 0 | unknown | unknown |
| db852bff | 0 | unknown | unknown |

后两组的 80 个层配对因没有合格的 SparseFlashMla 八卡组而排除，不可填零。例如 1b826883 的层 14，EP8–14 的 MLA 输入首维为 1023，而 EP15 是 1019；db852bff 的对应层 EP0–6 为 1023、EP7 为 1019。MC2 在同组八卡形状一致，并不消除另一个算子的形状差异。两个合格 digest 也不能据此断定两类算子工作量相同、说明根因或估算可加速幅度。

## 复算及边界

[analyze_recapture_all_kernel_groups.py](analyze_recapture_all_kernel_groups.py) SHA-256 `7c309d0cd20889a9e234447e53e01d9a99473261af7f2809ccdad5c4bbf3f506`。默认输出有界 [all-kernel-groups-summary.json](../../diagnostic-20261029/all-kernel-groups-summary.json) SHA-256 `fa2065112662601df9c73333ea532a2bbcac9ed52b2308933ecce1d268cf417e`；追加 `--all-groups` 可输出全部 AIC、AIV 合格组以复核。输入 JSONL SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`。这里的比例为采样导出的任务字段算术，不是芯片利用率、用户线上时延或理论上界。


采集覆盖补充：`csv_missing` 的 39160 条选中 flow 在本提取器的规则下均映射到 trace 中的同步命名 X（22783 EVENT RECORD、16377 EVENT WAIT），不是未匹配的 AI 算子；该计数未证明 X 的反向唯一性或所有事件覆盖。EP8 原始 trace/CSV 定点证据与复算见 `CSV-MISSING-SYNC-COVERAGE.md`。其余未归属情况、缺指标和排序范围限制不变。

## 不同物理行数下的注意力时长描述

对上文因 1019/1023 行不同而不能进入严格同形排名的两个摘要，各取 40 层、每层 8 卡各唯一一条 `SparseFlashMla`。`db852bff…` 的 EP7 记录为 1019 行、EP0–6 为 1023 行；`1b826883…` 的 EP15 为 1019 行、EP8–14 为 1023 行。逐层核对完整输入/输出 shape 中的 1019 归一到 1023 后，名称、Type、shape、dtype、format、core、block 与 mix block 签名一致。只对每条记录计算自身 trace duration，以同层 1019 行卡的时长除以同层其余七卡的时长中位数，不对跨卡绝对时钟相减。

`db852bff…` 的 40 个比值最小/中位/最大为 **0.9844 / 0.9961 / 1.0185**；`1b826883…` 为 **0.9627 / 0.9974 / 1.0115**。按事后选定的 ±10% 描述阈值，两个摘要各 40/40 都在范围内。这排除了“这些选定记录的 1019 行卡相对于各自同层七卡中位数，时长偏差超过 10%”的描述。

这是同一次重 profiler 的层和卡，不是独立请求；物理行数、动态有效长度、输入内容和设备工作量均未控制。该比较不能证明少 4 行没有性能影响，也不能归因于硬件或算子，更不能恢复先前要求八卡严格同形的利用率排名。阈值不是预先设定的显著性标准或线上性能目标。

从运行根目录执行 `python3 agents/diagnosis/compare_attention_tail_rank.py`；脚本 SHA-256 `9e5cbbeafcfde5854a260f472c8185ff71b593f3cd6ee0d6fc15e3b7635815fc`，固定输入 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`。脚本逐 scope 检查唯一记录、归一化签名和正时长，仅读取本地数据。
