# F08 scatter 耗时与选中前序事件间隔：采集内分组

## 样本、方法、可复算性

对现有 `f08-actual-triples.json` 与 `f08-scatter-variation.json` 中全部 T=8180 的 640 组，按 EP、scope、flow ID 精确关联到固定 SHA 的 `target-layer-all-kernels-16rank.jsonl`。三任务的 stream、task ID、设备起始时间均与已选 records 一致；3 × 640 条事件的 trace 与 CSV duration 差不超过 0.001 微秒。用十进制减法计算每组所选 rotary 设备 X 结束到所选 scatter 设备 X 开始的间隔。数据与原始 trace 同源，选择和阈值是事后探索，不是 640 个独立请求，不代表生产统计或完整采集。未做新采集。

重算：`python3 agents/fusion/f08_gap_stratification.py`，脚本 SHA-256 `4e6c7990c45f131b09af8c37438488ab93a39caf78577aeeab6f93d161abdedf`；聚合 `f08-gap-stratification.json` SHA-256 `44c44d7685a55ff8fd2b6126be96623ceaf2d84a3b4c56d9990bb5119af15493`。报告全部对应三任务的原始选择、覆盖与不能证明的数据依赖，见 `F08-KV-NORM-ROPE-SWA-REVIEW.md` 和 `F08-SCATTER-VARIATION-AUDIT.md`。

## 观察

| 选中 scatter 持续时间 | 数量 | 所选 rotary→scatter 间隔范围（中位数），微秒 |
| --- | ---: | ---: |
| < 50 μs | 31 | 283.480–1834.480（1187.8165） |
| > 200 μs | 609 | 0.0195–0.04075（0.020） |
| 50–200 μs | 0 | — |

在该次导出和选定范围内，两个类别的间隔完全分离。仅看 EP3 同摘要 `db852bff...eefc8` 的 40 层：29 个短任务的间隔为 283.480–1834.480 μs（中位数 1244.580 μs）；11 个长任务为 0.020–0.040 μs（中位数 0.020 μs）。短任务为 L1–20、L22–29、L32；长任务为 L0、L21、L30–31、L33–39。

这种分离只出现在选定的 rotary→scatter 间隔。EP3 的短任务中，L22、L23、L32 的 norm→rotary 间隔仅 0.040、0.040、0.020 μs；因此不能泛称每个短样本在两个前序间隔都较长。

同一导出的 scatter `host_start_us` 减所选 rotary 的设备结束时间：EP3 短任务范围为 -1,449,586.125 至 -857,862.215 μs，长任务为 -1,402,647.505 至 -511,691.045 μs，范围重叠。这是两个导出时间戳的算术差；尚未证明时钟、graph replay 或 flow 对当前执行的具体关联，**不能作为主机下发延迟或因果解释**。

## 两个样本的同 scope 其他设备任务

在各自 36 条已选 layer-scope 任务里，L9 短样本的 16.680 μs scatter 与其余选中任务没有时间重合；L30 长样本的 265.060 μs scatter 与另一 track 47 的 `aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3` 任务重合 247.280 μs。这里的重合只按导出时间区间计算，不能证明它们同时争用相同硬件资源或解释时长差异，也不能排除 scope 之外及未被提取的任务，尤其不能把 L9 的 scope 内无重合说成设备空闲。

## 解读边界和下一步

上述关联提示应优先核对设备事件时间戳如何记录、前序关联和采集完整性，并与工作量、实际 slot、缓存状态、并行资源争用等备选解释一起分析。它没有证明间隔造成 scatter 变短、scatter 等待或设备空闲；选中任务间的空隙可能包含其他任务或未记录内容。也不能排除时间戳生成与导出关联、图重放、性能状态等共同因素，或由于数据选择形成的模式。不同 rank 和不同层记录也不能当作独立重复实验。后续若不能从现有记录证明时间戳及运行时对象对应关系，就保留这些解释，不能用分离现象估算融合收益。
