# Indexer：分采样轮次与配置分支

`indexer-attention-by-run.json` 是 16 rank 的已有 kernel CSV 和 source-validated 模块窗口的只读复算结果；`analyze_indexer_attention_by_run.py` 记录了输入的哈希并校验每层每轮 16 个 rank。表中为各轮 16 rank 单次 Indexer 耗时的中位数，run 是各 rank、model 内的采样轮次；不单靠这个数字认定请求身份。

| Indexer 层 | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| 2 | 19.72 μs | 91.13 μs | 90.92 μs |
| 8 | 21.02 μs | 92.36 μs | 92.44 μs |
| 14 | 21.20 μs | 92.51 μs | 92.59 μs |
| 20 | 20.09 μs | 209.77 μs | 210.03 μs |
| 24 | 21.56 μs | 495.50 μs | 495.58 μs |
| 28 | 21.68 μs | 494.45 μs | 494.31 μs |
| 32 | 22.78 μs | 495.39 μs | 494.98 μs |
| 36 | 21.65 μs | 495.03 μs | 494.81 μs |

这解释了此前每 rank 三次均值明显低于合并 48 次的中位数：在该样本中 run 1 较短，run 2 与 3 较长。对于每一层，三轮中的输入形状类别和 rank 数一致；但 profiler 形状只是缓冲区维度，未记录每轮有效 cache/query 长度、候选数量等动态值。**同形状不等于实际工作量相同，不能因此推断时间差由调度、缓存或算子实现中的哪一项造成。**

冻结配置 `evidence/model-config.json` 指定 `candidate_source_layer_id=20`、`candidate_topk_blocks=2048`，并列出 index source 层 2、8、14、20、24、28、32、36。冻结 `model.py:519–548` 根据这些配置建立层角色：20 是 candidate source，24 及之后的 index source 使用 candidate filter。冻结 `indexer.py:274–285` 在**非 packed 分支**把 candidate source、candidate filter、普通情况分别设为 mode 1、2、3，并只在 filter 时传 `candidate_topk_index`；候选张量在源码中形状是 `[query_rows, 1, 2048]`。

这与后四个 indexer 层 profiler 输入中出现 `[3,1,2048]` 的现象一致，也构成后续核查具体分支的线索；目前尚未证明每个 trace 记录的参数位置与 tensor 身份、真实运行分支或 `candidate_mode`。它并不证明额外候选输入、模式、shape 或其他因素中的某一项导致约 495 μs 的耗时。

继续分析应优先检查已有 trace 是否记录 `cache_seq_lens`、`candidate_lengths`、候选数及动态元数据，或记录关联该调用的 mode；若没有则明确未知。比较之前需固定样本轮次、条件分支与实际工作量，避免把 3 轮混合均值当成稳定表现，更不能推断端到端收益。

## 同窗口 SparseFlashMla 也随轮次变化

已有 `indexer-attention-by-run.json` 同时保存每层每轮同 rank、model、主 stream 窗口内 SparseFlashMla 的耗时。将其组中位数与 Indexer 的组中位数并列，单位 µs，每格各是 16 张卡的独立中位数：

| 层 | run 1 Indexer / attention | run 2 Indexer / attention | run 3 Indexer / attention |
|---:|---:|---:|---:|
| 2 | 19.72 / 17.82 | 91.13 / 46.37 | 90.92 / 45.77 |
| 8 | 21.02 / 16.64 | 92.36 / 44.36 | 92.44 / 43.91 |
| 14 | 21.20 / 18.75 | 92.51 / 47.11 | 92.59 / 47.12 |
| 20 | 20.09 / 16.22 | 209.77 / 44.30 | 210.03 / 44.24 |
| 24 | 21.56 / 16.54 | 495.50 / 44.08 | 495.58 / 44.28 |
| 28 | 21.68 / 22.21 | 494.45 / 46.37 | 494.31 / 46.27 |
| 32 | 22.78 / 17.08 | 495.39 / 44.16 | 494.98 / 43.83 |
| 36 | 21.65 / 18.05 | 495.03 / 44.81 | 494.81 / 44.53 |

在这 8 层、3 个已保存采样轮次中，Indexer 中位数由短变长时，同窗口 attention 的**自身任务持续时间中位数**也由短变长；这能排除“除了 Indexer 耗时，attention 自身任务时长完全不变”这一狭义描述。它不证明逐 rank 都发生相同变化、这些轮次是相同请求，不提供 attention 的有效输入形状或动态数据，不证明 Indexer 导致 attention 变长。请求/cache/query 工作量变化、算子内部数据相关路径、设备资源竞争和采集扰动仍是未区分的可能性；整体指标、物理 shape 和静态模式不能替代实际动态值。

已有 EP8 精确记录证明目标层 24 的三次 Indexer 复用相同 graph TASK ID、无名属性和物理 shape，但不包含有效 query/cache/candidate 值；已查范围及最小关联缺口详见 [indexer-dynamic-metadata-ep8.md](indexer-dynamic-metadata-ep8.md) 和 [request-dynamic-metadata-coverage.md](request-dynamic-metadata-coverage.md)。当前同窗口 attention 对照增添了多算子共同变化的事实，尚无已保存的可绑定工作量值可判定根因，不重复相同字段搜索。

复核输入 `indexer-attention-by-run.json` SHA-256 `0f728cbf37a5fc32bdaa924a8c206482240c7a23241549ef54efcf60344e01e3`，生成脚本 `analyze_indexer_attention_by_run.py` SHA-256 `f81a3140f6f36f139c4b5907d96d52fe05d075e8158ba17895a6c8058b834100`；脚本以 `(rank, run, layer)` 选同一 model/stream 的唯一 attention 与至多一个 Indexer，每个 Indexer 层/轮有 16 个 rank。来源是已保存导出，未触设备。

## 逐 rank、逐层的跨轮配对复核（stage175）

对已归属的 16 个 rank、8 个 Indexer 层，各按**相同 `(rank, layer)`**连接 run 1、2、3 中同 model、同主 stream、同层窗口内唯一 Indexer 与唯一 SparseFlashMla；每轮都验证 Indexer 结束不晚于 attention 开始。按每个 rank、每层分别计算 `后轮任务时长 / 前轮任务时长`，再计算范围与中位数。16 rank × 8 层共 128 个配对单位，每个单位做 1→2、1→3、2→3 三组比较；每层每组均有 16 个 rank。两个不同算子的统计中位数不必来自同一个 rank。

| 层 | run 1→2 两项都变长 | Indexer 倍数范围 / 中位数 | Attention 倍数范围 / 中位数 |
|---:|---:|---:|---:|
| 2 | 16/16 | 3.95–5.36 / 4.63 | 1.72–2.89 / 2.63 |
| 8 | 16/16 | 3.39–5.38 / 4.41 | 1.75–3.00 / 2.67 |
| 14 | 16/16 | 3.27–5.32 / 4.36 | 1.63–2.82 / 2.52 |
| 20 | 16/16 | 6.85–12.42 / 10.44 | 1.67–3.06 / 2.72 |
| 24 | 16/16 | 12.91–26.42 / 23.02 | 1.68–3.02 / 2.64 |
| 28 | 16/16 | 15.91–28.67 / 22.91 | 1.60–2.99 / 2.25 |
| 32 | 16/16 | 15.19–27.85 / 21.87 | 1.57–3.00 / 2.59 |
| 36 | 16/16 | 13.88–27.86 / 22.97 | 1.59–3.00 / 2.51 |

1→3 同样是 128/128 个 `(rank, layer)` 两种算子都变长；2→3 不是持续单向增长：Indexer 63 增、64 降、1 相等，attention 41 增、86 降、1 相等。这个范围内，1→2 与 1→3 的共同变长不是由少数极值推高组中位数造成；128 对的 Indexer 导出输入 shape 文本相同，仍不能证明内容、有效长度或工作量相同。每个 `(rank,layer,run)` 的 Indexer 在其 attention 之前是同 stream 导出时序；不单凭相邻关系证明实际 tensor 身份及所有依赖。

**重要口径：**跨轮并未关联为同一请求、同一有效输入、同一 cache 或受控实验；run 是各卡内采样序号，并非请求 ID；算子任务时长不等于端到端关键路径。故上述成对方向只描述采样现象，不证明某因素造成变慢、不决定计算/访存/调度哪项是瓶颈，也不能推算可节约的时长。不做跨卡时间戳相减。

独立复核：384 个 `(rank,layer,from,to)` 唯一，128 个单位三组齐备；逐条重算时间差、倍数、方向和 Indexer→attention 间隔，复核 24 个分组统计；三轮每层两算子中位数均与原 `indexer-attention-by-run.json` 完全一致。原始窗口 SHA 与 16 份 CSV SHA 与 stage082 一致；保留相同 CSV 采集、shape/时间精度与角色归属限制。结果 [indexer-attention-paired-runs.json](indexer-attention-paired-runs.json) SHA-256 `d8d9a7e651a82616b5387557f4568dba96ed78b30da17cccf575da9b602de0d3`；查询脚本 `analyze_indexer_attention_paired_runs.py` SHA-256 `4e5ca637f0697d415989e004c935e6a0c6fbdf0426546ab4b7ebacc3b307a032`。ZCode stage `operator-ceiling169-v2-175-paired-indexer` exit 0、`model_verified=true`、执行模型 `deepseek-flash`；只读既有 CSV，没有 NPU 负载。

## Indexer 在同卡层窗口中的主流时间占比（stage177）

将 stage175 的 Indexer、attention 时间区间与固定 `module-call-metrics.json` 按 `(rank, layer, run)` 对齐：384/384 个窗口内，两者在同一已归属 model/main stream，Indexer 结束后 attention 开始，且两条均位于层窗口内。对每个窗口先计算 `Indexer 时长 / 层窗口时长`，每层每轮再取 16 卡的中位数；下表括号内是 16 卡范围：

| 层 | run 1 | run 2 | run 3 |
|---:|---:|---:|---:|
| 2 | 2.13% | 8.41% | 8.41% |
| 8 | 2.20% | 8.66% | 8.23% |
| 14 | 2.16% | 8.54% | 8.71% |
| 20 | 2.20% | 18.86% | 19.78% |
| 24 | 2.45%（2.10%–4.39%） | 36.92%（36.58%–37.49%） | 38.26%（38.01%–38.98%） |
| 28 | 2.67% | 36.16% | 36.02% |
| 32 | 2.93% | 36.09% | 36.39% |
| 36 | 2.79% | 35.75% | 36.87% |

第 24 层按同窗非重叠的 `Indexer + attention` 两条任务时长相加，再除以该窗时长，每轮 16 卡的中位数分别为 **4.35%、40.28%、41.72%**。这是对同一条已归属主流的时间占比，并非两项分别取中位数后相加；窗口是当前 source/角色规则定义的 decoder 层区间，可能与其他流、设备、CPU 或其他请求并发，分母也不是整请求耗时。这里没有量化完整关键路径、可优化收益或未被其他流掩盖的比例。

附带检查已有 F07 选中记录：EP8、层 2 和 24 的 run 1/2/3 共六窗，筛选结果中没有 stream 标签不同于主流、且与 Indexer 时间相交的记录。**限定范围很窄**：F07 原始查询仅保留七种指定算子类型且必须完整落在该层窗口内；它没有按照 model 过滤，缺 model/stream 标签时也无法判定实际归属。其他算子、跨层窗任务、未导出活动及其他 rank 都未被这个结果排除。stage177 输出 limits 中将 F07 的选择描述成“same model”过窄；这里以原查询脚本实际条件为准，不应据此认定有完整其他流覆盖，更不能因无相交声称没有通信或争用。

独立复核 384 条任务时间与 stage175 全部匹配、窗口端点/model/主流与源窗口匹配、每条时间次序及分子分母一致、24 组 16 卡最小值/中位数/最大值均可复算；六个 F07 结果与保留的事件记录逐项相符。结果 [indexer-layer-window-occupancy.json](indexer-layer-window-occupancy.json) SHA-256 `ca1d1dabed7cede2a95bc1ac117d24ad68f8fd2368f32c3aa225583cb3e9afc8`；脚本 `summarize_indexer_layer_window.py` SHA-256 `bb717cfae2e9524d6b5afa8aad728f34528f096a4d186cd82a7195239915e638`。ZCode stage `operator-ceiling169-v2-177-indexer-layer-window` exit 0、`model_verified=true`、模型 `deepseek-flash`；仅复算本机保存文件，未运行 NPU。

## Indexer 窗口内其他 stream 的已导出记录（stage178）

对上述 16 卡 × 8 层 × 3 轮共 384 个 Indexer 任务，按同卡目标时间窗读取 16 份冻结的 `kernel_details.csv`，每份 CSV 与 stage175 的 SHA-256 一致；每个目标必须在其 model、主 stream、起止时间和算子类型上唯一匹配。扫描 CSV 中所有正时长、与目标有交集且 stream 标签不等于主流的行，裁剪至 Indexer 窗口，**先按集合求时间并集**。此口径允许任务跨层窗口边界，且不限制被比较行的 model；不涉及跨卡绝对时间相减。

结果是在 384 窗口中：

- 380 个窗口没有匹配到任何上述其他 stream 的 CSV 行；包括第 24、28、32、36 层三轮的全部 192 个窗口。其余 4 个窗口是第 8 层 run 1 的 EP3、EP4、EP12、EP15，各出现一条 `allgatherAicpuKernel`（`Accelerator Core=AI_CPU`、stream 92、model 字段 4294967295）。这四条裁剪区间各自覆盖其 Indexer 窗口的 100%；这只是导出区间重合，不能解释为该通信已全部隐藏或实际链路持续传输。
- 严格标签集合 `AI_CORE`、`AI_VECTOR_CORE`、`MIX_AIC`、`MIX_AIV` 的其他**数字** stream 在 384/384 窗口无匹配行；上述 AI_CPU 的四条被保留在未分类数字 stream，而没有误算进这一组。按 `Type` 或 `Name` 以 `hcom_` 开头的记录单列为可能汇总，非数字 stream 单列为未知；这两组本次均没有相交行。核心标签只是 CSV 名称分类，不等于设备活动计数或对其他类型的完全覆盖。
- 这比前述 F07 七类、六个窗口的检查覆盖更广，但**仍只覆盖这次 CSV 已导出行**。未采集/未导出的设备任务、同主流事件、硬件内部并发、图内部工作、CPU 侧活动均不在“其他 stream CSV 行”的集合中；0 行不能证明芯片空闲、未发生资源竞争、没有通信或 Indexer 在端到端关键路径。不同类型、核心和 stream 的时间并集有可能交叠，不能相加；这四条记录也未证明与目标同请求或同一数据链。

独立核对 384 个唯一窗口与 stage177 目标逐一一致、输入 16 个 CSV 哈希与 stage175 记录一致；每条已保存的裁剪区间、类别、并集、24 组中位数及非零卡数均重算通过。分类规则同时检查源行 `Name` 和 `Type` 的 `hcom_` 前缀；已保存结果没有输出 `Name`，本次四条均为未分类类别、没有 hcom 命中，输出本身无法独立重建任意历史行的 `Name` 分类，需结合查询源码和原 CSV，不将其升级为更广泛的通信覆盖证明。重复行时间按并集计算，本结果中没有完全相同的裁剪记录。

结果 [indexer-other-streams.json](indexer-other-streams.json) SHA-256 `4edc655cf0126bd3752a55cf5d9785c6224639d8aceddd1aa998bf0309adfdb3`；查询脚本 `query_indexer_other_streams.py` SHA-256 `dcda91448ecf33b3de5ce39dd977f50abc13645ecf72cad345ce07b9ecde4b09`，wrapper `run_indexer_other_streams.py` SHA-256 `53a0b5fd28bca4832a5013a305bd4e6830cebefc20b4eb690289dc3c07a17ba3`。ZCode stage `operator-ceiling169-v2-178-indexer-streams` exit 0、`model_verified=true`、模型 `deepseek-flash`；只读 CPU staging 容器中的既存文件，无新 NPU 负载。

## EP8 的固定 Indexer 计数器导出（stage179）

从与 stage175/177 同一 SHA 的 EP8 `kernel_details.csv`，以 Indexer 类型、model、stream、起止时间精确选取第 2 层和第 24 层各三轮；六条各唯一匹配，所选 14 个计数器字段均有列，以下保留 CSV 数值原文：

| 层 / run | 时长 µs | AIC cycles | AIC MAC ratio | AIC MTE2 ratio | AIV cycles | AIV vector ratio | AIV MTE2 ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 / 1 | 20.240 | 554323 | 0.0 | 0.007 | 1519055 | 0.083 | 0.1 |
| 2 / 2 | 90.438 | 660807 | 0.061 | 0.137 | 7581350 | 0.039 | 0.024 |
| 2 / 3 | 89.638 | 667918 | 0.06 | 0.136 | 7512366 | 0.039 | 0.024 |
| 24 / 1 | 32.020 | 677479 | 0.0 | 0.007 | 2535250 | 0.058 | 0.07 |
| 24 / 2 | 498.491 | 1500990 | 0.054 | 0.111 | 42844778 | 0.03 | 0.006 |
| 24 / 3 | 498.170 | 1501769 | 0.054 | 0.109 | 42816443 | 0.031 | 0.006 |

这六条 CSV 记录在短轮与长轮呈现不同的**导出字段轮廓**；例如第 24 层从 run 1 到 2，按报告显示值直接计算 `AIV cycles × AIV vector ratio` 分别为 **147044.500** 和 **1285343.340**，后者约 **8.74 倍**。这只是字段乘积，是否能视作对应活动周期、其跨 core 聚合及计数分母，需要当前实际 exporter 的版本和语义契约；它不是 FLOPs、有效计算时间或已证明的工作量增长。相同计算得到的 `AIV cycles × MTE2 ratio` 约 177467.5 和 257068.7，也不能解释为 HBM 字节或可将两者相加的任务阶段。显示值如 `0.0` 或 `0.006` 精度有限，无法将前者视为实际零，也无法凭小数位推断无误差；实际四舍五入和低值误差界未证。

短、长轮次的显示字段不同，不支持把两次记录简单描述成“所有已导出计数器都不变”；但缺少可验证的 workload/tensor 值、计数器硬件与 exporter 的严格语义和性能采样误差，仍不能断言有多少真正计算或搬运、等待占比、造成长耗时的根因或收益。不同轮不是同一请求/缓存/受控实验。第 24 层三个相应 TASK 的 `globalTaskId=19466`、`connectionId=184330`、`taskId=1275` 复用；第 2 层三个 CSV `taskId=198` 也相同。因此此处使用时间、model、stream、类型在 CSV 内唯一对应任务记录，**没有用重复 ID 或 PMU 时间邻近声称逐次 raw-PMU 已绑定**。各 CSV 数值由 exporter 写到对应行，但 exporter 是否用正确的重放粒度生成每个计数器，尚未独立证明。

`indexer-fixed-metrics-ep8.json` SHA-256 `037b4d2b5a1aeb2a7d8d65bf54291b6c4fe327616211b3b727c96fd92d814e7e`，保存各行其余 scalar、MTE、icache 和 cube 字段原文；查询脚本 `query_indexer_fixed_metrics.py` SHA `a24612c882654bedb9c99647f8de9affa4581ab8d03e91b65242241a98e0e846`，wrapper `run_indexer_fixed_metrics.py` SHA `4568ef04c37de8b07240ce6dbba8c3a8dbb7a984974abad4c62b2b1b9434e69c`。EP8 CSV SHA `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 与 stage175 一致，六个起止与 stage177 逐条核验；stage `operator-ceiling169-v2-179-indexer-metrics` exit 0、`model_verified=true`、模型 `deepseek-flash`。只读已有记录，没有启动 NPU。

### stage179 的计数器语义审计边界

已保存 [Ascend MindStudio Insight 固定版本字段说明](https://github.com/Ascend/msinsight/blob/efb28234772093b0cabc3c538e71375c957540b2/docs/zh/user_guide/system_tuning.md)，其本地快照 SHA-256 `4c431dac256f6bb69927b721d49316f2716e2b1fc2228924d8d8a84bff5d9fa3`。该文档将 `aic_total_cycles`、`aiv_total_cycles` 描述为相应核心执行的总周期，将 `aic_mac_ratio`、`aiv_vec_ratio` 描述为相应单元耗时与总耗时的比例，并列出 PipeUtilization 和 Level1 以上的展示条件。但其措辞未给出本次 MIX 多核场景的具体分母、对多核心的聚合算式、周期与比例是否使用相同时间基准或每次图重放如何分配记录。不能仅凭一般文字证明 stage179 的 `cycles × ratio` 是真实活动周期数、工作量或等待时长，也不能凭文档中的简单瓶颈标签判定实际根因。

此前同镜像 profiler 的 849 个 `.py`/`.sql` 文件，围绕 `TASK_PMU_INFO`、大小写变体及 `globalTaskId` 搜索，仅找到两个 TASK 消费处，没有找到生成 PMU 时间戳、逐次重放匹配与 MIX 聚合的契约；范围不含二进制、其他路径与拼写，记录详见 [loop-004](loop-004-pmu-export-source-conclusion.md)，证据 `pmu-export-source-locations.json` SHA-256 `94c96fbafd25f632845de145b2c62b1050496088081824a8925f6397b6c0af40`。这是旧查询的实际覆盖，并不证明实现不可获取，也不因结果不足继续重复扫描。

stage179 六条在 CSV 内通过 `start/end/model/stream/type` 唯一筛选且保留原文；短长轮报告的字段不同是可直接复核的**导出数据事实**。第 24 层既有数据库记录证明 globalTaskId 和 connectionId 在重放间复用，不足以将 raw PMU 行唯一配对；该点与 CSV 每行导出数字的独立精确匹配是不同问题。实际 exporter 版本、报告精度、取数与重放映射以及动态工作量尚未全部闭合，低 ratio 不能解释成等待。

可针对六条所在的**单个 EP8 profile**，只读检索已哈希绑定的 `profiler_info_8.json` 版本与采集配置白名单：源 SHA-256 `a249b01a52dc6d9cbbd34953f2573b52d1d9f7280ff2d561b228b4a9417eb544`。若能确定采集等级和 metrics，只能检验与文档展示前提是否相容，不足以证实导出公式或 replay 契约。已准备查询 `query_ep8_profiler_config.py` SHA-256 `cd5d8baa7ed40943428fee364e574c572a6e3479357c126a81ba8f52f0c7f121` 及 wrapper `run_ep8_profiler_config.py` SHA-256 `fb4b4aefa7b9e54e8cc64f32bcaeea6a9a243b3f5db4430e92c441e0a96215f3`；等待主控安排 ZCode，未运行 NPU。

### EP8 采集元数据只读核验（stage181）

在与目标样本相同 EP8 profile 的 `profiler_info_8.json` 中，固定白名单查询得到 `torch_npu_version=2.10.0.post7`、`cann_version=9.2.0`；本次没有匹配到白名单中直接命名的 profiler level、AIC metrics 或 export type 字段。这只表示指定键名在该文件限定遍历范围没有命中；字段可能用其他名称、结构或另一个文件记录，不能断言相关采集开关未设置，也不能据此验证 Level1/PipeUtilization 条件。版本字段是这份会话元数据自述，不单独证明现场实际使用的 exporter 二进制、其算式、MIX 计数聚合或按重放写入机制正确。

源文件的路径、1391 字节与 SHA-256 `a249b01a52dc6d9cbbd34953f2573b52d1d9f7280ff2d561b228b4a9417eb544` 与先前原始文件清单对应；输出 [indexer-ep8-profile-collection.json](indexer-ep8-profile-collection.json) SHA-256 `7ac285853f5a57f4b43e4c36f6907c23bac6a24eedd5e9245dcda1e83ab0c405`，查询脚本 SHA-256 `cd5d8baa7ed40943428fee364e574c572a6e3479357c126a81ba8f52f0c7f121`。ZCode stage `operator-ceiling169-v2-181-ep8-profiler-config` exit 0、`model_verified=true`、模型 `deepseek-flash`；只读固定元数据，没有 NPU 负载。阶段 179 的数值解释与不确定性不变。

## 同卡同层跨轮窗口残差（stage198）

为检验“跨轮层窗口增长仅由这两段任务变化组成”的狭义描述，复用 `indexer-layer-window-occupancy.json` 的 16 卡、8 层、3 轮共 384 个已归属窗口，按 `(rank,layer)` 配对，分别比较第 1 轮与第 2、3 轮。逐条核对：每个窗口的 Indexer、SparseFlashMla 都处于层窗口内，且 Indexer 结束不晚于 attention 开始；两段和窗口自身时长均与其保存的纳秒起止端点一致。结果依然继承原 exporter、时间边界和层归属限制。

每一对样本计算 `残差变化 = 层窗口时长变化 − Indexer 时长变化 − attention 时长变化`，再对每层 16 张卡各自的残差取中位数。1→2 的结果（μs）如下；其他列也是各自的成对差值中位数，因此各列中位数不一定直接满足减法关系。

| 层 | 层窗口变化 | Indexer 变化 | attention 变化 | 残差变化 | 残差正/负卡数 |
|---:|---:|---:|---:|---:|---:|
| 2 | 155.69 | 71.63 | 28.95 | 55.94 | 16/0 |
| 8 | 111.27 | 71.59 | 27.98 | 11.79 | 16/0 |
| 14 | 99.40 | 71.75 | 28.52 | 1.20 | 9/7 |
| 20 | 199.87 | 190.72 | 27.75 | -18.95 | 3/13 |
| 24 | 459.37 | 474.23 | 27.47 | -40.84 | 0/16 |
| 28 | 559.33 | 474.40 | 26.98 | 58.43 | 16/0 |
| 32 | 598.39 | 474.43 | 27.27 | 97.88 | 16/0 |
| 36 | 614.24 | 474.33 | 27.20 | 111.50 | 16/0 |

这 128 对层窗口均增长；第 24 层每张卡的残差变化均为负，第 28、32、36 层每张卡均为正。就已导出的层窗和这两段任务而言，不能把跨轮变化简单描述为“其他窗口部分完全不变”。**残差是层窗内扣除两段区间后的剩余时间的跨轮差值，不是某个算子的时间或等待时间**；其他流可重叠、记录有盲区。轮次并未绑定同一请求、真实动态长度或等工作量，所以这也不能确定慢的原因、推导可节省时间、证明关键路径或预测端到端收益。1→3 的完整逐卡结果和统计一并保留于 JSON。

输入 `indexer-layer-window-occupancy.json` SHA-256 `ca1d1dabed7cede2a95bc1ac117d24ad68f8fd2368f32c3aa225583cb3e9afc8`；脚本 `compare_indexer_layer_deltas.py` SHA-256 `848df1794909b909a2515cf30febf4ce05e01bdb458ea553968d3394e625b367`；结果 [indexer-layer-paired-deltas.json](indexer-layer-paired-deltas.json) SHA-256 `b56d438b88ff615be37ce4dfd149ad7a8426c2dd73f4f982a0f3bf0f4b4cc0a8`。结果 256 对逐条按保存的纳秒端点独立重算，16 组中位数和正负计数复核一致。ZCode stage198 exit 0、`model_verified=true`；只读既有聚合文件，没有启动 NPU。

## EP8 第 24 层三个样本的属性与 metadata 分工线索

复核已有 [indexer-runtime-metadata-ep8.json](indexer-runtime-metadata-ep8.json)：在该 rank、该层的 run 1、2、3 中，导出的 13 项 `attrInfo` 字符串完全相同，输入形状和 dtype 字符串也完全相同；任务时长分别为 32.020、498.491、498.170 微秒。这个对照说明，所导出的属性字符串本身无法区分这三次样本；它并不证明实际有效长度、输入数据、metadata 内容、tiling 或设备工作量相同。只覆盖这三条记录，不能推广到其他 rank 或层。尽管当前源码的 OpDef 注册属性顺序与这些字符串形式相容，尚未核实 exporter 的序号契约和历史二进制，不把 `attr_N` 映射为已确认的语义。

当前容器的已保存 arch22 源码 [`quant_lightning_indexer_v2_kernel_arch22.h`](../fusion/frozen-source/csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h) 第 221–312 行展示：条件满足时从设备侧 Q/K 长度与 residual 输入读取或计算实际长度；第 363–403、466–469 行在 metadata 非空时读取每核 enable、BN2/M/S2 起止和部分 LD workspace 信息。字段索引及布局见 [`quant_lightning_indexer_v2_metadata.h`](../fusion/frozen-source/csrc/attention/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_metadata.h) 第 23–64 行。该文件的 `GetTotalBaseBlockNum` 有声明与定义，但没有在此头文件内找到调用，不能认为历史调用执行了该函数。上述均是当前源码的条件行为，历史二进制和实际分支尚未绑定。

这些信息为区分不同工作量提供了取证路径：若未来在已有资料中找到与目标 rank、层和具体 replay 可验证关联的实际长度、metadata 内容及相应 tiling，并绑定执行版本，便可复核各核划分与归约信息，再与时长对照。当前保存的任务记录只有输入的形状与 dtype，没有这些张量值；固定属性与形状、CPU 侧的乐观长度或请求的全局 32K 口径都不能代替同次设备内容。即使获得分工，也不能仅由它直接判定时长差异的根因或收益。

## 已采样的同卡跨层耗时台阶

复用 stage177 的 384 条已归属 Indexer 窗口，按同一 rank、同一采样序号，对相邻 Indexer 层分别计算后层任务时长 / 前层任务时长。输入哈希、任务端点与时长、16 卡 × 8 层 × 3 序号的完整性均由脚本校验。结果：

| 跨层 | run 1 后层更长卡数 | run 2 后层更长卡数 / 倍数范围 | run 3 后层更长卡数 / 倍数范围 |
|---|---:|---:|---:|
| 14 → 20 | 8/16 | 16/16，2.112–2.346 | 16/16，2.127–2.355 |
| 20 → 24 | 12/16 | 16/16，2.322–2.419 | 16/16，2.321–2.421 |

因此在这些样本中，后两轮的两个台阶不只来自少数 rank，第一轮则没有同样一致的方向；先前按层报告的中位数趋势得到了同卡复核。冻结配置/源码中的第 20 层 candidate source 和第 24 层 candidate filter 是解释方向，但不能凭切换位置判定实际分支，也不能把倍数归因于角色切换：历史调用尚未绑定有效长度、候选数、请求身份、二进制和 mode，其他因素也可能不同。同 rank 同序号不证明同请求或等工作量；这些任务时长不等于关键路径、通信时间或可得收益。

本机只读复算脚本 [`compare_indexer_cross_layer_steps.py`](../../tools/compare_indexer_cross_layer_steps.py) SHA-256 `9ffe9932f84e437c460d516aef45dfd7e6d4404ab936651e937e58ac4b7e8c7d`，结果 [`indexer-cross-layer-steps.json`](indexer-cross-layer-steps.json) SHA-256 `2a6745e12ce007505002e8a87409c9dcd79310020790f5bde79566a5a418e46d`，输入 `indexer-layer-window-occupancy.json` SHA-256 `ca1d1dabed7cede2a95bc1ac117d24ad68f8fd2368f32c3aa225583cb3e9afc8`。结果含 21 组、每组 16 卡明细。只读已存聚合数据，没有启动 NPU。

### 按实际长度可证伪的内核工作量条件预测

对同一已存 arch22 源码作更细的静态审查：`SplitCore` 从非空 metadata 读取 `coreEnable` 和 core 0 的 enable；`Process` 在 core 0 未启用时走 `ProcessInvalid` 清理输出后返回，`ProcessMain` 对未启用的核直接返回（`kernel_arch22.h:363–374,668–705`）。当参与的 batch 上实际 Q 或压缩后 K 长度为零，`CalcGS1LoopParams` 标记零长度，主循环调用无效输出处理并跳过该 batch 的 S2 计算（`:571–579,735–750`）。因此，**若**某次真实执行能绑定这份实现、metadata 非空且同次 core 0 enable 明确非零，就可排除“此次完全走 core-0-disabled 清理分支”的具体假说；但不能排除某些 batch 的零长度、局部空核、其他早退或等待，也不能由此推断耗时原因。当前数据没有这些同次 metadata 值。

长度还可以给出明确的 tile 数预测：源码 `S2_BASE_SIZE=2048`；在无 attention mask、参与核启用、某次 `(batch, head, query tile)` 的 `s2Start=0`、该次有效末尾未被 `s2End` 截断的条件下，`CalcS2LoopParams` 的 S2 基本块数等于 `ceil(压缩后有效 K 长度 / 2048)`。例如同一条件下有效 K 长度分别为 2048 与 2049 时，对应 **1 与 2 个**有效 S2 基本块；`ProcessBaseBlock` 对每个有效块走 `ComputeMm1`（`:540–560,751–758,780–797`），额外 preload 迭代不能算作有效块。若以后有逐次设备长度、mask、分片边界与历史 binary 绑定，可直接复算并证伪这个预测；只有这组值满足条件才能适用它。**这不表示耗时必然翻倍**，也不说明当前短轮/长轮实际跨越了任何边界；现有物理 shape、固定候选容量和全局 32K 输入都不能替代实际 K 长度。现有 F12 已记录此路径先算后应用候选 mask，以上是与动态 K 长度、空任务分支有关的补充，不是已证本次性能根因。

源码 `kernel_arch22.h` SHA-256 `d35c26e6cb781aebad163a2c85cbd84f68c7c838ebe5ef9f67e6a42777949df6`。本次仅复核已有源码，没有新增 NPU 工作。
