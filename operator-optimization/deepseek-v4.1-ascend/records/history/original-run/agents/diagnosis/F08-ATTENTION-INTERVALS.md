# F08 缓存写入与注意力区间：同卡同 scope 配对

复用 [target-layer-all-kernels-16rank.jsonl](../../diagnostic-20261029/target-layer-all-kernels-16rank.jsonl) 和 [F08 triples](../fusion/f08-actual-triples.json)，执行 `python3 agents/diagnosis/check_f08_attention_intervals.py`。原数据哈希由脚本硬校验，所有计算使用 Decimal。只比较同 EP、同 layer scope、相同 device track 的已匹配流链接任务；每个 scope 内没有重复 flow ID，目标 F08 三个 flow 各唯一。不同 flow 不保证是彼此独立的底层硬件工作或采集完整。

| 已匹配类型 | scope 数 | F08 scatter 在前且不重叠 | 不同 stream | scatter 结束到该任务开始的间隔，最小 / 中位 / 最大 |
|---|---:|---:|---:|---|
| SparseFlashMla | 1280 | 1280 | 1280 | 25.53925 / 36.039 / 30908.780 µs |
| VllmAscendQuantLightningIndexerV2 | 256 | 256 | 256 | 176.09625 / 384.75175 / 28818.960 µs |

1280 个 scope 均恰有一条匹配的 SparseFlashMla；其中 256 个 scope 匹配到一条 Indexer，其余 1024 个没有匹配该类型。256 组中 Indexer 和 attention 同 stream 且前者先结束，间隔最小 / 中位 / 最大为 148.67725 / 218.640 / 1706.98575 µs。注意存在 Indexer 的 256 个 scope 只来自层 2、8、14、20、24、28、32、36，每层各 32；另组来自其他层。以 Indexer 是否匹配分组，scatter 到 attention 的间隔中位数分别为 3634.297125 µs（256）和 35.469125 µs（1024）；该分组伴随层与工作内容变化，不是等工作量实验，不能归因于 Indexer，不能把间隔视为设备空闲。

## 结构重组线索与限制

冻结的 [dsa_v1.py](../fusion/frozen-source/vllm_ascend/attention/dsa_v1.py) 中，`_maybe_update_compressed_caches_and_select_topk` 在 `compress_ratio == 4` 分支返回 Indexer 的结果；`_forward_attention` 在该分支将结果赋给 `cmp_sparse_indices`，再调用 attention op。这支持一个条件性生产消费关系，但该源码路径、返回张量和 256 组 trace task 的逐条运行时绑定尚未证明。Indexer→attention 可研究分块流式交付、局部选择与全局合并，但必须保留完整 top-k、掩码、排序及相同输入的数值与 tie 语义；不能假定每个局部 top-k 已能给出全局结果或省掉完整选择。对 F08 scatter→attention 的拆分融合，必须先确认实际缓存写入的别名、读写顺序、跨 stream 事件、块映射和持久化边界；同 scope 且先后出现不证明两者访问同一张量。缺少这些证据时不把融合写成已可实现或可省下某个间隔。

此次只判断选中记录的区间关系，不证明其他流没有并行任务、不证明内部等待、计算资源闲置或端到端关键路径，不提供提速预测。profile 本身存在扰动，间隔不能与未 profile 的服务耗时混算。缺少某类型只说明该 scope 中没有匹配导出记录，不能证明相应工作不存在。每个范围和每个任务的精确 flow/时间保存在 [f08-attention-intervals.json](f08-attention-intervals.json)，哈希 `355cd1a831403c1129ab76e128c0803da39b05712c3304c5c285aa82e6744e6c`；脚本哈希 `14fb2175b2797eb6e51718412e7d14f8124296992792c66cb903cc0ab02159bd`。仅 CPU 读取已保存文件，未启动 NPU 工作。

## EP3 第 36 层的 Indexer：同摘要、同物理签名对照（2026-10-30）

上文 256 组只建立 scatter、Indexer、attention 的部分区间关系；为核查单条 **5074.260 µs** Indexer 是否在所选数据内突出，复用同一个既存重采集聚合（输入 SHA `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`），筛匿名调度摘要 `db852bff…`、EP0–7 和层 2、8、14、20、24、28、32、36，共 **64 个层 scope**。每窗在已关联的记录中各恰有一个 Indexer、一个 `_prepare_indexer_indices_kernel` 和一个 `SparseFlashMla`；三者都在同一设备 track，导出顺序依次为 Indexer → prepare → attention，记录区间不交叠。这是**已关联任务的筛选结果**，不排除未匹配任务或其他流；不同任务同轨及次序不证明数据、事件或请求依赖。

进一步要求三项任务的完整导出输入/输出 shape、dtype、format **均与 EP3 第 36 层一致**，并保留同 track、依次不交叠条件，得到 **28 窗**：EP0–6 各四窗，层 24、28、32、36 各七窗。它们的单任务记录时长为：

| 任务 | 数量 | 时长最小 / 中位 / 最大（µs） |
|---|---:|---:|
| Indexer | 28 | 5072.040 / 5148.727 / 5743.40525 |
| prepare | 28 | 215.376 / 217.315875 / 224.9955 |
| SparseFlashMla | 28 | 878.82225 / 890.342 / 905.940 |

参考 EP3 L36 的三条分别为 5074.260、217.160、898.260 µs。QLI 的 5074.260 µs 在这 28 个数里从短到长排第二；另外六张卡和四个目标层在该固定筛选下同样出现约 5–5.7 ms 的 Indexer 记录。这个发现排除“仅 EP3 L36 一条记录具有如此时长”的狭义说法，**不能证明它代表线上请求或该算子的一般水平**：28 条来自同一次带 profiler 的采集、同一摘要和有限层，跨 rank 不是独立请求；摘要相同和三任务物理签名相同不保证各 rank 的实际有效长度、索引、输入内容或工作量相同。三个任务的中位数可能来自不同 scope，不能相加为墙钟、关键路径或收益。

36 个剩余 scope 也均有上述三条任务：EP0–6 的层 2/8/14 中 Indexer 和 attention 的完整物理签名与 L36 不同；层 20 的 Indexer 签名不同；EP7 的八层三种签名均不同，L36 三项分别以物理行数 1019 而不是 1023 导出，因此都没有纳入 28 窗。它们不是缺失或失败，也不能从物理行数差单独推算有效 token、原因或性能浪费。若把 64 窗按上述同 scope 条件宽泛汇总，Indexer 的时长范围为 792.640–5743.40525 µs，说明层和签名混合时不宜直接以 5074 µs 代表这批全部情况；该范围并未控制物理签名和工作量。

CPU 复算脚本 [summarize_qli_target_scope_distribution.py](summarize_qli_target_scope_distribution.py) SHA `32be6c76b1ff125d44be1cbb9e3bf52a96258897a56385e802300142b9a352ef`，逐 scope 结果 [qli-target-scope-distribution.json](qli-target-scope-distribution.json) SHA `61aa739e43197d4c0725d81aea30624728de0df9c7b2ffea772d5de53b0ca804`；原始提取包含未匹配 start event，不能由本复算补齐。ZCode 只读独立提取 EP3 的 8 层 × 3 类型数据，本机核对 24/24 数字一致；不把它的其他解释当成额外证据。没有访问 NPU 或修改推理实现。

## 28 个已选同签名 QLI 的导出流水字段

复用上节 `qli-target-scope-distribution.json` 已筛出的同一摘要、同三项物理签名的 28 条 QLI；不再筛选或统计其他同名任务。只读对应 EP0–6 历史 profiler DB：每条按已保存的精确 device start、taskId 匹配 `TASK`，并确认该库内 `globalTaskId` 只出现一次；再查 `TASK_PMU_INFO` 中同 ID 的字段。28/28 各有 28 项不重名字段和一个时间戳，时间戳为该 TASK 结束时或提前 20 ns。该时间接近支持选中记录的对应，不证明是分阶段时间线，也不证明 exporter 的完整语义。下表是数据库数值的任务级最小、中位、最大，不是跨核或跨卡合计：

| 导出字段 | 有值 / 缺失 | 最小 / 中位 / 最大 |
|---|---:|---:|
| AIC MAC ratio | 28 / 0 | 12.612% / 12.901% / 13.219% |
| AIC scalar ratio | 28 / 0 | 11.664% / 11.952% / 12.415% |
| AIC MTE2 ratio | 28 / 0 | 10.010% / 10.278% / 10.540% |
| AIV vector ratio | 28 / 0 | 79.746% / 89.228% / 90.081% |
| AIV scalar ratio | 28 / 0 | 66.717% / 74.462% / 75.376% |
| AIV MTE2 ratio | 28 / 0 | 1.909% / 2.166% / 2.223% |
| AIV MTE3 ratio | 28 / 0 | 0.069% / 0.075% / 0.082% |
| AIC MTE3 ratio | 0 / 28 | 未导出为所查名称；不能当作零 |

AIC 和 AIV 的正周期字段分别为 211,533,505–215,801,438、438,035,691–496,046,777；两者分母与核的聚合口径不同，不能互相加减或折算墙钟。在独立本机校验中，这 28 个任务的 AIC/AIV 周期与已存聚合中的同次 CSV 数值逐条一致；两个 CSV 显示比例与数据库精度值差均不超过 0.000501。两个导出来自同一采集，并非独立实验；选择源和 raw DB 没有在本次逐文件重新散列，但按保存的已哈希选择文件和完整起点/taskId 唯一关联。

这组结果排除的仅是“这 28 个导出记录的 AIV vector 字段都接近零”这一狭义解释。Vector 字段高不证明该算子被 vector 算力限制，也不排除数据、同步、其他流水、采集扰动或不均衡；同一记录 vector 和 scalar 的报告值之和均大于 1，不能将它们当作互不重叠的墙钟阶段或用 1 减比例作为等待。若分母实际一致，二者不能被解释为互斥比例；历史采集的每核、MIX 主从核、分母和 exporter 版本仍需绑定，不因此断言具体指令重叠。

已存 [固定版本 MindStudio 字段说明](https://github.com/Ascend/msinsight/blob/efb28234772093b0cabc3c538e71375c957540b2/docs/zh/user_guide/system_tuning.md)以相应单元和总耗时描述比率；本机 CANN 9.2 的不同非 MIX/MIX 和 chip6 处理路径见 `../fusion/F08-AIV-CANN92-PMU-FORMULA.md`。这两份资料都未直接证明这批 MIX task 的历史导出器和跨核分母。QLI 的实际 mode、每行候选数量/有效长度、设备二进制和具体流水阶段仍未知；因此不能把 AIV 字段差异或 EP6 任务更长认定为原因，也不能由数据估算提速。

复核脚本 [query_qli_28_existing_pmu.py](query_qli_28_existing_pmu.py) SHA-256 `924ab741ee030e53e73f209f9370b981c6979b815f51a8f3c8f15e5a0e694bf9`，只在 stage 容器读取已存 SQLite；结果 [qli-28-existing-pmu.json](qli-28-existing-pmu.json) SHA-256 `d9159196dc1b07b0d303213eed028e20d4d0b9258366ae55a14a2a5625ca8fbb`，保存每条匹配 key 与字段，无输入内容。CSV 对照使用既存 `target-layer-all-kernels-16rank.jsonl` 中已唯一匹配的具体记录，按 `(EP,scope,flow)` 关联，不重复开展全量形状或时长统计。本次未启动 NPU 或修改实现。

## 两张卡的同层对照：比例与分母共同变化

对上一节已保存的 28 条 QLI，固定选 EP3 与 EP6 的同层 24、28、32、36 四对，分别从 `qli-28-existing-pmu.json` 和 `qli-target-scope-distribution.json` 按 `(ep,layer,scope)` 对接；两卡属于同一重采集与同一匿名调度摘要，其三种算子物理签名满足前述固定条件。这个对照不是独立实验或相同实际输入的证明。

| 层 | EP3 / EP6 设备时长（µs） | EP3 / EP6 AIV vector 报告比例 | EP3 / EP6 `aiv_total_cycles × aiv_vec_ratio`（百万，算术乘积） |
|---|---:|---:|---:|
| 24 | 5072.040 / 5743.405 | 90.081% / 79.746% | 394.587 / 395.576 |
| 28 | 5076.040 / 5741.185 | 90.008% / 79.777% | 394.587 / 395.576 |
| 32 | 5074.400 / 5733.265 | 90.038% / 79.887% | 394.587 / 395.576 |
| 36 | 5074.260 / 5733.565 | 90.042% / 79.884% | 394.586 / 395.576 |

逐对计算，EP6/EP3 的 task 时长比为 **1.12984–1.13237**，AIV total_cycles 比为 **1.12989–1.13243**；但 AIV total_cycles 与 vector ratio 乘积的比仅 **1.0025056–1.0025082**，AIC total_cycles 比为 **1.00206–1.00431**。该乘积使用数据库精度而非 CSV 三位显示值；它只是**字段乘积**，不能认定是真实向量指令数、单核活动周期或等量有效工作。分别选取的时长与字段按单条 `(EP,scope)` 对应，仍不能推断跨设备时钟同步、实际动态频率或每层跨 rank 行值相同。

这组算术说明一个具体错误推断：**不能仅从 EP6 的 vector ratio 比 EP3 低约 10 个百分点，认定向量工作量同比减少或低比例自身就是慢的根因**；这些记录的分母随时长一同变化。它也不能证明两个采样执行了同等计算或 EP6 的额外时间是等待，可能存在指标采集/归约方式、数据、分片、内核阶段、资源争用、频率及 profiler 扰动等差异。现有 mode、实际有效 key 长度/候选、时间分段及底层事件来源未绑定，不能在这些假设中选定根因。表中数据也不是生产负载下可兑现的节省值。

可复算：读取上节两个 JSON，取 `(ep,layer)=(3 或 6,24/28/32/36)`，直接以每条 `aiv_total_cycles * aiv_vec_ratio` 计算并逐层相除，分母和时长单独计算；选择和数据库来源、SHA 与条件见上一节。没有重复扫描 raw 数据或发起 NPU 工作。

## 同卡四层的周期与比例乘积复核（2026-10-30）

直接复算上节同一份 `qli-28-existing-pmu.json`（SHA-256 `d9159196dc1b07b0d303213eed028e20d4d0b9258366ae55a14a2a5625ca8fbb`）中 EP0–6、各层 24/28/32/36 的 28 行。计算方式为每条 `aiv_total_cycles * aiv_vec_ratio` 与 `aic_total_cycles * aic_mac_ratio`；每卡四层的相对极差为 `max(product) / min(product) - 1`。AIV 四层相对极差的跨卡最大值为 **0.000525%**，AIC 对应值为 **0.03497%**。将七卡四层所有 28 行一起比较，AIV 和 AIC 乘积的相对极差分别为 **1.9472%** 和 **6.8639%**。这些比值都只是相对于各自样本最小值的算术范围，不是误差界、性能差距或异构设备差异。

在 Python 双精度运算下，28 行的两种乘积与其四舍五入整数的绝对差均不超过 `3.73e-9`。这与导出比例由整数分子除周期字段得到相容，但**不能反过来证明**历史导出器确实使用该公式，或识别原始 PMU 事件。尤其是四层乘积接近不能证明向量指令次数、每层有效计算量、数据相同或四层使用同一底层计数值；比例和分母均来自同一轮有扰动的 profile，并不是独立实验。该复核只涵盖四个后续层，不包含第 20 层；不能将其用于解释第 20 层与后续层 2.6 倍时长差、认定瓶颈或估计收益。历史 9.2 导出器、核聚合、真实输入和执行二进制仍未绑定。

可在运行目录下用以下 CPU 命令复算，不访问设备或数据库：

```sh
python3 - <<'PY'
import json
r = json.load(open('agents/diagnosis/qli-28-existing-pmu.json'))['rows']
assert len(r) == 28 and {(x['ep'], x['layer']) for x in r} == {(e, l) for e in range(7) for l in (24, 28, 32, 36)}
for cycle, ratio in [('aiv_total_cycles', 'aiv_vec_ratio'), ('aic_total_cycles', 'aic_mac_ratio')]:
    by_ep = [[x['metrics'][cycle] * x['metrics'][ratio] for x in r if x['ep'] == e] for e in range(7)]
    all_values = [v for group in by_ep for v in group]
    print(cycle, ratio, 'within_ep_max_percent', max(max(g)/min(g)-1 for g in by_ep)*100,
          'all_percent', (max(all_values)/min(all_values)-1)*100,
          'max_integer_distance', max(abs(v-round(v)) for v in all_values))
PY
```

## 同采样 QLI 层间物理输入输出角色与时长（2026-10-30）

对上文同一重采集摘要 `db852bff…` 的 EP0–6，在每卡配对 layer 20 与 layer 24/28/32/36 的唯一已匹配 QLI 设备 task，共 35 条记录、28 个逐卡比值。每条取同一种 trace event duration，并由原提取器唯一匹配 CSV，绝对偏差 ≤0.001 µs；配对任务在每卡同一 device track。EP0–6 内同层完整 `shape/dtype/format` 签名一致，四个后续层签名一致；35 条记录的 core、Block Num、Mix Block Num 均为 `MIX_AIC`、24、48。EP7 对应 query 物理行数 1019 而非 1023，因此不纳入这一相同签名对照；不代表 EP7 缺任务。

两个签名的完整字段可以由下方复算脚本打印。共同的前五个输入分别为 INT8 ND `[1023,32,128]`、INT8 ND `[17315,128,1,128]`、FP16 ND `[1023,32]`、FP16 ND `[1023,32]`、FP16 ND `[17315,128,1]`，其余共同输入的各 shape、dtype 和 format 同样完全一致；layer 20 最后一输入记录为空形状、`DT_UNDEFINED/NULL`，其第三输出记录 INT32 ND `[1023,1,2048]`；四个后续层对应最后一输入记录 INT32 ND `[1023,1,2048]`，第三输出形状为空。导出 dtype/format 字段在空输出项仍有值，不能以其独立推断有效输出或读写。冻结配置与源码把 layer 20 描述为候选来源、后四层在条件满足时使用候选；这和观察到的物理字段角色**相容**，但没有绑定运行时 scalar mode、来源输出与后续输入的 storage/版本，不能把它们直接当作同一 tensor 或实际候选读取工作量。

| EP | layer 20 记录时长（µs） | 24/28/32/36 各层记录时长范围（µs） |
|---|---:|---:|
| 0 | 1911.502 | 5079.579–5080.819 |
| 1 | 1992.660 | 5144.557–5147.797 |
| 2 | 2042.819 | 5182.276–5183.556 |
| 3 | 1912.120 | 5072.040–5076.040 |
| 4 | 1989.820 | 5147.797–5150.757 |
| 5 | 2032.420 | 5183.676–5190.376 |
| 6 | 2146.437 | 5733.265–5743.405 |

28 个配对的后续层 / layer 20 时长比最小、中位、最大为 **2.5368、2.5875、2.6758**。这使高时长与一个跨层候选输入/输出**记录角色变化**形成可追查的限定关联，而不是证明“候选过滤导致 2.6 倍开销”：不同层的动态值、有效长度、候选分布、计算路径和 profiler 干扰均未控制，同 shape 和 Block Num 不证明工作相等。设备 task 时长也不是未采样请求的 TPOT、端到端关键路径或可兑现收益。要确认原因，至少需同次绑定 mode、candidate tensor 身份与实际有效候选/长度，并取得算子内部阶段或可靠工作量信息；不授权新增采集时只保留候选。

### 同卡后续层与第 20 层的周期比例字段乘积

对上面完全相同的 35 条记录，使用该 JSONL 所存 CSV 显示字段、同卡逐层相除，28 个后续层 / 第 20 层的 `aic_total_cycles × aic_mac_ratio` 乘积比最小/中位/最大分别为 **0.99612 / 0.99999 / 1.00426**；`aiv_total_cycles × aiv_vec_ratio` 对应为 **2.45114 / 2.51422 / 2.56309**。这些是来自同一 profile 的**字段代数乘积和比值**，不是 MAC/向量指令数、真实工作量或时间分解。CSV 比例只有三位小数，第 20 层未与高精度 PMU 样本绑定，尤其不能根据接近 1 的 AIC 乘积比判定精确相等或差异；后续层高精度对照只覆盖前述 28 条，不能替代缺失的第 20 层高精度值。

这一结果说明两类字段随层的显示变化不同，可继续作为调查线索；不能把 AIV 乘积比与时长比相近当作 AIV 造成时长差的因果证据，不能推断完全没有新增 AIC 工作，或推出可消除的开销。实际 mode、有效输入及候选、硬件事件、频率和历史导出聚合口径都未知。与上一节的“同卡四个后续层之间乘积极差”是不同对照，不要混为一谈。

复算下面的 CPU 命令；输入是上文已核验的聚合文件，不扫描数据库或使用 NPU：

```sh
python3 - <<'PY'
import hashlib, json, statistics
from decimal import Decimal as D
p = 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(open(p, 'rb').read()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
digest = 'db852bff987d2b98d3dc0734e3731ec5528e37dcf6e874860d1ebb4fd12eefc8'
rows = {}
for line in open(p):
    j = json.loads(line)
    if j.get('kind') != 'rank': continue
    for s in j['data']['layers']:
        if s['scheduler_digest'] != digest or s['ep'] not in range(7) or s['layer'] not in (20,24,28,32,36): continue
        tasks = [t for g in s['operator_groups'] if g['operator_name'] == 'VllmAscendQuantLightningIndexerV2' for t in g['tasks']]
        assert len(tasks) == 1 and (s['ep'], s['layer']) not in rows
        rows[s['ep'], s['layer']] = tasks[0]['csv']
assert len(rows) == 35
for cycle, ratio in [('aic_total_cycles','aic_mac_ratio'), ('aiv_total_cycles','aiv_vec_ratio')]:
    def product(e, layer):
        r = rows[e, layer]
        return D(r[cycle]) * D(r[ratio])
    values = [product(e, layer) / product(e, 20) for e in range(7) for layer in (24,28,32,36)]
    print(cycle, ratio, 'min/median/max', min(values), statistics.median(values), max(values))
PY
```

可复算：`python3 agents/diagnosis/check_qli_candidate_role_comparison.py`。脚本 SHA-256 `1a9320b7441f5e3b9766425f56706c88ff099484c1e9cce7852585c23f9708aa`，校验既存 JSONL SHA `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`，打印每卡每层时间和全部物理签名。源码条件参照 `../fusion/F12-CROSS-LAYER-INDEX-PUBLICATION-REVIEW.md`。本次离线只读，未启动设备负载或修改目标实现。

## 256 组 Indexer→attention 间隔里的索引整理任务

进一步覆盖前述全部 **256 个匹配 Indexer 和 attention 的 scope**：间隔中每组均匹配到一个 `_prepare_indexer_indices_kernel`，三条任务在同一个设备 track 和 stream；索引整理任务区间完整位于 Indexer 结束和 attention 开始之间。索引整理记录时长占这段间隔的中位数为 **98.1775%**，241/256 组不少于 90%，212/256 组不少于 95%。这段间隔包含设备任务的记录区间，不能仅凭两个端点就把整段判为设备空闲或纯主机下发耗时；任务区间自身仍可能包含等待，不能据此证明硬件持续工作。

这也不意味着全部间隔都由该任务解释。占比最低的组为 8.7720%；最大间隔 1706.98575 µs 的一组，索引整理任务开始前为 1555.18875 µs。该剩余间隔可能包含其他流任务、遗漏记录、下发或同步等情况，当前不能认定其原因。覆盖率按每一组的两个导出时间区间计算，不是全设备覆盖率，不可和其他任务耗时相加当作服务端到端占比。

冻结的 [prepare_indexer_indices.py](../fusion/frozen-source/vllm_ascend/ops/triton/prepare_indexer_indices.py) 描述对选中索引进行可见性过滤、按位置排序和条件性长度计算；这为 Indexer→整理→attention 的潜在重组提供源码方向，但仅凭任务名称和顺序不能证明历史运行时加载的二进制、实际张量身份或同步依赖。重组时需要保留过滤、顺序、索引缓冲生命周期及相关数值语义，性能收益未知。

下列查询仅重用已存数据，不操作 NPU：首个文件是 `target-layer-all-kernels-16rank.jsonl` 的 16 rank scope 关联记录；第二个文件是上文保存的 256 组 Indexer/attention 配对。查询核对两个文件 SHA、每 scope 所有记录的 flow 唯一、配对两个 flow 存在、目标间隔里恰有一个 prepare、三任务同设备 track/同 stream、prepare 完全位于两个端点之间。`device_start_us` 和 `trace_duration_us` 在文件中以微秒命名，使用 Decimal 避免大时间戳精度损失；只在同一设备范围相减，不进行跨卡时间比较。

```sh
python3 - <<'PY'
import hashlib, json, statistics
from decimal import Decimal as D
from pathlib import Path
p=Path('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl')
q=Path('agents/diagnosis/f08-attention-intervals.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(q.read_bytes()).hexdigest()=='355cd1a831403c1129ab76e128c0803da39b05712c3304c5c285aa82e6744e6c'
index={(x['ep'],x['scope_id']):x for x in json.loads(q.read_text())['indexer_attention_pairs']}
assert len(index)==256
rows=[]; seen=set()
for line in p.open():
    j=json.loads(line)
    if j.get('kind')!='rank': continue
    for scope in j['data']['layers']:
        key=scope['ep'],scope['layer_scope_id']
        if key not in index: continue
        assert key not in seen; seen.add(key)
        tasks=[t for group in scope['operator_groups'] for t in group['tasks']]
        by_flow={t['flow_id']:t for t in tasks}
        assert len(by_flow)==len(tasks)
        pair=index[key]; i=by_flow[pair['indexer_flow']]; a=by_flow[pair['attention_flow']]
        assert i['csv']['Type']=='VllmAscendQuantLightningIndexerV2' and a['csv']['Type']=='SparseFlashMla'
        start=D(i['device_start_us'])+D(i['trace_duration_us']); end=D(a['device_start_us'])
        matching=[t for t in tasks if t['csv']['Type']=='_prepare_indexer_indices_kernel'
                  and D(t['device_start_us'])<end and D(t['device_start_us'])+D(t['trace_duration_us'])>start]
        assert len(matching)==1; t=matching[0]
        t0=D(t['device_start_us']); t1=t0+D(t['trace_duration_us'])
        assert start<=t0<t1<=end
        assert i['device_track_pid']==t['device_track_pid']==a['device_track_pid']
        assert i['csv']['Stream ID']==t['csv']['Stream ID']==a['csv']['Stream ID']
        rows.append((end-start,t1-t0,t0-start,end-t1))
assert seen==set(index)
ratios=[100*t/g for g,t,_,_ in rows]
print('pairs',len(rows),'prepare_pct_min_median_max',min(ratios),statistics.median(ratios),max(ratios))
print('prepare_coverage_ge_90pct',sum(x>=90 for x in ratios),'ge_95pct',sum(x>=95 for x in ratios))
print('longest_gap_us,prepare_us,before_prepare_us,after_prepare_us',max(rows))
PY
```

## 最长间隔：已存设备与主机记录的边界

选定前述 EP2、第 14 层、scope `1113-65`。原采样 QLI→prepare→attention 三个设备起点在该 EP 的历史 SQLite `TASK` 表各精确命中一行；`COMPUTE_TASK_INFO` 名称符合目标，device ID、采样进程、stream 均一致。以 SQLite 整数纳秒计算 QLI 结束到 prepare 开始 **1555.189 µs**；与先前 trace 的 1555.18875 µs 相差 0.00025 µs，不视为两次测量。

- 查该历史 DB 中 **device ID 2、采样进程 1113、全部 stream**：半开区间内没有 TASK 起点。按 `startNs < prepare_start AND endNs > indexer_end` 查正相交记录有一条，仅向间隔跨入 0.960 µs；它与 QLI 同 stream、taskId、connectionId，但 globalTaskId 不同且无匹配计算信息。可能是不同导出视图，不能当成独立物理工作，也不能仅据此确定记录重复机制。查询排除其他进程和非本采样设备，TASK 未导出、内部同步、通信或采样以外的活动仍不可见，不能推断整卡空闲。
- 三个精确匹配设备任务的 QLI 和 prepare 分别通过自己的 `connectionId` 各匹配到一条 CANN `launch`。同一 globalTid 里，一条 `aclnnVllmAscendQuantLightningIndexerV2` CANN 记录在时间上完整包含 QLI launch，持续 **2102.520 µs**；launch 结束到外层 CANN 记录结束为 **2076.380 µs**，外层记录结束到 prepare 的 launch 开始为 **15.640 µs**。这些差值均来自同一个 CANN 表、同一线程，外层记录的时间包含关系并非对内部执行机制的证明。
- 同一 globalTid 的 `PYTORCH_API` 中，`Dequeue@aclnnVllmAscendQuantLightningIndexerV2` 记录持续 **2112.080 µs**，下一条 `Dequeue@_prepare_indexer_indices_kernel` 在其结束 **2.940 µs** 后开始；各自的时间范围分别包含上面相应 CANN 记录。它们是采集到的调用/出队区间，不等于 CPU 全程计算、设备同步、等待或某种确定根因。不能用这些主机时间与设备 TASK 时间直接相减；外层调用的精确因果关系、内部活动与 tensor/event 身份仍未绑定。

只读复算：`python3 agents/diagnosis/query_ep2_indexer_gap.py`。脚本在拥有的 stage 容器中以 SQLite `mode=ro`/`query_only` 读取该历史 EP2 profiler，运行前后核对容器 ID 和镜像 ID，且检查 DB SHA-256 `2f51a3a608eaf21fbadbb98d54e79032bc23638da76cd80344e9b5c363add7c9`。查询限定采样进程、设备和已关联线程，不读取其他任务的日志或运行负载。脚本 SHA-256 `01c46dc8072883db85271176981d1029c6f94ea4143d6dccef5ee37e9349b6c6`；输出 [ep2-indexer-gap-query.json](ep2-indexer-gap-query.json) SHA-256 `c66cea4873aa214b965490e393982b471e7fcae3185e3582745095673ab75fed`。该分析没有启动 NPU 或修改目标实现。

## EP2 最长间隔：同线程 CANN 子 API 和采集范围

进一步查询上节的 CANN 外层记录，在同一 `globalTid` 下按正区间相交选取全部 `CANN_API` 条目。除外层以外有五条，均完整位于外层：两条 `aclrtGetStreamAttribute`、一条 `aclrtGetHardwareSyncAddr`、一条 `launch`，以及位于该 launch 内部的 `aclrtLaunchKernelWithHostArgs`。5 条时长之和有嵌套，不能直接相加；以区间**并集**计，已记录的子 API 覆盖 **15.450 µs / 2102.520 µs = 0.7348%**，外层中其余 **2087.070 µs** 没有这些更细 CANN 子 API 的记录；最后一条子 API 结束到外层结束为 **2071.020 µs**。

该结果缩小了能够用“已记录子 API 耗时”描述的范围，但不能把其余部分定性为 CPU 计算、CPU 等待、设备同步或设备空闲。`CANN_API` 中无更细记录不表示这些工作没发生，也不证明调用栈。这里只比较 CANN 同一线程的时间区间；不将 CANN 与设备 TASK 时间相减，也不推广到基线或未采样请求。

该 EP 的 `profiler_info_2.json` 记录 activities 为 CPU 与 NPU、profile level 为 Level1、`with_stack=true`，同时 `_host_sys=[]`、`_sys_io=false`。这说明启用了相应框架 profile 选项，并不证明保存了能够判定当时线程运行/睡眠状态的操作系统调度轨迹。当前 SQLite 的表清单没有识别出 OS 调度采样表；未排查的其他采集资源是否存在可用系统轨迹仍未知。现有冻结 Indexer 接口源码有对 `aclnnVllmAscendQuantLightningIndexerV2` 的调用，但单靠名称无法将历史运行二进制或调用内部控制流绑定到源码，也不能利用源码猜测这段时间的原因。

复算 `python3 agents/diagnosis/query_ep2_indexer_host_children.py`，脚本只读 EP2 本任务 DB 和该文件的采样配置，校验容器及 DB 身份并检查外层/子项及嵌套关系。代码 SHA-256 `f63156e03b9849b87c148ec18a0158eab16edebf280b2aa48219e087f6c31fa7`；结果 [ep2-indexer-host-children.json](ep2-indexer-host-children.json) SHA-256 `ba97be21f3f5b2fa3b481bb1f62bdc2a1a6653e60270f41bdece4f96790f1066`；配置文件 SHA-256 `03694c891e64e7bc70e1ff86276c8c70521bf439a3044be32be008b4230cb867`。未启动 NPU 或改变目标实现。

## EP2 栈与线程状态：提交线程有栈，Dequeue 线程没有绑定栈

对上节 QLI TASK 的 CANN launch `connectionId=126633`，`CONNECTION_IDS` 给出 framework ID 244391；该 ID 在 `PYTORCH_API` 唯一匹配 `aclnnVllmAscendQuantLightningIndexerV2`，其 `callchainId=547589` 可在 `PYTORCH_CALLCHAINS` 解得 38 帧，包括 `indexer.py:275 select_projected` 和 `dsa_v41.py:585 _select_sparse_indices`。prepare TASK 的 launch 126640 同法匹配 framework ID 244392、`_prepare_indexer_indices_kernel`，其 `callchainId=547593` 有 41 帧，包括 `prepare_indexer_indices.py:123` 和 `indexer.py:291 select_projected`。这是现有 profiler 的关联关系，具体 exporter 版本与关联语义仍需保留；它加强了两个设备记录对应的提交调用位置，但不证明实际输入输出地址或运行二进制内容。

保存的 [indexer.py](../../sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py) 该条件路径先获得 `selected`，再将 `selected.squeeze(1)` 和位置传给整理 helper；这为 Indexer 结果→整理提供源码级条件依赖。当前 trace 没有逐张量 storage 身份，也没有历史源码哈希与运行模块的直接绑定，不能由形状、栈与时间顺序宣称该次实际 tensor 已验证。

上述框架 API 的 `globalTid` 为提交线程，**不是**上节 2.1 毫秒 CANN 外层和 Dequeue 所在的线程。该线程里两个相关 `Dequeue@...` 条目的 `callchainId` 均为 NULL；外层 CANN connection 126630 经 `CONNECTION_IDS` 得 ID 117246，但 `PYTORCH_API` 没有该 ID 的匹配。故提交 Python 栈不能当作 Dequeue 或 ACLNN 内部执行栈，也不能用来判断线程是在 CPU 上运行还是睡眠。

限定本次 EP2 profiler 文件夹的 **184 个文件名**，以 `sched|ftrace|perf.data|syscall|osrt|thread.state` 大小写不敏感检索，命中 0；这是有限的文件名筛查，未解码所有二进制内容，不证明 OS 线程采样不存在。结合上节 DB 表和 profile 配置，目前没有取得能覆盖该线程、该区间并区分运行/可运行/睡眠的 OS 调度事件。最小补充证据是可验证与此工作线程及时间窗口对齐的线程调度状态，若要解释具体等待还需相应调用栈或等待对象；现有资料不能判定原因。

复算 `python3 agents/diagnosis/query_ep2_indexer_callchain.py`；脚本 SHA-256 `819b576e2a40fed65de436a70e62ceedb0a23917bb0af25c09abb7808f2df59a`，结果 [ep2-indexer-callchain.json](ep2-indexer-callchain.json) SHA-256 `c3aa180a13d913f06da330b12704bf854bdf35249763ab9cc3845581212f516e`。所有关联与文件名检查只针对本任务 EP2 原采样目录，未采集新数据、未启动 NPU、未修改目标实现。

## 28 个同签名 scope 中 Indexer 到 Attention 片段的覆盖

在一次已有重型 profiler 采样中，选择 `qli-target-scope-distribution.json` 里 `all_signatures_equal_ref=true` 的 28 个 scope，即同一保存 digest `db852bff` 的 EP0–6、L24/28/32/36。按 `(ep, layer, scope_id)` 与 `target-layer-all-kernels-16rank.jsonl` 关联，并以每个 scope 保存的 flow ID、CSV 类型及 task ID 唯一定位 Indexer、prepare、SparseFlashMla 三条任务；复算时检查三者 track PID/TID 与 stream 相同，且按 Indexer → prepare → SparseFlashMla 顺序不重叠。

定义片段为 Indexer 开始到 SparseFlashMla 结束，使用同一 trace 的时间戳和持续时间计算。28 个片段时长的最小值、中位数、最大值分别为 **6182.516、6262.925、6852.983 微秒**；三个选中任务时长之和占各自片段的比例为 **99.947%、99.952%、99.961%**，其中 Indexer 自身占片段的比例为 **81.836%、82.264%、83.809%**。例如 EP3 L36 scope `2128-42` 的片段为 6192.860 微秒，Indexer、prepare、SparseFlashMla 分别为 5074.26、217.16、898.26 微秒，三者占 99.949%，Indexer 占 81.937%。这说明所选片段的大部分由这三个记录的任务区间覆盖，不能把片段耗时主要归于这三个记录之间未覆盖的间隔。

比例只描述选定同流片段，不是 NPU 利用率；任务区间内部仍可能包含等待，其他流也可能并行。事件关联尚未绑定实际 tensor 或 storage、实际运行二进制及任务内部阶段，不能据此判断数据依赖、融合收益或端到端关键路径。28 个 scope 来自同一次重型采样，不是 28 次独立请求，不能直接外推到无 profiler 性能或整体吞吐。

在本运行根目录执行 `python3 agents/diagnosis/check_qli_selected_span.py` 可只读复算；脚本 SHA-256 `6333f0d2df35361cffdaa7a0d3bbf8c51373d16d4527b80b38497d7998dfc6b6`。输入 `agents/diagnosis/qli-target-scope-distribution.json` SHA-256 `61aa739e43197d4c0725d81aea30624728de0df9c7b2ffea772d5de53b0ca804`，`diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`；脚本核对源哈希、完整关联与筛选条件，不访问设备。

## 两个调度摘要的 QLI 物理形状和时长配对

对同一次重 profiler 采样的 EP0–6 和层 24/28/32/36，分别在 `7ddf7ae9…` 与 `db852bff…` 两个摘要中按同 EP、同层选唯一关联的 Indexer，得到 **28 对**。前者首个输入 INT8 shape 为 `[595,32,128]`，后者为 `[1023,32,128]`；两档各条第二输入均为 INT8 `[17315,128,1,128]`，对应第五输入均为 FP16 `[17315,128,1]`。这两个相同的物理 shape 可描述所记录的缓冲容量，**不说明实际使用了相同页数、有效 K 长度或数据内容**。

两档并非仅首维不同：其他以 Q 行数为首维的 shape 也变化，相关 INT32 输入的记录 shape 从 `[5]`、`[4]`、`[4,272]` 变为 `[4]`、`[3]`、`[3,272]`；记录的候选输入分别为 `[595,1,2048]`、`[1023,1,2048]`。脚本核对每档全部 28 条这些物理 shape 和其余 dtype、format、核类型与 block 字段。既存调度摘要记录的两次 `scheduled_total` 分别是 4760 和 8180；摘要及同层 scope 的关联不证明每个 QLI 的真实有效行或 cache 工作量等于这些数量。

逐对计算 `1023 行记录时长 / 595 行记录时长`，最小/中位/最大为 **1.37983 / 1.55552 / 1.74052**；28/28 大于 1，24/28 小于物理首维比 `1023/595≈1.71933`，4/28 大于该比值。这排除了所选记录中“每对时长严格按物理首维同比变化”的描述；它不说明实际有效工作按行线性缩放，也不确定改变行数会造成上述时长变化。

两摘要的实际请求、已处理上下文、页表内容、有效候选、metadata 值、实际二进制和竞争状态均未绑定；层与 rank 来自同一次重采样，不能当独立受控试验。所记录的相同 cache shape 不排除活跃页不同，另有 metadata 容器形状变化和 scheduler 结构变化，不能归因于行数、缓存页或具体内部阶段，也不能换算线上收益。

CPU 只读复算 `python3 agents/diagnosis/compare_qli_query_rows_across_digests.py`；脚本 SHA-256 `59a76119a8c9be6d5907e5427fad8a1b7b3e5bff3f35e368547e3544a871973c`，校验源 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`、56 个 scope 唯一任务和物理字段。没有访问设备或修改目标实现。

## 同摘要 L1 与 L24：相同 query 形状不代表相同注意力工作

对同一个重 profiler 摘要 `db852bff…`，选择 EP0–6 各自第 1 层和第 24 层唯一匹配的 `SparseFlashMla` 设备记录，每卡先算两条时长比，再汇总七对。两层的第一输入和输出都是 BF16 `[1023,64,512]`，第一 paged 输入都是 BF16 `[17315,128,1,512]`；L24 还记录了第二个 BF16 `[17315,128,1,512]`、INT32 索引 `[1023,1,512]` 及其他 metadata，L1 这些位置未记录同样的输入。额外 paged 输入和索引的形状与压缩 KV 分支相容，但仅凭形状不能证明 tensor 身份或运行时分支。

七卡逐对的 L24/L1 trace 时长比最小/中位/最大为 **5.102 / 5.184 / 5.439**；CSV 的 AIV vector 显示比例，L1 范围 **0.503–0.542**，L24 为 **0.219–0.225**。这表明仅用同一个 query 物理形状不能把这两层记录视作相同工作或直接解释两项指标差异。额外输入的有效长度和页面、索引内容、mask、具体实现与 profiler 扰动均未知；不能将比值归因于行数、缓存或硬件效率，也不能把显示比例等同等待或整卡利用率。两层属于同一次采样，scheduler digest 相同不能证明同一 token、tensor 或等工作量，更不能外推线上吞吐。

在运行根目录执行 `python3 agents/diagnosis/compare_attention_l1_l24_same_digest.py`，只读核对输入 JSONL SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`、14 条唯一任务、所列形状及 dtype，并输出配对比值；脚本 SHA-256 `21884fceaefa1c11db4d0acc968c76ce72e1fbed36c9865fe742019e2d5b1d35`。不访问设备。
