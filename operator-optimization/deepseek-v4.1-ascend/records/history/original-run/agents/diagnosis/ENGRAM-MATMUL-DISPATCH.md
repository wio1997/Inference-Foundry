# 两个 Engram 候选矩阵任务的下发关联

## 已确认到哪一层

对 EP8、0→1 和 13→14 两处、三轮采样共六条 BF16 `[3,6144] × [25600,6144]^T` 矩阵任务，以 model、stream、精确开始和结束时间及 op type 在 profiler 数据库中各唯一定位到一条 `TASK`。`COMPUTE_TASK_INFO` 记录名称 `aclnnMatmul_MatMulCommon_MatMulV2`，输入为 BF16 `[3,6144]` 和 `[25600,6144]`，输出 BF16 `[3,25600]`，属性记录 `transpose_x2:1`。这比只靠 CSV 类型名更具体，但仍没有找到对应实际权重地址或 tensor identity，不能把它严格定名为 Engram `wkv`。

两处任务的 globalTaskId 分别为 18536 和 19042；每个 ID 在同一个 model 49 / stream 89 上各有 7 条不同时间的 `TASK` 记录，覆盖目标三次采样。这说明 **globalTaskId 单独不能识别单次执行**；其复用与图重放一致，但仅凭这些行不能证明全部重放语义。六条都使用 connectionId 184330，该 ID 在数据库中被 19971 条 TASK 共享，因此也不能用于唯一定位一次矩阵调用。

按 184330 直接查询 `CANN_API`、`CONNECTION_IDS`、`PYTORCH_API` 没有命中，数据库中也未发现名称含 callchain/stack/frame 的表。本次没有精确 Python 调用栈或逐张量连接；**没有精确命中不等于没有 host 下发**，图重放可能将操作收进图，或相应事件未采集。

## 能关联到图级提交，但不是单算子调用栈

再复用 `model-execute-wait-chain-v2/calls.json`：每条目标矩阵任务均落在 EP8 同一 model 49 的唯一一个 `MODEL_WAIT_COMPLETE` 区间之内。每轮两条任务对应同一个候选区间；三轮分别为已保存的 graph ordinal 1、6、7。已有记录以该上层区间的 connection ID 连接到 `aclmdlRIExecuteAsync` 和 `MODEL_EXECUTE`；这是现有证据支持的**图级关联**。

任务本身 connection ID 184330 与上层图提交的 connection ID 不同；该判断包含同设备 profiler 区间包围、同 model 和此前 exact connection join，尚未通过运行时 model handle 或逐 tensor 身份独立核实。不能据此声称哪个 Python 函数直接下发了矩阵，也不能将 `MODEL_WAIT_COMPLETE` 的名称或整个区间理解为纯同步等待。没有进行 host 与 device 时间戳相减，或跨 rank 绝对时间比较。

## 复核与待查

`engram-matmul-task-links.json` 和 `engram-graph-enclosure.json` 分别保存逐样本 TASK/来源与包围候选；脚本、冻结窗口、DB 和 CSV 的哈希均记录在结果中。ZCode `operator-ceiling169-v2-114-engram-task-links` 与 `operator-ceiling169-v2-115-engram-graph` 均 exit 0，`model_verified=true`，模型 `deepseek-flash`。图级结果 SHA-256 `74344c0665c1916aef0dc2acfab1d24af431bc85498599e2299be1ce5e07173b`。查询只读已有数据，没有执行设备任务。

## 六条任务的 CSV 和 PMU

ZCode stage `operator-ceiling169-v2-116-engram-metrics` exit 0、执行模型 `deepseek-flash` 且 `model_verified=true`。`engram-matmul-metrics.json` 的 SHA-256 为 `a63b984daa873bbce474529bd4b66655f9cc40104987e77f5f9eb017d485defa`；脚本和输入窗口、间隔文件的哈希与本机一致，数据库及 EP8 CSV 的哈希与先前冻结记录一致。仅本 rank 的六条样本：

| 图采样轮次 | 边界 | 任务总时长 μs | `aic_mac_ratio` | `aic_mte2_ratio` |
|---|---|---:|---:|---:|
| 1 | 0→1 | 280.574 | 0.117 | 0.984 |
| 1 | 13→14 | 264.455 | 0.125 | 0.983 |
| 2 | 0→1 | 287.454 | 0.117 | 0.985 |
| 2 | 13→14 | 269.074 | 0.123 | 0.983 |
| 3 | 0→1 | 281.554 | 0.117 | 0.984 |
| 3 | 13→14 | 271.015 | 0.122 | 0.983 |

这些字段来自同一条经 model、stream、精确开始/结束、dtype、shape 筛选的 CSV 行。已保存的 [Ascend MindStudio Insight 字段说明](https://github.com/Ascend/msinsight/blob/efb28234772093b0cabc3c538e71375c957540b2/docs/zh/user_guide/system_tuning.md)将 MAC ratio 和 MTE2 ratio 描述为相应单元时间与总时间之比，并将高 MTE2 视为访存瓶颈的排查线索；本地快照 SHA 与证据登记一致。**本样本显示较高的 MTE2 指标，是优先排查数据搬运路径的依据，不单独证明 HBM 带宽已饱和、权重一定从 HBM 读取或访存就是确定根因。**不同流水活动可重叠，不能把它们加和，亦不能用 `1−MAC ratio` 算空闲或等待；这不是整卡利用率。

CSV `Wait Time` 为约 0.7–1.01 μs，但该字段在本次 exporter、图重放与任务边界中的精确含义尚未核验，不能视为 kernel 内等待时间。`Block Num=20` 是记录的任务切分参数，不能只凭它断言整卡利用率。现场缺任务字节数、数据来源、带宽与 cache 命中证据，异步硬件同步是否影响该算子也未知。

原始数据库中两个 globalTaskId 各查询到 7 个 PMU 时间戳组，符合任务复用的数量；对六条目标任务，最近组分别比 TASK 结束早 120–220 ns，每条在预设的 1 μs 搜索窗内仅有一组。选出的候选组的 `aic_total_cycles` 与对应 CSV 数值完全一致，MAC/MTE 比率与 CSV 的显示精度一致。这增强了它们是同次样本的可能性，但 profiler 对 `TASK_PMU_INFO.timestampNs`、重放键与允许误差的写入规则尚未证实，**仍只标为候选映射**；不能将 PMU 时间差当成下发、等待或同步耗时。

## 16 rank 一致性

ZCode stage `operator-ceiling169-v2-117-16rank-engram-metrics` exit 0、模型 `deepseek-flash` 且 `model_verified=true`。按冻结的 model、stream、精确起止、BF16 dtype 与 `[3,6144] × [25600,6144]^T` 输入形状匹配，**96/96 条样本找到唯一 CSV 行，未报告缺失/异常字段**：16 rank × 3 轮 × 2 个边界。16 份 CSV 哈希与冻结清单一致；窗口、区间、脚本和 manifest 哈希复核通过，六条 EP8 结果与前述独立提取逐字段一致。结果 `16rank-engram-matmul-metrics.json` 的 SHA-256 为 `35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d`。

这些任务的 CSV `aic_mac_ratio` 在 0.114–0.129，`aic_mte2_ratio` 在 0.981–0.986；每个边界与轮次的 16 rank 均有记录，详细分布见 JSON。这表明上述指标不是 EP8 单卡的偶然值，仍仅覆盖这 96 个已经选定的算子样本；采样可能改变时序，也不能推广到其他形状、阶段或整卡。

同一批行的 `cube_utilization(%)` 为 78.334%–82.253%，不能把它和 11.4%–12.9% 的 `aic_mac_ratio` 当作同一个指标。现有 `engram-cube-mac-formula-check.json` 对 EP8 六条样本按另一版 CANN 文档的候选公式复算，与导出数字的显示精度一致；这为二者统计口径不同提供线索，但尚未证明当前实际 exporter 使用了完全相同的版本与计算规则，也不等于物理算力峰值利用率。

## 下一步区分原因所需的证据

- **HBM、L2 或重复搬运**：关联同一任务与已有采集中的 HBM/L2 读字节、缓存命中或失败、MTE 实际传输字节及资源带宽数据，查明计数器的范围、复用和硬件文档定义；只看到 MTE2 高不能决定数据来自哪一级。若已保存采集没有这些字段，应标 UNKNOWN，而不是补测或猜数。
- **小 M / tile 资源使用**：核对具体 kernel 编译/tiling 元数据、有效矩阵行数、padding、每 block 的有效处理量和硬件指令统计；`Block Num=20`、shape 中的 3 以及低 MAC ratio 单独不能证明是哪种利用不足。
- **同步或排队**：关联 graph replay 的明确事件、流等待或 task 内可识别的同步区间；已知图级 parent 与相邻时间不够，MTE2/MAC 指标不能拆出同步时间。`Wait Time` 不能直接当内核内部等待。

## stage125：现有字段是否能分辨 HBM / L2 / tiling

在 EP8 的六条目标 MatMul 中，逐条按 model、stream、完整时间、形状匹配原始 CSV，并完整读取两个 `globalTaskId` 的所有已导出 PMU 名称，而不是只看之前选取的七个名称。DB、CSV 和目标文件均经冻结 SHA 校验。

- 六条 CSV 的 44 列中，`Context ID` 均为 `N/A`；其他列非空不表示所有字段口径已知。两条复用的 globalTaskId 各有 28 个 PMU 名称、7 个记录时间戳、共 196 条记录。重复 ID 的 PMU 只能按时间邻近作为候选，不能当成已证明的逐次重放关联。
- 新核到的 `aic_icache_miss_rate` 在六个 CSV 样本中均为 `0.002`（导出精度）；对应候选 PMU 的原值约为 `0.00181–0.00189`。它是以 icache 命名的指标，**不是数据 L2/HBM 命中率**；其本次版本的具体分母、计数范围和对任务耗时的贡献仍须核对，不能据此排除所有指令侧问题。
- 本次完整读取的目标 PMU 名称和 CSV 列中，没有明确的 HBM/L2 读写字节、数据缓存命中/失效、任务内部等待时长或 tile M/N/K 统计。EP8 数据库存在 `OVERLAP_ANALYSIS`，其表结构为 device 时间与类型，没有任务 ID 或 HBM/L2 流量字段，不能由此把流量或等待归于这六次 MatMul。
- `COMPUTE_TASK_INFO` 记录 `blockNum=20`、`gridDim=5`、`blockDim=5` 和算子属性；后两者的编码和语义尚未核实，不能把原始数字 5 当作实际 tile 维度。属性提供转置、输入输出 shape/dtype 等信息，不提供可还原的实际分块、padding 或读写次数。

结论是当前已检查的六条 CSV 与这两个任务的全部 DB PMU 仍无法区分数据从 HBM 还是 L2 搬运、内部同步等待、分块利用不足或资源争用。这个结论不覆盖 DB 之外的 profiler 原始文件、二进制产物及其他未检查表/版本，不能据此宣称所有采集都缺少相关指标，也不能推算可实现提速。

复核数据：[engram-all-metrics-fields.json](engram-all-metrics-fields.json)，SHA-256 `d55871db62f75b97d18b1e4912b153a5602eff776bd13488030ebc6d5ec63bd9`；查询脚本 [query_engram_all_metrics_fields.py](query_engram_all_metrics_fields.py) SHA-256 `2f06fa9b8534d5ff0e53feb9ffd89ab4955980e631137e8ccc0c3c87773543d3`。本机 ZCode 的 `operator-ceiling169-v2-125-engram-all-fields` 退出码为 0、`model_verified=true`、模型为 `deepseek-flash`；仅在已有 CPU staging 容器读取文件，无新 NPU 负载。
- **采集和归属**：确认采集设置、指标定义及 profiler 对事件重放的对应键，并尽量找到该矩阵权重/tensor身份或调用栈。不同 rank / capture 的一致性只是支持线索，不能把来源未证的任务写成已确认 Engram。

## 条件性计算对照：96 个 BF16 小 M 矩阵任务

复用 stage117 的同一 CSV 行和同一次任务的 `Duration(us)`：96 条覆盖 16 rank、3 个采样 ordinal、两处边界，其输入/输出形状均为 `[3,6144]`、`[25600,6144]` 和 `[3,25600]`，dtype 均为 BF16。EP8 的六条另经 DB 属性独立证实 `transpose_x2=1`；其余 90 条是形状、dtype 和算子类型的匹配，不能声称逐条核实了该属性。按 `[3,6144] × [25600,6144]^T` 计算直接稠密逻辑工作为 `3×6144×25600=471,859,200 MAC`。该批单卡任务时长最小/中位/最大为 **257.015 / 275.765 / 293.474 μs**。采用 1 MAC=2 ops，用逻辑工作分别除以最大/中位/最小时长，得到等效吞吐约 **3.216 / 3.422 / 3.672 TOPS**；中间数是**按时长中位数折算**，不是先求逐条吞吐后再取中位数。两种统计在本批以三位小数表示同为 3.422 TOPS。这只是规范化的逻辑形状和计时对照，不是 profiler 测得的实际 BF16 指令数；不能把 rank 同 ordinal 合起来当同步 collective 或业务吞吐。

下面用**假设的理想计算模型**计算参考值，不是 169 已证硬峰值：假设各设备有 24 个可并发 Cube、硬最高频率 1.8 GHz、每 Cube 每周期可持续 4096 BF16 MAC、无其他执行资源能绕过这个限制、全部逻辑 MAC 必须在本次局部窗口内执行。它们对应 `P_assumed=176.9472 TMAC/s = 353.8944 TOPS`，理想全有效计算时间 `T_ideal=471,859,200/P_assumed=2.667 μs`。按同一局部任务时长定义，逻辑等效吞吐与假设峰值之比为 `T_ideal/T_observed`：按最大/中位/最小时长分别折算约 **0.909% / 0.967% / 1.038%**。方向是 `观察到的等效吞吐 ÷ 假设峰值`，不是峰值除以观测、也不是实测硬件利用率。

再限定每条 BF16 Mmad 只能处理原始 M=3 行、不能拼入其他有效行，N/K 可分块；每条遵循已存 CANN 文档的 M 对齐模型、`Δt≥0`，且同一 Cube 上各条指令模型周期作为不可重叠的工作量可加，全部计算必须采用这个 Cube 路径，则累计理想 Cube 工作至少 `614,400` 模型周期。假设所有 24 Cube 在整个窗口都可用且频率持续 1.8 GHz，可写 `T_tile≥614400/(24×1.8×10^9)=14.222 μs`；对 275.765 μs 中位样本，`T_tile/T_observed≈5.157%`。该比值是受限模型里的**有效逻辑工作速率 / 模型容量**，不是整体 NPU 利用率，也不证明 14.222 μs 可以达到。模型的模型周期和对齐、跨核心并行的条件及样本未确认的 tile/接口详见 `agents/bounds/hardware_sources_audit.md` 的“2201 单指令公式与受限分块推导”；一条处理整个矩阵的 Mmad 不满足 L0 容量，应分块，不能当实际指令。

**不能推断的内容。** 24 AIC 和 1.8 GHz 是 SDK/profiler 字段，并未证实全部合法状态硬最高频；厂商表给出 BF16 Mmad 的并行度，但未证明 4096 MAC/周期是 9382 的全路径硬容量。历史 kernel 的真实 tiling、padding、weight identity、mask 和计算路径均未闭合；允许跨行拼接、融合、缓存、重算或其他计算单元时，M=3 受限模型不覆盖所有算法。算子可能仍需访存、量化、其他同步等耗时，也可能与其他任务重叠，因此不能把 `0.967%`、`5.157%` 解释成可优化余量、可达加速倍数或全引擎 TPS。此前相同行的 `aic_mac_ratio≈11.4%–12.9%` 与 `cube_utilization≈78%–82%` 是不同的导出统计口径，不与上面的比值互换；没有获取该任务实际 HBM 字节或硬带宽，不能从权重逻辑大小推 HBM 利用率。

复算：

```sh
python3 - <<'PY'
import json, statistics
p='runs/169-vllm-v41/agents/diagnosis/16rank-engram-matmul-metrics.json'
s=json.load(open(p))['samples']
t=[float(x['csv_fields']['Duration(us)']) for x in s]
f=3*6144*25600
p_assumed=24*1800e6*4096
print('n, time us min/median/max:', len(t), min(t), statistics.median(t), max(t))
print('MAC, assumed TMAC/s, ideal us:', f, p_assumed/1e12, f/p_assumed*1e6)
print('logical TOPS min/median/max:', *(2*f/x/1e6 for x in (max(t),statistics.median(t),min(t))))
print('tile-limited model us and median ratio %:',614400/(24*1800e6)*1e6,100*614400/(24*1800e6)*1e6/statistics.median(t))
PY
```

该文件 SHA-256 `35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d`；两个模型是对既存资料的条件算术，复算不访问设备、无 NPU 操作。

复核备注：本机重新核对 96 个 `(rank, run, after_layer)` 唯一，96 条 CSV 的 Name、Type、输入/输出 BF16 shape 均一致；排序后中央两次时长为 275.755 和 275.775 μs。逐条折算后吞吐中位为约 `3.4221833853 TOPS`，由中位时长折算为约 `3.4221833808 TOPS`，差约 `4.5×10⁻⁹ TOPS`。14.222 μs 依赖同核模型周期独占可加、全部有效 MAC 不重不漏和 24 核频率上限，单条文档公式不足以证明上述假设。stage125 已查的六个 EP8 目标 CSV 与全部导出 PMU 字段未含可绑定 tiling、缓存字节或硬峰值；`Block Num=20` 的含义也未闭合，当前没有新增材料可以缩小这些缺口。此复核仅重读已存文件，不执行 NPU。

## 同卡两处候选 BF16 矩阵的成对时长和导出字段

复用 stage117 保存的 96 个历史任务，以相同 `(rank, run)` 配对目标层 0→1 与 13→14 两个边界，共 **16 rank × 3 轮 = 48 对**。逐条核对两个任务均在同卡 model49 的同一主流、各自已归属层间窗口内，导出的 `MatMulV2` 类型、输入 BF16 `[3,6144];[25600,6144]` 和输出 BF16 `[3,25600]` 一致，且 CSV 时间/时长与已存纳秒端点一致。**两处任务的权重和输入身份尚未绑定**，只有 EP8 六条在既有 stage114 的 DB 中另核对了 `transpose_x2=1`，不能推为其余 90 条均已查到相同属性。

三轮每轮 16/16 对中，0→1 的该矩阵任务均比 13→14 更长。逐卡先作时长差，再按轮取中位数和范围（μs）：

| 轮次 | 0→1 减 13→14 的逐卡时长差 | 前者耗时更长的对数 |
|---:|---:|---:|
| 1 | 18.569（13.659–24.979） | 16/16 |
| 2 | 19.240（11.740–28.200） | 16/16 |
| 3 | 15.690（8.680–21.700） | 16/16 |

这一配对补充了原报告按边界各自汇总时长的证据：不能把两个独立组中位数相减来当作逐卡差；在本批样本中，差值方向不由少数异常卡造成。48/48 对中前者的 `aic_total_cycles` 导出值也更大、`aic_mac_ratio` 更小；`aic_mte2_ratio` 三轮中位数分别为前者 **0.985/0.985/0.985**、后者 **0.983/0.984/0.984**，这些小数是有限精度导出值。`cube_utilization(%)` 的逐卡方向并不一致，三个轮次前者较高分别为 **11/16、5/16、8/16**。这些是可复核的字段差异，不是对物理运算、访存字节或资源空闲的直接测量；指标间存在可能的统计关系，不能独立相加。

这些对照排除的是“本次已选 48 对的两处任务时长完全相同或方向由极少数样本决定”，并不证明不同权重或内容下的有效工作量一致，更不能区分缓存/数据位置、tiling、运行状态、内核路径或采集扰动。高 MTE2 ratio 的含义和 replay 归属仍受上文约束，不能以其接近就断言同样带宽，更不能把时长差称为必然节约。每个差值都先在同一 rank 各自求两个任务持续时间后计算，**未比较跨卡绝对时间**；ordinal 也不证明这些卡对应同一请求。

复算：`python3 agents/diagnosis/recheck_engram_pairwise.py`。脚本校验冻结窗口、CSV 哈希清单和 stage117 汇总 SHA，检查每个 `(rank,run)` 两个目标、边界、流、类型、形状、dtype 与时间，再输出逐对值和分轮统计；输出 `engram-pairwise-existing.json` SHA-256 `0487234fa2637a5c1c6add39bb54960b865f11b25644dedc9c3792fb15be9efc`，脚本 SHA-256 `3d5165d1b214d14c24f3e16b685042d7cdde902314c02ea6d579e6aaecf466ab`。该脚本只读取本机已存数据，没有新 NPU 负载。

该离线复算由本机 ZCode stage238 执行，exit 0、model_verified=true，标准输出 SHA-256 `0487234fa2637a5c1c6add39bb54960b865f11b25644dedc9c3792fb15be9efc` 与保存结果一致；无 NPU 操作。

### EP8 两个候选任务的已导出 compute 元数据对照

为进一步核对两处候选差异，复用 stage114 的 EP8 六条精确 TASK/COMPUTE 记录，并按同 rank、run、边界、model、stream 及完整起止时间与 stage117 的六条 CSV 候选逐条关联。三轮中，每轮的 0→1 与 13→14 任务对应不同的 `globalTaskId` 18536 和 19042；**排除这个用于标识不同记录的字段后，所保存的其余 16 个 `COMPUTE_TASK_INFO` 原字段全部一致**，9 个已解码字段（包括完整属性字符串）也一致。已解码两边的名称均为 `aclnnMatmul_MatMulCommon_MatMulV2`，ND 格式、BF16 和形状相同，转置属性均显示 `transpose_x1:0|transpose_x2:1`，`blockNum=20`；这些属性先前已分别记录，本次新增的是对同轮两边**整组已保存字段的一致性**检查。

因此，对于 EP8 这三对，不能把**所查这些导出 compute 元数据的不同**当作解释；这不能排除实际权重、输入 buffer/stride、数据位置、真实 tiling、编译二进制、有效工作或缓存状态不同。属性字符串相同也不能证明各次运行参数的动态内容相同。两种 `globalTaskId` 均复用，关联的 compute 记录不提供独立 replay 的 runtime 对象或数据身份；此项检查未覆盖另外 15 张卡，更不是完整 kernel/硬件状态检查。

既有证据仍不能把耗时差归因于某一机制：stage120/123 对第二、三轮图前发现的 host-UVA gather 已提供下发线索，但没有把这些 gather 的输出 buffer、版本、ready event 与两条矩阵输入或请求一一绑定；stage125 的六条 CSV/相关 DB 已导出 PMU 名称中没有明确的 HBM/L2 读写字节、数据缓存命中、内部等待或 tile M/N/K。图级 `MODEL_WAIT_COMPLETE` 包围只是图级关联，不能据此量化单个任务的调度或等待。现有资料不允许区分这些解释，也不证明其他未检索产物中不存在更详细数据。

复算 `python3 agents/diagnosis/recheck_engram_ep8_compute_pair.py`，脚本校验 stage114 和 stage117 输入 SHA，逐一核验六个匹配事件和成对字段，不输出属性原文；结果 `engram-ep8-compute-pair.json` SHA-256 `f78c402a752e6cac17c11416f1df808b9f621bba4ae675eb3068ad9d397b9351`，脚本 SHA-256 `f34fc5f8ea539634f802900a733197401fe0ed23c54c666937b9d8d10770ee83`。仅本机读取已有数据，无新 NPU 工作。 ZCode stage239 exit 0、model_verified=true，输出 SHA-256 `f78c402a752e6cac17c11416f1df808b9f621bba4ae675eb3068ad9d397b9351` 与保存结果一致。

## 同卡配对的 MAC 周期乘积：条件性显示精度区间

上节 48 对中，前者的 `aic_total_cycles` 均较大、`aic_mac_ratio` 均较小，这两个方向不能单独推出对应 MAC 周期的大小顺序。对每对以导出值作条件乘积 `c₀r₀−c₁r₁`，并**假设** ratio 是 MAC 周期除以 total cycles、显示值按最近值舍入到 0.001、导出的 total cycles 没有截断且两者均正确对应同次 replay，则取最保守的舍入误差，差值落在中心值加减 `0.0005×(c₀+c₁)` 的区间。

在这个条件模型下，三个轮次各 **16/16 对**的区间都包含零；仅看显示值乘积的中心差，前者大于后者分别为 **11/16、7/16、6/16**。48 对的最大绝对中心差是 **9194.184 cycles**，各对 `|中心差| / 区间半宽` 的最大值约 **0.94855**。这表示假设的显示舍入误差足以覆盖每对差值，**不能据此判断哪边的真实 MAC 周期更大，也不能证明真实 MAC 周期相等**。ratio 的实际计算和显示精度/舍入方式、cycles 导出的聚合及截断、replay 的同事件绑定均未得到本次 exporter 版本的充分验证；这些条件不成立时区间须重新推导。结果也不证明两矩阵的数据、工作量、硬件指令或原因相同，更不构成性能界或可兑现收益。

复算 `python3 agents/diagnosis/recheck_engram_pairwise_mac_rounding.py`：脚本只读已有配对文件、核对 SHA 和 48 对唯一性、使用 Decimal，输出逐对区间与统计。输入 `engram-pairwise-existing.json` SHA-256 `0487234fa2637a5c1c6add39bb54960b865f11b25644dedc9c3792fb15be9efc`；脚本 SHA-256 `efd5fbb852dc3890f87a176aff44ec16d4f15b01cc0256b7712ba02c00ff1108`，结果 `engram-pairwise-mac-rounding.json` SHA-256 `7ab7182245ff2e885bd9f001d1e215b83048e00bcfb65102ad3cbbcbe526acb8`。本机 ZCode stage240 `deepseek-flash` exit 0、`model_verified=true`，标准输出 SHA 与保存结果一致。没有启动 NPU 或修改推理实现。

## EP8 固定事件的原始 trace 字段复核（stage259）

对第 2、3 轮各两条 host-UVA gather 与两条候选 MatMul，使用 `X`、精确纳秒起点、pid、tid、model、task 和名称 SHA 定位：8 条均唯一命中。每条原始 args 恰有 7 个键：Model Id、Task Type、Physic Stream Id、Task Id、Batch Id、Subtask Id、connection_id；没有地址、tensor、storage 或 buffer 身份字段，键列表未截断。这检查了原始 args，而不只是 stage121 的白名单摘录；不覆盖其他事件、字段或文件，也不能据此推断运行时没有依赖。Task ID、connection ID 的复用和图级关联都不能替代 tensor/版本绑定。

只读脚本 `query_engram_exact_trace_identity_keys.py` SHA-256 `d94f311ca19fecdd4eb9e8cc98fe708b0397d4fef98583c793699b51a68808e6`，wrapper SHA-256 `8afed4031bea281a6710540a4bfb3c7bca6af7548dd50d5b168b503c23ff19bd`，结果 `engram-exact-trace-identity-keys.json` SHA-256 `180a2c9489cbab81e56fc46f34f79148c8e15c871781de0064c9518b701d4bcf`。输入 trace SHA `3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99` 与 stage121 首次观察值一致，但不是采集时独立冻结值；读取前后核验 SHA 和状态。查询仅返回白名单键名、其他键名及疑似标识的临时盐 HMAC，未输出原始地址。ZCode stage259 exit 0、model_verified=true、deepseek-flash；没有新 NPU 负载。
