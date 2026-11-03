# 全算子排名：剩余覆盖缺口复核

只基于固定 SHA 的已提取 JSONL 和已验证的排名汇总，不启动 NPU。`device_x_missing_name_or_task_identity` 共 3,840：其有名称的事件全部匹配 `hcom_allGather`、`hcom_alltoall`、`hcom_reduceScatter` 三类，每类 1,280；每个 EP 各有这三类 80 条。提取器在这一步只记录“名称或身份字段缺失”的合并原因，所以这份数据无法区分缺少哪一个字段，也无法据此断言 X 的反向唯一性。这些计数是选中的 flow 对应记录，不是 3,840 个未知计算算子或物理通信任务数；仅凭名称亦不能解释通信内部或等待原因。

加上此前 `csv_missing` 已证实全为同步命名事件，先前两个最显眼排除项都不能作为“计算算子数量缺失”的计数。**但计算覆盖仍未证明完整**：提取器只选取目标层同线程内、匹配 `torch_to_npu` 的开始事件；无该 flow、未落在选中范围或 trace 未采集的设备工作不在分母内。两条 `flow_start_nonunique` 统计单位为 flow ID，选中总数 96,922 为 start 事件（对应 96,920 个 ID）；不能将不同实体的计数简单相减称为设备覆盖率。53,918 是通过反向排重及其他校验的接受任务记录。

排名还有独立的分组和指标缺口：6,862 个物理签名组中 520 缺 shape/dtype/format，1,412 缺一个或以上 EP 同签名，4,930 通过 8 EP 同签名等数条件；它们是**组数，不是被漏采的算子任务数**。在后者中 AIC 1,596 组合格、3,334 组因任一任务分母非正/字段无效排除；AIV 3,966 组合格、964 组排除。零周期不代表观测到无计算，特别不能把所有核类型一概视为本应有正 AIC 分母。缺口不构成整体 NPU 利用率的已知置信区间，不可用来估算全模型最低值或整体性能损失。

复算：在本 runs 目录执行 `python3 agents/diagnosis/audit_ranking_coverage_remaining.py`；脚本核验 JSONL 和排名汇总 SHA，输出 [ranking-coverage-remaining.json](ranking-coverage-remaining.json)，附每 EP 明细。同步项分类见 [CSV-MISSING-SYNC-COVERAGE.md](CSV-MISSING-SYNC-COVERAGE.md)，提取规则见 `diagnostic-20261029/extract_target_layer_all_kernels.py`；原排名见 [RECAPTURE-ALL-KERNEL-RANKING.md](RECAPTURE-ALL-KERNEL-RANKING.md)。

对满足八卡物理签名完整且等数的组中 43,184 条已匹配任务记录，进一步按 CSV 所报 `Accelerator Core` 复核，AIC 的零周期 28,688 条全部来自 `AI_VECTOR_CORE`（27,728）或 `MIX_AIV`（960）；AIC 正周期 14,176 条来自 `AI_CORE`（7,520）和 `MIX_AIC`（6,656）。AIC 缺失/无效 320 条来自 `AI_CPU`。AIV 的零周期 7,520 条全部来自 `AI_CORE`，AIV 缺失/无效 320 条来自 `AI_CPU`。因此这些分母排除与所报核心类型明显对应，不能统称为“硬件未工作”或“丢失的计算记录”。这些是任务记录级对应关系，不证明设备内核资源实际使用情况，也不能把 AIC 和 AIV 值相加代表整体利用率；核心类型及计数的版本语义仍需结合工具链确定。
