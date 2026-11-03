# 已保留目标层中 MIX_AIC 任务的 AIC/AIV 字段配对

## 范围与方法

只读取已保存的 16 rank `target-layer-all-kernels-16rank.jsonl`，核对输入 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`。16 个 rank 的记录状态为 completed，提取数据包含 1,280 个不同 `(EP, digest, layer)` scope，每个 scope 均保留一条 `DispatchFFNCombine` 的 MIX_AIC 记录。本分析的 scope 是相关联的采样单位，涉及四个 digest；不能视作 1,280 次独立实验，也不等价于端到端任务。

在同一个 EP、digest、层 scope 内，分别对 **CSV 所报 `Accelerator Core=MIX_AIC`** 的原始算子名和 Type 分组。每组对 AIC 和 AIV 分别计算 `sum(对应 total_cycles × 对应 ratio) / sum(对应 total_cycles)`；只有该组每条任务的分母为正、ratio 为有限数且位于 0–1 才纳入。这样不把 AIC 与 AIV 混在一张榜单上。HcPre 与量化矩阵乘在部分 scope 中同名组有多条、形状可能不同，该聚合是**同一 scope 内的任务字段算术**，不是单一物理形状组。各项之间计算量、数据量、时长、任务数与内部通信均不同，跨算子比较不是等工作量或效率比较。

## 结果：分别比较 AIC MAC 与 AIV vector 的导出比例

表中的比较次数是满足指标条件的 scope；MC2 较低表示其周期加权显示比例严格小于同 scope 中另一个原始算子名及 Type 的对应聚合，不是性能更差的证明。比例原始值显示精度和历史导出器解释未完全确定。

| 同一 scope 的 MIX_AIC 比较对象 | AIC：MC2 较低的 scope | AIV：MC2 较低的 scope |
|---|---:|---:|
| HcPre | 1,213 / 1,280 | 1,280 / 1,280 |
| SparseFlashMla | 1,219 / 1,280 | 1,280 / 1,280 |
| QuantBatchMatmulV3（完整原名见 JSON） | 1,253 / 1,280 | 1,280 / 1,280 |
| VllmAscendQuantLightningIndexerV2 | 222 / 256 | 256 / 256 |

四个 digest 中对应 scope 数分别为 320、320、320、320；Indexer 每个 digest 各 64。表内所有配对均存在且分母和比例通过上述条件。比较的每一项没有相等值；其余次数均为 MC2 更高。比此前仅以 AIC 排序，新增证据是：在这批保留 MIX_AIC 配对中，MC2 的 **AIV vector 导出比例也全部较低**。这是进一步调查 MC2 内部执行的线索，不能将低比例归因于通信等待、调度、缓存或算子缺陷，也不表明整个设备空闲。

## block 字段的差异

记录的 `Block Num` / `Mix Block Num` 字段：MC2、HcPre、SparseFlashMla 和 Indexer 均为 24 / 48（分别 1,280、2,560、1,280、256 条），QuantBatchMatmulV3 为 24 / 24（2,816 条）。这两个字段的确切版本定义尚未核实，不能直接解释为实际参与的 AIC/AIV 物理核数量。尤其不能假设上述对象占用同样数量的 AIV 资源，或把这些 ratio 当作单位芯片资源的等分母效率；这进一步限制了跨算子排名的解释。

## 对“最低 NPU 利用率模块”的影响和缺口

- 这些值按任务而非模块全时窗计算；未覆盖全部设备工作，不能得出模型范围内或整卡最低模块。其他 AI_CORE、AI_VECTOR_CORE、AI_CPU、MIX_AIV 工作也不能与 MIX_AIC 的比例直接混排。即使两个任务均报 MIX_AIC，AIC 和 AIV 两个分母也分别独立，不能相加得出总利用率。
- 该提取选择目标层同线程 `torch_to_npu` flow，经唯一关联 CSV；非选中流、缺失/不可关联事件不在分母。scope 不是整个层/模块工作的完整证明；任务可能跨流重叠。选中情况、其他未归属和 zero/invalid 详情见 [RANKING-COVERAGE-REMAINING.md](RANKING-COVERAGE-REMAINING.md) 和 [RECAPTURE-ALL-KERNEL-RANKING.md](RECAPTURE-ALL-KERNEL-RANKING.md)。
- 两个 ratio 只分别代表各自采集导出的算术字段，并非两类核心所有计算操作的覆盖率；通信协同、纯搬运、非 MAC 或非 vector 工作和等待无法从 ratio 反推。需要与同版本采集/导出实现确认 numerator、denominator、显示精度及混合核口径，再结合覆盖完整的设备多流时间窗、源码调用归属、必要的硬件占用或等待指标才能有条件地比较模块。
- 在当前“只读既有数据、不启动新 NPU 负载”的条件下，优先查看已有采集是否包含同版本定义、可归属每个设备活动的完整任务流或可验证的多流计数器；找不到则明确最低模块和原因仍未知，不能用本表替代。若将来允许新的诊断采集，需先冻结对应时窗、统一设备/模块资源分母并记录丢失率与干扰，再考虑执行；本结论不预设新增采集授权。

## 复算

在 runs 目录执行 `python3 agents/diagnosis/compare_mix_aic_in_scope.py`，仅 CPU 读取已有提取文件，生成 [mix-aic-in-scope-comparison.json](mix-aic-in-scope-comparison.json)，包括四个 digest 的分组明细。没有启动服务或 NPU，也没有修改目标实现。
