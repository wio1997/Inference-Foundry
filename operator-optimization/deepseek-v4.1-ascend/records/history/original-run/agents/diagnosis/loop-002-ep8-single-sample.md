# 单样本内部阶段取证结论

范围：169 已保存数据库中的 EP8、model 49、stream 89、task 64，目标 TASK 开始于 1793152958875785664 ns；精确匹配的 `DispatchFFNCombine` TASK 记录时长为 2035.220 μs。原数据库 SHA-256、查询脚本哈希和字段结果见 [mc2-raw-sample-ep8.json](mc2-raw-sample-ep8.json)。首次查询遇到 PMU 行数超过预设上限而停止，第二次只按计数器名称作计数和最小／最大值分组，没有把多行求和。

**能确认：**这个 TASK 是被 profiler 记录为一个融合算子的设备时间区间；所查数据库有通信事件、设备 TASK、PMU 等表结构。独立冻结源码包含 GMM1、GMM2、DispatchAndCombine 和 flag 等待逻辑，说明实现具备这些阶段与同步机制，但没有记录它们在此 TASK 内各自的开始和结束。

**不能确认：**该单次调用内 dispatch、计算、combine、readiness 等待各用了多久，不能把 kernel 总时间或流水占比当作其中任何一个阶段的时长。通信事件表存在本身不证明有与此 TASK 内部阶段关联的记录。此次枚举只显示 `ENUM_MSTX_EVENT_TYPE`，其没有时间列；没有在查询结果中观察到可归属本 TASK 内部的 MSTX 时间标记。

`TASK_PMU_INFO` 通过 globalTaskId 查询得到 28 个名称，每个名称有 7 行；这些行的重复维度尚未辨认，也尚未证实全部属于同一次实际执行，不能把范围或合计解释成单次样本的阶段耗时，不能将 min/max 当做内部阶段对照。首次行数上限检查失败保留为事实，不等于没有数据。

此结论只针对一个 rank、一个 TASK 样本和这次查询覆盖的数据库表；不能推广为其他 15 个 rank、其他原始导出、或整个运行过程都没有内部阶段数据。下一步需弄清 `TASK_PMU_INFO.timestampNs` 与 graph replay 的键关系，并检查通信事件是否通过确切标识关联目标 TASK；时间重叠只能用于提出候选。
