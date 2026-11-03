# 16 rank 目标层 flow 中未匹配 kernel CSV 的事件

**结论：**对已保存的 `target-layer-all-kernels-16rank.jsonl` 内 16 rank 的 `operator_coverage` 逐项求和，39,160 条 `csv_missing` 全部有唯一对应的 trace X 名称：`EVENT RECORD` 22,783 条、`EVENT WAIT` 16,377 条。原提取器将在目标 layer host scope 内选中的 `torch_to_npu` start，要求全 trace 唯一 start、唯一 finish，并按 `(track pid, track tid, finish timestamp)` 精确匹配唯一 X；X 还必须有 Model Id、Task Id、Physic Stream Id，才用 `(start time, model, task, stream)` 搜索 `kernel_details.csv`。仅当 CSV 该键零行，计入 `csv_missing`。因此这些条目不是缺 trace X，也不是推断设备空闲，属于 trace 中可见的同步命名事件；它们也不能直接视为 AI Core 计算算子或额外计算时间。

这一结论的域只是**提取器已选中的事件**，不是该 workload 全部同步事件、所有流或所有设备任务的普查；被排除、漏采和 X 没有身份字段的事件不在域内。每个 `csv_missing` 有唯一 flow→X，但未额外证明该 X 只与唯一 flow 关联，也不能将 39,160 直接解释为不重复的物理任务数。保存的聚合只有按名称计数，逐条时间需要重新从原 trace 按提取器规则读取。等待的依赖对象、发生原因及内部实际空闲率仍未证实；流上等待的显示区间不可直接与其他流的区间相加。

## 关键原始样本与对照

对 EP8、层 14、scope `737-20`，核对原 trace 与同 rank 的 CSV 哈希，并要求三条样本每条全 trace 有唯一 start、唯一 finish 与唯一同 track/时间 X：

| trace X | 物理 stream / task | X 时长（μs） | CSV 精确键条数 | 同 model/task/stream CSV 条数 |
|---|---:|---:|---:|---:|
| EVENT RECORD | 47 / 44157 | 0.02 | 0 | 0 |
| EVENT WAIT | 47 / 44162 | 0.02 | 0 | 0 |
| RmsNorm（对照） | 47 / 44163 | 25.2395 | 1 | 1 |

EP8 CSV 共 38,439 条，`Name` 为 EVENT RECORD 和 EVENT WAIT 各为 0，RmsNorm 为 1,852。该样本支持此 rank 的缺失来自这些事件未出现在 kernel CSV，而不是简单时间精度偏差；不能据此证明 CSV 发布方在所有情况下都不会导出同步事件，也不能证明另外 15 rank 的底层导出机制。由 trace 重获这两类任务的 track、时间与身份字段可行；它不恢复缺失的 kernel CSV 的算力计数或 wait/record 的依赖边。先前同 scope 的完整已选 flow 时间线 `EP8-L14-L39-FLOW-INTERVALS-20261029.md` 展示部分 EVENT WAIT 实际可长于上述 0.02 μs 示例。

## 复算

- `python3 agents/diagnosis/summarize_missing_event_names.py > agents/diagnosis/csv-missing-event-names.json`，基于既有结果，不访问 NPU。该脚本校验 16 rank 的每个 rank 分类和与 `csv_missing` 一致，并记录输入 SHA。
- `python3 agents/diagnosis/run_ep8_missing_events_check.py`：只读已存在 stage 容器中的历史文件，执行前后核对容器、镜像及挂载，核对原始 trace/CSV SHA，输出 `ep8-missing-events-check.json`；不启动服务或新 NPU 负载。
- 所用提取器：`diagnostic-20261029/extract_target_layer_all_kernels.py`，固定输入哈希及样本字段见两个 JSON。
