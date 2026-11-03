# EP8 L14：关联通信子任务和 notify ID 的限定范围

以已保存 EP8 采集的 `COMMUNICATION_OP` 中 `opId=209 / connectionId=98554` 为范围，固定输入 DB SHA-256 为 `e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486`。该 opId 在此库仅匹配一条 COMMUNICATION_OP，关联 168 条 COMMUNICATION_TASK_INFO；168 个 `globalTaskId` 互不重复，每个均在 TASK 中恰好匹配 1 行，且所有通信子任务的 `timestampNs` 与对应 TASK 的 `startNs` 相同。这些是该导出库内的字段关联和唯一性，不自动证明采集完整或跨版本字段契约。

这 168 条 TASK 的导出区间中，**没有一条与已选 L14 MC2 窗口相交**；最早起点比 MC2 结束晚 226.807529 ms（同库坐标，跨轨校准仍未知）。前一报告的 95.231037 ms 交叠对象是另一个 `connectionId=98566` 的 AI_CPU TASK，并非这 168 条中任一条；两条 connection 共连同一个 host 调用，但尚不能证明它们是同一物理通信事务或证明通信所有阶段均未与 MC2 交叠。不能根据本范围否定其他通信活动、资源争用或因果。

`COMMUNICATION_TASK_INFO` 里 opId 209 的非零 `notifyId` 在同库其他 opId 中出现：例如 `2084` 关联 278 个不同 opId，其中其他 opId 下有 3937 条行；`-4294704740` 关联 278 个不同 opId，其中其他 opId 下有 277 条行。这表明本库中的相等数值不足以唯一标识一次操作或直接证明 record/wait 对应，更不能单靠相同 ID、时序或者名称认定对 L14 MC2 的依赖；这些数值的实际对象编码、代际和作用域尚未核实。锚点 L14 MC2 的 globalTaskId 2538 和 AI_CPU 的 2874，以及 connection 98566 的三个 notify TASK ID，在 COMMUNICATION_TASK_INFO 中均无精确 globalTaskId 匹配；只说明该表中没查到关联，不能证明不存在其他同步。

rankSize=8、`srcRank=0` 和其他字段中出现的 `dstRank=1/7` 是该导出表字段，不足以还原 group_name_5 的全局 rank 成员或跨进程的同次 collective；切勿把名称或数字相同当作组映射证明。参考 [CANN 9.0 数据库格式](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_1144.html)中对 `COMMUNICATION_TASK_INFO` 关联字段的说明；本次历史采集生成版本与该说明的适配尚未完全证明。

复现 `python3 agents/diagnosis/run_ep8_l14_comm_notify.py`：wrapper 检查 stage 容器身份、镜像和 `/work` 挂载，再只读、SHA 校验指定 DB。结果为 `agents/diagnosis/ep8-l14-comm-notify.json`，含局部通信字段和 TASK 行、不含 prompt 或权重。查询按 `opId` 聚合所有子任务，并在全库内检查所选通知 ID 的每个 opId 的行数；脚本不启动新 NPU 负载，不修改推理实现。这里无法确认采集覆盖、跨轨时间精度、事件实例代际、通信组实际成员或与 MC2 的数据关联。

查询脚本 SHA-256 `0e1682a3064856785c3dc3348209cede4f76f62fa27da8db6063f2668b38393b`；wrapper SHA-256 `842930a4f41aa5fda307812fa0d3cad228cdaa930b60ca0d4f9fa73f2e4ce4c6`；结果 SHA-256 `ef26746403b842bb4933df12b354d9eeaa08f7d4acdd2e2c02495f21245f3a1b`。脚本显式校验 168 个任务均不携带 `connectionId=98566`，且不含 AI_CPU 的 `globalTaskId=2874`；这仍不排除两组记录存在未识别的上游事务关系。

## 同一历史 profile 中通信组元数据的精确匹配（后续补充）

上面的“仅凭 rankSize 等字段不足以还原成员”仍成立；之后对 **同一 10 月 29 日 EP8 profile DB** 找到了另一条元数据线索。固定 `COMMUNICATION_OP.opId=209` 的 group 字符串，与该 DB 唯一的 `parallel_group_info` JSON 字典键逐字节相等；三条字典项中恰好一条匹配，JSON 未发现重复键。该条目 `group_name` 按严格 allowlist 为 TP 标签，`group_rank=0`，`global_ranks` 含 8 个不重复整数，与 opId 209 保存的 `rankSize=8` 数量一致。同库 `RANK_DEVICE_MAP` 中目标 device 8 只有一条记录、其 rank 出现在列表的 group-rank 索引位置。完整成员列表只保存在本地证据 JSON，不应复制到公开报告。

这些是 **同库数据内部的一致性**，不依赖另一日期的 profile 或同名推测。它提高了把该条通信记录关联到元数据所标注 TP 组的可信度，但不能单独证实历史采集/exporter 的字符串与真实 communicator 的生成及生命周期契约、动态重建情况下元数据时点、组注册完整性或事件实例对应。冻结源码中的 `torch_profiler_interface.py:372–412` 描述了一个以 communicator name 为键保存 group 信息的可能生成路径，但它与这个历史 DB 的实际生产版本尚未完全绑定；TP 是元数据标注，不据此认定同一 collective、与 MC2 的依赖、等待或因果。之前无法仅靠通信任务局部字段推断成员的限制不变。

只读 CPU 复现：`python3 agents/diagnosis/run_ep8_l14_group_key.py`。容器 ID、镜像及 `/work` 绑定均于执行前后校验；SQLite 使用 `mode=ro` 和 `query_only`，对输入 DB 做 SHA-256 校验。数据库 SHA-256 `e07ee2428be27c6129afd6dd0be8b700830048f30f13411b701e97c5d88e2486`；查询脚本 SHA-256 `722365df670f88428b5dd79b058e652e7b3ea5a37cc8818f15d52b3d5e20c453`；wrapper SHA-256 `ab301e37196096cf67693677f5cc42fd5904765c96413182eb54407656f3079f`；结果 `ep8-l14-group-key.json` SHA-256 `e368e211fb5f2c21b153da733e7e79e064f012aea3e04143d0e05dba43b11f99`。不访问 NPU、不修改推理实现。
