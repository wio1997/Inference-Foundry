# EP8 两个第 14 层任务：二进制绑定审计

结论：**有两个可复算的二进制候选，尚不能严格证明设备 task 使用了它。** 只读取既有采集的 C 日志、同一 worker 的 sidecar 和已经离线导出的 profiler 结果，没有新 NPU 采集或修改目标代码。

已有 queue / HostToDevice 证据将层 call 746-20、746-65 分别关联到设备 8、stream 47 的 task 47465、50913，并提供每次 dequeue 线程里的 `aclrtLaunchKernelWithHostArgs` API 时段；该链的导出完整性本身受 SQLite I/O error 限制，详见 `EP8-LAYER14-QUEUE-DEVICE-REVIEW.md`。本审计进一步检查 C 日志中对应 PID/TID 的完整调用，不能消除前述限制。

## 条件候选

先假设 profiler 的 globalTid 采用已保存 PyTorch parser 的 `(pid << 32) | tid` 且跨工具、跨 PID namespace 对应关系成立，得到候选 PID 746、TID 6857。同一 `events-746.jsonl` 的 292 条记录给出 `time_ns - perf_counter_ns` 范围 `[1791451440214551180, 1791451440214552830] ns`。为探索性搜索，将此范围两端各外扩 1 ms；这是**未证明的经验假设，不是时钟误差上界**。若另外假设 profiler 时间、C recorder 的 MONOTONIC、sidecar 两时钟间映射在整个区间内遵循这一范围，则：

| 层调用 | 设备 task | 可包围 profiler API 时段的完整 C 调用 | 记录的 entry / generation | pre-call SHA-256 |
| --- | --- | --- | --- | --- |
| 746-20 | 47465 | 331572 | 1000010 / 86 | `4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9` |
| 746-65 | 50913 | 334558 | 1000010 / 86 | `4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9` |

在上述假设和**已记录、成功返回、begin/end 可唯一配对的调用集合内**，两次各有一个满足区间包围的候选；候选两端的 `dropped_since_last` 均为零，但这不证明中间或别处没有丢失。详细的实际时间、函数/二进制句柄、stream 指针与各自可行偏移区间见 `ep8-binary-existing-audit.json`。与表中相同的 digest 曾在已检查的安装文件中出现，不能据此断言 runtime 加载或执行了完全相同字节。

## 为什么不能作为严格绑定

- 两套时间戳无法仅凭一次侧录确定数学上可靠的转换误差：sidecar 先读 monotonic 再读 wall，两次调用间隔无已证上界，也未证明时钟没有跳变或 profiler 使用相同时间映射。两次均有满足经验区间的候选是支持性现象，不是机器检查的因果连接。
- PID/TID 的解码来自同版本 PyTorch parser；CANN 工具与 C interposer 的线程/命名空间、PID 生命周期没有独立证明。C 日志的 stream 是指针，没有与 profiler stream 47 的严格映射；也没有 `aclrtGetThreadLastTaskId` 记录把该函数 handle 直接接到 task ID。
- 该 PID/TID 有 475 对完整调用，197 组不完整或重复；出现在该线程记录上的 `dropped_since_last` 合计 585，是进程共享丢失计数在该线程上的报告，不一定是该线程造成。最新可见统计有 `state_incomplete=true`、`pending_drops=236517` 和 launch/mutation 重叠；没有 process-exit 统计。即使记录中只有一个候选，也不能排除真实调用被漏记或句柄复用。
- entry 候选仅识别 binary-level 关联，不等于函数名；hash 为输入 pre-call hash，不证明 runtime 实际读取字节。模型或 graph replay 缺少严格 graph 节点→task→函数的关联，缺失 graph 记录不能证明图没有参与。

所以两个 task 的具体二进制/函数绑定状态均为 **UNKNOWN**；以上候选可以指导后续取证，但不能作为准确性确认或算子理论上界输入。

## 复现

`python3 run_correlate_ep8_binary_existing.py` 在自己的 stage 容器内对冻结任务目录进行低优先级、只读 CPU 分析，生成 `ep8-binary-existing-audit.json`。记录输入读取前后大小与 mtime 一致；C 日志 SHA-256 为 `3fc3a80b67cbfd08dba9d259bfba2c70ada1bb1414aa2c1772984e25be62bcc1`，sidecar SHA-256 为 `1796e6fc17fed3ec136a5c98978acdcc54872b4bdd3b4cdc64f66a539a0fd434`。输入与输出均是既有采集的元数据；没有输出权重、张量或提示词内容。
