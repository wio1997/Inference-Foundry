# 16 rank 第 0 层 MC2：其他 stream 的导出时间重叠

已有 trace 中，对 16 rank 各 3 个 source-attributed target 第 0 层窗口，分别在同一个 rank 的 CSV 中唯一选出主流 MC2，计算其他数字 stream 导出行在该 MC2 时间内的裁剪并集。每卡单独求 `并集 / 本卡 MC2 时长`，再对 16 个值计算中位数和范围。

| 采样轮次 | MC2 时长中位数 | 其他数字 stream 有导出重叠的 rank | 同卡重叠比例中位数 | 比例范围 | 其中命中 `allgatherAicpuKernel` 的 rank |
|---|---:|---:|---:|---:|---:|
| 1 | 1151.137 μs | 0/16 | 0% | 0%–0% | 0/16 |
| 2 | 666.727 μs | 16/16 | 34.45% | 13.79%–100% | 16/16 |
| 3 | 1007.430 μs | 12/16 | 7.94% | 0%–88.87% | 12/16 |

这 48 个样本中，其他数字 stream 的匹配导出行一共 28 条，类型都是 `allgatherAicpuKernel`；没有命中 `hcom_` 名称或 stream 非数字的行。**范围中的 100% 仅表示该卡上有导出行覆盖了该 MC2 的整个时间区间**，不意味着实链路被 100% 掩盖。第一轮没有命中也不代表没有其他设备活动或 MC2 内部没有通信。

如果按本地 rank 标识分组，第二轮 DP0 的 8 个本卡重叠比例中位数约 61.76%，DP1 约 15.09%；两组 MC2 时长和导出行时序本来不同。这只是分组描述，run 序号未与跨 rank 的相同请求严格对应，不能解释组间差异的原因，也不使用跨卡时间戳做差。

## 这个结果能和不能说明什么

存在同卡、同一 CSV 导出时间坐标下的区间重叠；记录并未证明被称作 `allgatherAicpuKernel` 的行与该 MC2 的数据或请求相同、真实链路正在传输、算力/带宽是否争用、通信是否被计算隐藏，或是否位于端到端关键路径。其他 stream 完全可能属于另一模型或请求；model ID 为 `4294967295` 也不能直接赋予模型身份。MC2 是复合内核，内部通信、计算和同步阶段仍无法由此表拆开。

没有数字 stream 导出行的时间，是针对该 CSV 的集合差，不能称为设备空闲、纯计算或等待。数字 stream、`N/A` 行及不同类型并集可能有重叠，不可把它们简单相加。结果受 profiler 覆盖、采样与时间转换限制；没有确认真实工作负载的请求阶段，不能外推三次样本代表完整服务。

## 可复核证据

`analyze_16rank_mc2_other_stream.py` 将各 rank 的任务裁剪到其 MC2 区间，分别保存逐行位置与并集；`16rank-mc2-other-stream.json` 包含 48 个样本及聚合、16 份 CSV 的 SHA。冻结窗口 SHA、trace 清单 SHA 和 16 份 kernel CSV 哈希均与对应记录一致。对 48 个结果独立重算其他数字 stream 区间并集，验证都处于本卡 MC2 范围且并集与未覆盖部分之和等于 MC2 时长；EP8 三个样本与前一版 `ep8-mc2-other-stream.json` 一致。

ZCode 完整 stage `operator-ceiling169-v2-102-16rank-stream` 的 exit code 为 0，`model_verified=true`，执行模型 `deepseek-flash`。它通过自己的 staging 容器只读既有 CSV，没有运行 NPU，也没有改服务。

## EP8 补充：数据库中的跨流事件窗口

将已有 `engram-producer-events-exact.json` 的所选 EP8 `EVENT WAIT` 行，与 `mc2-pmu-replay-links-ep8.json` 中同一数据库的 MC2 原始任务记录对齐；再用原 16 rank CSV 结果的 `(rank, run, start, end)` 精确选出 EP8 三次第 0 层 MC2，每次 DB 任务都唯一匹配。上述两份 DB 提取文件记录的源 DB SHA-256 均为 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`。在同设备的导出时间轴上，这三次 MC2 的整个区间均分别被 stream 87 和 stream 88 的 `EVENT WAIT` 行覆盖：

| 内部轮次 | MC2 时长（μs） | stream 87 事件先开始／后结束（μs） | stream 88 事件先开始／后结束（μs） |
|---|---:|---:|---:|
| 1 | 2035.220 | 235.395 / 613.947 | 78.118 / 592.128 |
| 2 | 944.321 | 244.676 / 622.367 | 78.339 / 602.008 |
| 3 | 1661.606 | 242.716 / 620.188 | 79.759 / 594.949 |

`EVENT WAIT` 是导出标签；上述是任务区间相交和包围关系，不是对 event 对象、其 producer/consumer 或等待语义的确认。共享 model 或 connection 在图任务中复用，不能证明因果依赖、MC2 导致等待、链路通信被计算隐藏、资源争用、设备空闲或可节省时间。事件在 MC2 之后仍延续也不能排除间接依赖或其他后续条件。事件源原先只选了与 0→1 和 13→14 层间隔相交的行，故本结果不是完整 event 覆盖，亦不能推广其他 rank 或层。它与此前仅检查 kernel CSV 其他数字 stream 的结论不冲突：CSV 的已导出 kernel 行不等于 DB 全部事件记录。

复现：从任务根目录执行 `python3 agents/diagnosis/recheck_ep8_mc2_selected_events.py > /tmp/ep8-mc2-selected-events.json`，将输出与 `agents/diagnosis/ep8-mc2-selected-events.json` 比较。脚本校验同源 DB 摘要、三次任务唯一匹配、来源未截断及覆盖，记录输入 SHA；仅操作本机既存数据，不启动 NPU。进一步定因仍需版本对应的 event 对象及记录/等待绑定、具体数据依赖和完整时钟契约。

### 六条事件在已提取 trace 参数中的精确关联

复用 stage121 的 `ep8-trace-event-args.json`，依据轮次、开始/结束时间、stream 和 global task ID 将上述 6 条 DB wait 与其预选目标关联；每条分别找到唯一 `EVENT WAIT` X 行，时间起点、model、stream、task ID 均一致，所选邻域无截断。三个轮次均为 stream 87 task 10 / global task 66327，以及 stream 88 task 16 / global task 66341。六行顶层 `id`/`id2` 为空，筛选后的标识都包含复用的 model 49 和 connection 184330，因此目前不能用这些字段把 wait 和特定 record 或 producer 配对。采集脚本仅保留符合规则的部分参数键及数字值，不能据此断言原始 trace 的所有参数必定没有事件句柄，也不能用相邻时间证明依赖。

可在任务根目录运行 `python3 agents/diagnosis/recheck_ep8_mc2_trace_waits.py`，与 `agents/diagnosis/ep8-mc2-trace-waits.json` 比较；脚本只读取两份已有小型 JSON，不重新访问 trace 或设备。该结果 SHA-256 `ed59179edca6627ad1784460afaba6b62c8c033464b9b18745a6f91a459a4352`，脚本 SHA-256 `06044ef787d75ea592491ea6556df81afbac69cdb30c550b9f01d515be60f452`，stage121 输入 SHA-256 `d2e9c8ee7ce178c106dbb02d7b3aae1bf59dc1f65206607e971ed65d156a7959`；其记录的 EP8 trace SHA-256 `3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99` 未被先前独立冻结 hash 验证。事件生产端、关联数据和运行时行为仍未知。

### 六条 wait 原始 trace 的完整字段核查（stage219）

对上述 EP8 的六条指定事件，ZCode stage219 以开始时间、stream、task ID 和 model 49 从原 trace 精确筛选，每条恰好匹配一个 `EVENT WAIT` `X` 行。六行原始顶层键完全相同：`args`、`dur`、`name`、`ph`、`pid`、`tid`、`ts`；**均不含顶层 `id` 或 `id2` 键**（此前提取的 null 不能区分缺键与值为 null）。六行 `args` 完整键也完全相同：`Batch Id`、`Model Id`、`Physic Stream Id`、`Subtask Id`、`Task Id`、`Task Type`、`connection_id`。按事件、通知、信号、句柄、信号量、栅栏、flow 和 correlation 字样筛选，这六行没有命名明显对应这些概念的参数键。这里只能说明**这六条选中行**没有找到上述名称的字段，并不能排除其他字段间接承载关联、其他 trace 表存在关联，或这些记录在运行时实际上有事件对象；复用的 task/model/connection 标识也不足以把它们与特定记录或生产者配对。

证据为 `ep8-six-wait-keys.json`（SHA-256 `c37c65ba99c391beb86ea9cf33b9a75287e2c21e8cc1417ea60d569a5a40f761`）及 `inspect_ep8_six_wait_keys.py`（SHA-256 `4aa5aa54a369fa3febb2ec4500f3f40c897ca3abb7e4db8e0b7b467f94aeea08`）；脚本以 6 条已选事件的摘要校验输入，检查 trace 文件大小及 SHA 与 stage121 一致，在读取前后核对文件未变化。此 trace 的 SHA 是先前采集时记录，**并非独立冻结校验**。查询仅输出结构性键名与候选标识的非空/类型，不导出任意参数值；不执行 NPU 或改服务。匹配范围不覆盖其他 rank、其他时段、其他事件或独立的 event-record 信息。
