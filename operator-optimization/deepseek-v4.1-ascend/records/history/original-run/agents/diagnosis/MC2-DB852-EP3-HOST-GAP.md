# EP3 第 15 层 MC2 前间隔：已有主机记录与设备记录

本分析沿用 [同签名记录](MC2-DB852-L15-L36.md) 和 [窗口](mc2-db852-ep3-gap.json)，只查询该 rank 现存 profiler SQLite。窗口起点记为 0，即选中 `MoeGatingTopKHash` TASK 结束，窗口终点为目标 MC2 TASK 开始，共 2055.040 µs；下文是数据库导出的同轴相对时间，不假定跨 host/device 时钟已校准。

- 所查数据库的 `SESSION_TIME_INFO`、TASK、CANN_API、PYTORCH_API 和 OVERLAP_ANALYSIS 的首末记录包围此窗口；这是覆盖范围检查，不证明采集未掉事件，也不证明设备其他活动被完整采集。COMMUNICATION_OP、MSTX_EVENTS 表在窗口没有相交条目，不排除内部通信或未被记录的同步。
- OVERLAP_ANALYSIS 有一条边界精确等于该窗口的 type=3，数据库枚举名 `FREE`。其来源、计算算法及采集覆盖尚未确认，不能将其解释为芯片所有资源都物理空闲；TASK 存在一条目标任务的候选另一视图，从终点前 1.040 µs 开始。该视图 globalTaskId 71645 与目标 2380 的 stream/task/model/connection 相同，起止相差 1.040/1.000 µs，但 ID、context 不同，尚无导出器契约证明它们等同。
- 目标 gating TASK 的 connectionId=95811 与同 ID 的 CANN `launch` 相接，目标 MC2 TASK 的 connectionId=95824 与同 ID `launch` 相接。两条 launch 在同一主机线程 `globalTid=9139690412736`，分别位于 -182.955 至 -173.625 µs 和 1912.535 至 1930.785 µs。连接号仅支持本数据库记录关联；不要把 host 结束到 device 开始的时间差直接当作硬件排队或通信耗时。
- 同一主机线程中，`Dequeue@aclnnMoeGatingTopKHash` 为 -192.575 至 1862.585 µs，嵌套的 `aclnnMoeGatingTopKHash` 为 -190.775 至 1857.125 µs，包含上述 gating launch；接着 `Dequeue@aclnnDispatchFFNCombine` 从 1866.725 µs 开始，嵌套 `aclnnInnerDispatchFFNCombine` 从 1869.055 µs 开始，包含 MC2 launch。两个 dequeue 同线程呈先后顺序；采集表明该主机调用跨度与设备窗口数值上高度重合，但 host/device 时间校准、CANN 调用内部行为和 profiler 干扰尚不明确，不能认定 2 ms 调用跨度是 CPU 运算、锁等待或导致设备空闲的原因。
- 另一个主机线程在该窗口记录 `Event::record` 和 `Event::wait`（分别 540.785–743.895 和 765.245–955.945 µs）及相关 enqueue；尚无 event ID、目标 stream 或因果关联，不能把它们作为此 MC2 的同步等待证据。它们也不是设备执行事件本身。窗口主线程有其他算子准备活动，不能说进程整体无工作。

## 仍缺什么

缺已记录事件的可靠时钟误差和采集丢失状态；缺两个主机线程之间 event/stream 的身份及与目标任务的依赖；缺 `aclnnMoeGatingTopKHash` 内部较长尾段的调用栈或运行时细粒度时间线。当前现有采集只能提出“主机侧长调用与目标间隔可能相关”的假说，不能决定因果，也不能把这一例推广到其他层或未插桩服务。继续检查现有 queue 关联和调用栈，若无进一步记录，将这些作为后续诊断需要的证据，而非补造结论。

## 复算

在任务根目录运行 `python3 agents/diagnosis/audit_mc2_db852_ep3_host_gap.py`，通过既有 SSH wrapper 在 169 自有 stage 容器只读数据库，生成 [结构化结果](mc2-db852-ep3-host-gap.json)。脚本 SHA-256 `021d9640368c0192174c51dfb5686cb48f1e2d8cc496936b142451d0dd47b2db`；结果 SHA-256 `f1d627e1788e4b411d539010387dc4b0a31a3504c903ad0e55546913f6591877`。JSON 带查询结果及输入摘要；不包含提示词、模型输出或权重。

## 三个样本的主机调用对照

选用同 digest 的 L15 EP0、L15 EP3 和 L36 EP3：对每个目标 MC2 TASK 按精确 start/stream/task/model 查出唯一行，在同设备同流中取其之前最近的 `MoeGatingTopKHash` 已关联 TASK。L15 两条此前与同 scope 结果一致；L36 的最近 gating taskId 41455 比 MC2 taskId 41456 小一，但尚未额外绑定其 layer 作用域。每条 gating 与 MC2 的 connectionId 各唯一匹配一条 CANN `launch`，再在该 launch 所在主机线程上找完整包围它的对应 API 和 dequeue；下面仅列每个自身主机区间长度和对应的设备 TASK 相对间隔，单位 µs：

| EP/层 | gate→MC2 设备记录间隔 | gating CANN API 时长 | gating dequeue 时长 | MC2 CANN API 时长 |
| --- | ---: | ---: | ---: | ---: |
| EP0/L15 | 1.340 | 1105.400 | 1110.340 | 1078.500 |
| EP3/L15 | 2055.040 | 2047.900 | 2055.160 | 1961.680 |
| EP3/L36 | 569.480 | 573.320 | 578.380 | 663.910 |

EP0 这个对照说明“gating 主机调用长”并不必然对应“该 rank 的两个设备任务间隔同样长”。其 gating launch 的导出时间比 gating TASK 开始早约 131.564 ms，EP3 两条分别早约 145.335 和 152.605 µs；在未证明跨域时钟对齐及 connection 语义前，不能把差值解释为物理延时或跨卡依赖，亦不能断言 EP0 是普通线上状态。样本都来自同一次带 profiler 的历史采集，不能推广到非插桩性能。

复算 `python3 agents/diagnosis/compare_mc2_three_host_gate.py`；脚本 SHA-256 `40ce98736e5a6eadcad6f161ec430f0e722e20b071ca720097ad4cd3c8068d28`，结果 [mc2-three-host-gate.json](mc2-three-host-gate.json) SHA-256 `4d2917f10a9cb2514d7b44457eb2b6e8aae821f08b07720febb55606fb259312`。脚本记录完整匹配行和覆盖关系，缺匹配会报错或保留错误，不访问 NPU。

## Gate 主机调用尾段的范围检查

选定 gating TASK 的 connectionId 为 95811，该 ID 唯一匹配的 host `launch` 落在 `aclnnMoeGatingTopKHash` 同线程 API 范围内；外层 API 自己的 connectionId 为 95809，二者不同，不能要求 parent 和 child 的 connectionId 相同。launch 结束后到该外层 API 结束，两个同线程记录的时间差为 **2030.750 µs**。在此前导出的设备任务间隔窗口内，窗口起点之后到该 API 结束的 **1857.125 µs**，已保存 CANN_API 匹配集中没有该线程的其他条目；该查询只包括与窗口相交的 API，不能把窗口开始前的约 173.625 µs 一并声称覆盖，也不能排除未采集、未导出或非 CANN_API 的子活动。

该结果限定在同一线程的 profiler API 标注；不能据此断定尾段为 CPU 有效计算、锁或通信等待，也不能把它推成线上 latency 或可消除开销。ZCode 基于现有 JSON 做了独立数值检查（`zcode/operator-ceiling169-v2-gate-host-tail.zcode.json`）；它对窗口外覆盖的进一步推论超出输入范围，本报告以上述严格范围为准。可运行 `python3 agents/diagnosis/check_mc2_gate_host_tail.py` 复核，脚本 SHA-256 `23f28ffd2c6820f6da7dba13969a743077b9e68426615e70213fc3ff04f4aff6`。
