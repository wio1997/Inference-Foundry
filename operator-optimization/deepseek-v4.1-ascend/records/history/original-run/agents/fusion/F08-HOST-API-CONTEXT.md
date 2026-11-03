# F08：EP3 选定主机 API 的嵌套覆盖

只读分析既存 EP3 profiler 数据库：短样本 task ID 13224/13225/13226，长样本 13466/13467/13468；同一记录的 launch 线程上，分别查找包围 launch 的 `aclnnRmsNorm` 与 `aclnnInplacePartialRotaryMul`。本页的“短/长”是按 scatter 设备任务持续时间定义，不能将二者差别直接归因于主机 API。

| 样本 / API | 外层 API 耗时 µs | 同线程内已记录子 CANN API 时间并集 µs | 外层扣除该并集 µs | launch 结束至外层 API 结束 µs |
| --- | ---: | ---: | ---: | ---: |
| 短 / RMSNorm | 1305.920 | 13.980 | 1291.940 | 1284.900 |
| 短 / RoPE | 1685.320 | 12.240 | 1673.080 | 1668.830 |
| 长 / RMSNorm | 392.050 | 9.470 | 382.580 | 378.470 |
| 长 / RoPE | 430.530 | 11.350 | 419.180 | 414.260 |

子 CANN API 包括 stream 属性、launch、内嵌的 `aclrtLaunchKernelWithHostArgs`，按区间**并集**而非直接相加；四个外层 API 都被同线程相应 `Dequeue@aclnn...` PyTorch 记录包围，外层区间内没有记录到嵌套 PyTorch API。四个外层调用的大部分时间落在 launch 结束之后，但以现有记录无法区分内部工作、等待、线程被调度、剖析扰动或其他未记录活动。这些时间不等于设备空闲、同步等待、纯框架调度或可节省时间。

实际配置 `activities` 同时包含 CPU、NPU，但 `experimental_config._host_sys` 是空列表；host sample 的 `host_cpu_profiling`、`cpu_profiling`、`host_osrt_profiling`、`hostAllPidCpuProfiling` 均为 `off`。数据库现有表中没有 CPU 调度事件表。配置只能证明这些项未开启，不能证明不存在其他来源的 CPU 状态证据。需要线程级运行/切换或用户态/内核态栈、同步等待与 profiler 开销的对照数据才能继续区分；本阶段遵守不启动新 NPU 负载的限制，未进行新增采集。

复核脚本 `f08_host_api_context_audit.py` 以 SQLite `mode=ro` 读取指定捕获；聚合输出 `f08-host-api-context-audit.json` 保存原始时间、全部匹配的子 CANN API、PyTorch 祖先、配置和表清单。时间为 profiler 中记录的同线程区间，采集完整性和 CPU 时钟执行语义未知；不推断未记录的活动不存在，不外推全局或无采样时的耗时。
