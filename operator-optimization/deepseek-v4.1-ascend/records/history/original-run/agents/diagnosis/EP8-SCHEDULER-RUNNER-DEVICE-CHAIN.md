# EP8 调度—主机提交—设备任务的链条和边界

这是已有一次高扰动 profile 的定向复核，只读取已保存的三个带哈希聚合文件，没有启动 NPU，也未改推理实现。这里选 EP8 的 runner `737-5`、scheduler digest `1b826883…` 和 L14 长 MC2。结论只覆盖这个样本和导出记录。

## 哪些链条已关联

1. 已采到的 scheduler 中，所选规范化调度内容摘要匹配 1 条 scheduler 返回、8 条 runner 入口；其中 EP8 是 `737-5`。摘要内容包括匿名请求标识和调度字段，**不是全局唯一调度步、graph replay 或客户端请求 ID**。所选摘要包含 3 个匿名描述符和 8180 个 scheduled token；这并不保证各 descriptor 的所有计划工作都已经执行。
2. EP8 侧通过同线程、配对的 `NPUModelRunner.execute_model` 和各层 host scope，再经唯一 `torch_to_npu` flow 起止与设备 X，40 层各有 1 条 `DispatchFFNCombine`；这 40 条设备记录的 CSV 身份各唯一匹配。flow 关联的是 host 提交，设备执行可超出 host scope；它不是 tensor 的依赖边或逐请求完成边。各层匹配和原始文件 hash 见 `diagnostic-20261029/ep8-737-5-layer-mc2.json`，L14/L39 局部 flow 的精确复核见 `ep8-l14-l39-flow-intervals.json`。
3. L14 任务 ID 44217、stream 47 的记录持续 160365.95575 µs；同卡同 trace 的 L13、L14、L15 MC2 记录按该次导出顺序不重叠。40 条 MC2 在同一设备 trace 的 stream 47 中均按层号顺序、记录区间无交叉。**这只确认所选任务顺序，不证明每个间隔设备闲置，也不证明任务位于某个客户端关键路径**；其他算子和其他 stream 不在此顺序断言中。

## 时间边界为何不能按 CPU scope 算设备完成

以导出时间坐标作算术对照，所选 40 条 MC2 中，26 条起点和 27 条终点显示晚于 runner host scope 的终点。L14 的设备起点在导出坐标里比 host scope 终点晚 21531.288 µs，L39 的终点晚 739624.80125 µs。

**这一跨域位置只是导出记录表象**：当前没有对该采集 host/device 时钟转换和误差做独立校准，不能把这些差值当成经认证的排队时间、同步耗时或线上性能。即便后续校准支持该排序，host 提交 scope 也只能说明提交结束，不能等同设备完成、请求完成或者最终用户等待了这段时间。不得将不同层设备任务的 wall duration 相加推算请求关键路径；图 replay、其他流和采样缺失仍未封闭。

## 当前最短证据缺口

已采到的 scheduler 与 runner 匿名请求集合相容，但客户端合成 ID 和保存的 engine ID 无法关联；旧客户端 HMAC 因 key 不同不可用，且 engine ID 含随机后缀，详见 `RECAPTURE-METADATA-REVIEW-20261029.md`。因此没有从这条调度到某个客户端响应的完整身份链。还需要绑定本次运行的 graph/输出 completion 到具体 request、同一任务的 record/wait 事件对象及跨流数据生产—消费关系；并核定跨域时钟映射、误差和 profile 覆盖。没有这些不能判断 L14 MC2 是否位于某一请求关键路径，或将其 160 ms 解释为通信等待、设备空闲或可节约时间。

## 复算

在任务根执行 `python3 agents/diagnosis/recheck_ep8_runner_device_tail.py | cmp - agents/diagnosis/ep8-runner-device-tail.json`。脚本核对三个输入 SHA-256、调度匹配数、40 层及设备键，随后对已导出坐标做 Decimal 算术，并校验 stream 与任务顺序。它不重读服务器原始 trace；其正确性依赖上游各聚合与 flow 提取，不能成为独立采样或采集完整性证明。

- 脚本 SHA-256：`3ec39035a2cca7fb29903d95255203f36e7c8d7967bbd4f1d80ad616de5668d8`
- 输出 SHA-256：`a0fe00932b2d2852198b507b542b7e80e044897a7fbfeb5e54dd3396d8484ee5`
