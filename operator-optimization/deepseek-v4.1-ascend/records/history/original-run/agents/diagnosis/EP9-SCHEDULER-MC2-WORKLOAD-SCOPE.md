# EP9 的 scheduler 记录与 MC2 物理行数

## 已绑定的信息

对同一已保存采集 EP9、同一 Python pid/tid 752，检查两次已选 target 第 0 层 host scope。依据 sidecar `perf_counter_ns`，层 scope 分别完整嵌于唯一选中的 `NPUModelRunner.execute_model` scope；所记录 scheduler digest 与先前 trace flow/CSV 关联的层 scope 相同。设备任务与该 host scope 的连接复用既有的 trace flow，异步设备任务可能在 host scope 返回之后完成；这些都不是客户端请求身份或 tensor storage 绑定的证明。

| 匿名调度摘要 | runner / layer scope | 记录的调度总数 | scheduler 条目数 | MC2 输入物理 M | 8 × M | 差值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1b826883…` | 752-5 / 752-6 | 8180 | 3 | 1023 | 8184 | 4 |
| `8c18c293…` | 752-50 / 752-51 | 5144 | 4 | 643 | 5144 | 0 |

第二行的 MC2 就是此前选中的 stream 47、task 46755、trace 时长约 273.565 ms 的 profiler 记录；不是线上服务时延。第一行选择 stream 47、task 43460。两个 scope 的已保存 scheduler 总数等于各自匿名条目的 `scheduled_tokens` 之和；这些数是主机侧调度元数据，不是各 expert 收到的有效 token 数。

选择乘以 8 只是依照本任务已冻结 8-way 配置做的算术对照。第一行 8×1023 与 8180 差 4，第二行数值恰好相等：**不能用这个公式在两组中都原样复原调度总数，更不能用它推断某个 rank 有效 mask、padding、专家分布或物理工作量**。两个样本属于不同调度摘要，不能拿时长比较声称相同负载或得出因果。

## 查找范围与官方格式

所选 `runtime/logs/service.log` 的 SHA-256 与已有采集记录一致；只检查九个精确关键词并输出行数，不输出日志原文、请求 ID 或 tensor。`expert_token_nums`、`expert_tokens`、`local_token_per_expert`、`topk_ids`、`expert_load`、`num_actual_tokens`、`x_active_mask`、`router_output` 各为 0 行；`routed_experts` 为 1 行，未在此查询中检查其内容或判定含义。这不能排除其他命名、其他日志、另一个数据源或该行含有可用信息，因此不能说整个采集没有专家值。`metadata/events-752.jsonl` 的固定目标 scope 提供 scheduler 条目和 tensor metadata，未在本次只读摘要提取专家数或 router 实际 ID。

冻结采集代码 `diagnostic-20261029/instrumentation/sitecustomize.py` SHA-256 `6e9bbadd7205a08f95de0c63a022645accbfe9fbae8a74e37d7859e3b4c8ef08`：其 `scheduled_total` 从对象属性 `total_num_scheduled_tokens` 取值，每条 `scheduled_tokens` 来自调度计数，并对请求 ID 使用 HMAC；这描述代码的预期行为，不能证明历史进程加载对象的二进制身份或设备 mask。

华为 [CANN 9.1 beta2 `aclprofEventAttributes` 原文](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_1371_2.html) 的 `aclprofTensorInfo`/`aclprofTensor` 结构列出 op/stream、张量类型、dtype、format 和 shape，未列出张量元素。它并不证明该版本与历史采集一致，也不证明所有 raw slice 无其他编码；本任务对 EP3 特定 slice 的实际解析边界见 [tensor_info 契约](MC2-EP3-TENSORINFO-CONTRACT.md)。此前将 `aclcppdevg_03_1367.html` 误指为该结构已纠正，实际该页是 `aclprofAicoreMetrics`。

## 复算

在 stage 容器运行 `python3 agents/diagnosis/query_ep9_scheduler_mc2_workload.py`，得到匿名 [ep9-scheduler-mc2-workload.json](ep9-scheduler-mc2-workload.json)。脚本对已保存 JSONL、EP9 metadata 和 service log 核验 SHA-256，验证两个 host scope 同线程嵌套、调度摘要及所选 MC2 唯一性，只输出汇总和精确关键词行数；不输出请求 HMAC、模型权重或 tensor 元素，也不调用 NPU。专家 count、每 peer 的实际贡献、输出在具体 replay 的值和 maxOutputSize 仍需同调用身份的额外证据，当前不能据此确定低活动原因。
