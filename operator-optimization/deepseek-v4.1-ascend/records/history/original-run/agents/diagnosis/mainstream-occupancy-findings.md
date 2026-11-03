# 已保存主流样本：Indexer 与 MC2 的时间分布

## 发现

在同一个 rank、同一个已归属采样轮次，先合计该 rank 的 40 个互不重叠的 target 层主流窗口，再分别求各窗口内的 Indexer 和 DispatchFFNCombine（MC2）任务耗时之和。对每张卡分别算比率，再取 16 个 rank 的中位数：

| 采样轮次 | 40 层窗口总时长中位数 | Indexer 总时长中位数 | Indexer 占窗口比中位数 | MC2 总时长中位数 | MC2 占窗口比中位数 |
|---|---:|---:|---:|---:|---:|
| 1 | 31.05 ms | 0.163 ms | 0.53% | 11.54 ms | 37.11% |
| 2 | 36.06 ms | 2.472 ms | 6.85% | 13.32 ms | 36.93% |
| 3 | 36.20 ms | 2.472 ms | 6.82% | 13.53 ms | 37.36% |

在第二、三轮中，仅 24、28、32、36 这四层的 Indexer，已各自合计约 1.98 ms（16 rank 中位数），占同口径 40 层窗口时长约 5.5%。其余四个 Indexer 层共同约 0.49 ms（表中分别为中位数，差值仅作近似）。各卡原始结果见 `mainstream-occupancy.json` 的 `rank_run_rows`，统计脚本为 `analyze_mainstream_occupancy.py`，冻结窗口和 16 个 CSV 的哈希保存在 JSON 中。ZCode stage `operator-ceiling169-v2-086-mainstream-occupancy` 成功，`model_verified=true`，未运行设备。

## 这说明什么

在这组有限采样中，MC2 的主流任务耗时比 Indexer 更值得继续拆查，同时 Indexer 的长耗时在第二、三轮中并非只发生在孤立的某一层；后四个候选过滤层值得一同研究。这是**采样窗口内的设备任务耗时分布**，是定位排查方向的证据。

这不等于端到端的百分比，不能将不同 rank 的耗时相加，也不能得出 37% 或 6.8% 可以消除：其他 stream、不同 rank 和主机可能同时工作；profile 有扰动；采样是否代表 32K/1K/C8 真实请求的各阶段尚未证实。MC2 包含通信和计算，现有字段不能拆分二者；Indexer 的动态有效长度、候选数和请求身份也未能获取。跨轮次耗时差不能归因于具体原因。

## 复算边界

脚本仅从 `module-call-metrics.json` 选取每个 rank 的 model 49、相应 main stream 上已归属的内部采样 run 1/2/3，校验每个 rank 每轮包含 40 个不同层且各窗口不重叠。每个任务须完整落入对应层窗口，另校验每个 rank 每轮的 8 个 Indexer 和 40 个 MC2 任务、以及两类任务时间之间互不重叠。其余 stream、采样之外及未归属的任务不在分母或分子中。

在自有 staging 容器内，只读调用脚本并传入冻结 task 与 trace 路径即可复算；本地包装器 `run_mainstream_occupancy.py` 经 SSH 调用容器并写入 JSON。若结果文件已存在，包装器会拒绝覆盖；可直接运行分析脚本并另存，以保留证据。
