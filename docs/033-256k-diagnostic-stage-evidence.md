# GLM 256K→32 路由诊断与 D rank0 阶段证据

**状态：单次未计分诊断和局部指标观察；不是 1024 输出正式 cold/hot、框架性能对齐、收益或根因证明。** 本文不包含请求文本、响应文本、原始监控快照、完整服务日志或大 trace。

## 1. 诊断终态与健康观测

在 G316 已有服务和 `/ready` worker 探测配置下，单次 AISBench 诊断发送 **20 请求、客户端 C16、实际每请求 262144 输入 / 32 输出**。原明细审计记录 20/20 成功、长度均符合；P 路由日志窗口记录 20 个唯一 POST 开始和结束、均为 HTTP 200，未匹配到 503。该次负载没有复现此前 256K 首次 cold 的路由失败，但不能以此证明其他负载下不会复现。[原 256K cold 失败记录](028-G316-first256-cold-router-failure.md)

健康监控共 **114 个间隔采样**：D `/ready` 114 次 200，router `/workers` 114 次 200 且采样时两 worker 均 healthy；P `/ready` 113 次 200、**1 次超时**。相邻样本中位间隔约 15.008 秒、最大约 23.661 秒，因此不能排除采样间隙或最后样本后的瞬态异常。诊断日志仍记录终端环境及临时状态文件读取/JSON 解析 warning；不能把它说成全程无任何 warning。

现场记录的配置为 `retry=1`、stream、`batch_size=16`、`min_tokens=max_out_len=32`。源端对源码与结果字段链路做了检查，**原始 SSE usage 帧没有单独捕获**，不能称每一帧已独立复核；包装命令继承的最终退出码为 0，但没有独立的内层/外层退出码收据。HTTP 200 是流式响应层观察，20/20 和 token 字段来自客户端明细审计。数据集、配置和原始明细只保留 SHA，不公开文本。[终态脱敏摘要及其源摘要 SHA](../evidence/diagnostic_033/terminal_summary.json)

## 2. 两个 D rank0 直方图窗口

只选 `tp_rank=0, moe_ep_rank=0, pp_rank=0`，分别计算同一角色、同一标签的 `after_sum − before_sum` 和 `after_count − before_count`。第一窗口 D：11:24:29.228438→11:30:03.406243 UTC；第二窗口 D：同一起点→12:03:45.027905 UTC。两个窗口**重叠，不能相加**；第二窗口终点晚于诊断结束，不能声称其 8 个样本全部属于这 20 个请求。P/D 两端采样时间不同，未进行跨机时钟校准。

| D rank0 指标 | 第一窗口 | 第二窗口 | 含义与边界 |
|---|---:|---:|---|
| `decode_bootstrap` | 4 个；总和 4410.119252 秒；均值 1102.529813 秒 | 8 个；总和 9386.526219 秒；均值 1173.315777 秒 | 阶段直方图的局部增量，不是请求配对或墙钟总和。 |
| `kv_transfer_bootstrap_ms` | 4 个；总和 72.240471 ms；均值 18.060118 ms | 8 个；总和 140.404890 ms；均值 17.550611 ms | D prealloc入口到首次poll观察`WaitingForInput`的候选子段；不能当作已证网络握手或纯wire时间，见[语义修正](057-kv-resource-and-early-policy.md)。 |
| **条件性整体减子段** | **1102.511753 秒/样本** | **1173.298227 秒/样本** | 仅在下述同 setter、同样本和采样条件成立时，作为剩余区间均值；不等于纯网络。 |

既有源码只读核验记录指出，两指标由同一 setter 更新：整体为 `decode_transfer_queue_entry_time − decode_prealloc_queue_entry_time`，子段为 `bootstrap_done_time − decode_prealloc_queue_entry_time`。**仅当没有进程/标签重置、抓取没有跨过该 setter 两个指标的更新、且两组增量确实属于相同样本**，才可用 `(整体增量秒 − 子段增量毫秒/1000) / 样本数` 得到上表条件差值。两个计数相等本身**不能证明样本身份相同**；也没有请求 ID 关联，不能断言它们全部来自当前诊断。

剩余区间可包含首次观察接收侧 `WaitingForInput` 状态后到 transfer queue 的等待、准入、分配等，不是单个算子耗时，也不是纯网络 wire time；此前称该状态为“握手完成”证据不足，详见[语义修正](057-kv-resource-and-early-policy.md)。P 端各指标的更新时点不同，不能因为第一窗口都有 4 个样本就直接相减；第二窗口甚至出现 P 阶段 8 个而 bootstrap 子指标 9 个样本。不能跨 P/D、跨 rank、跨重叠窗口求和或当作请求端到端墙钟。

## 3. KV 资源准入：有条件的候选，不是根因

已有 G316 D 启动记录的 full-KV 配置容量为 **345728 token**；既有 rank0 日志两个截面报告使用 **262272 token**，当时 running=1、prealloc 排队分别为 9 和 7。这些是间隔观测，不能与某个直方图样本按请求绑定。[归档的 D 启动及容量记录](../evidence/matched_SG_G316/readiness/2202/readiness.json)

若且仅若已用 262272 是同一池的不可回收占用、新 256K 请求无有效共享且需要各自分配完整上下文，则算术余量 `345728 − 262272 = 83456` 小于 `262144`。若再按配置为该请求预留 512 token、页对齐且无共享，需求约为 `262656`。这使 full-KV 准入等待成为值得验证的**假设**，但实际可驱逐/锁定前缀、allocatable、metadata、Mamba、请求槽、请求 ID 和命中哪个分支尚未逐请求验证。不能把该条件算术写成“已证实 KV break”，也不能外推到 32K/64K 或两框架性能差距。

可证伪的下一步是对已有日志按请求/room 关联 prealloc entry、bootstrap done、transfer entry、实际可用/可驱逐/锁定预算及其他准入条件；若字段不存在，标为缺失而非事后猜测。必要时在不干扰其他运行任务的隔离窗口补采。只有具体约束及正确性得到证据支持，再做单变量端到端 A/B。

## 4. 脱敏与离线复现

[六份快照的 rank0 白名单 sum/count](../evidence/diagnostic_033/rank0_snapshots.json)只保留角色、UTC 采样时间、状态、源 SHA 和指定数值，未复制原始 metrics 的主机字段或其他 rank。六份源快照、原聚合及原重算脚本的 SHA 均在该文件中；终态摘要另列四份审计源和原始明细 SHA。提交文件不包含这些原始资料或请求文本。

此前独立分析记录的 D `req_time_stats.py` SHA 为 `21e17d68d3f18ffdb1a1203fe814f2ddcea9869021a3fa9f43210222d114a1e6`，D `decode.py` SHA 为 `39cf972b6bcc04cc8c150c650d845daaa68ccb511a69a71c5b0b8bdf7322a70d`；当前 P 源文件与旧副本不一致，不能据旧 P 源单独断言当前调用链。源码 setter 语义来自既有只读核验报告及 SHA，**这里没有重新获取完整源码或做请求级 trace**。

在仓库根目录只读运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/reconstruct_033.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_reconstruct_033.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[复算脚本](../scripts/reconstruct_033.py)只从仓内六份脱敏数值重建两个窗口，历史兼容字段名含 `handshake` 但指的是 `kv_transfer_bootstrap_ms`，不证明网络握手；拒绝非 rank0、缺失指标、计数回退、异常数值或不匹配角色；计数不等时不输出条件差值，并始终标注样本身份未证实。[测试](../scripts/test_reconstruct_033.py)覆盖计数不等、重置、非零和无计数、非有限值和额外字段；这只复现归档的聚合算术，不复核未入仓原始请求、SSE、设备 trace 或源码行为。
