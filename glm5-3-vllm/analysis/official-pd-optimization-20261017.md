官方 PD 已进入机制性能对照。目前完成对照及有限保护检查的配置为 P-v5 原生 MTP1、D-v4 原生 MTP1 和共享专家多流重叠；原样 MooncakeConnectorV2、官方 proxy、W8A8C8、int8 KV/Indexer、TP16/TP16、8K 及 SDMA 配置保持固定。

相对无 MTP 的官方图基线，八组有效输出吞吐倍率的几何均值为 **1.5694**，TPOT p50 倍率几何均值为 **0.5533**。每种配置均两轮、每轮120条正式请求。下表为两轮各自统计值的均值；分位数均值并非合并样本分位数。

| workload | output tok/s | TTFT p50 ms | TPOT p50 ms |
|---|---:|---:|---:|
| 256-64-c1-cold | 23.27 → 39.30 | 288.7 → 302.8 | 39.00 → 21.06 |
| 1024-64-c1-cold | 21.67 → 35.95 | 371.3 → 390.1 | 40.90 → 22.05 |
| 256-64-c1-hot | 23.57 → 40.30 | 250.2 → 256.0 | 38.99 → 21.06 |
| 1024-64-c1-hot | 22.60 → 37.54 | 252.5 → 270.5 | 40.91 → 22.75 |
| 256-64-c4-cold | 79.83 → 121.01 | 605.1 → 637.2 | 40.86 → 23.03 |
| 1024-64-c4-cold | 61.50 → 84.97 | 1015.5 → 1029.6 | 42.87 → 23.85 |
| 256-64-c4-hot | 79.44 → 119.19 | 604.5 → 640.5 | 40.78 → 23.02 |
| 1024-64-c4-hot | 76.40 → 112.42 | 633.1 → 707.0 | 42.77 → 24.27 |

每组使用固定输入、64输出 token、温度0、相同 seed。C4 是六个同步封闭波次；这些数据不建立持续并发容量或生产 SLO。连接/编译 warmup 和每组 warmup 被排除，冷前缀使用独立 salt，热前缀使用实际 prefix hit 指标核实复用。正式测量未开启 profiler、全量权重 hash 或备份。

机制选择有实际反例：MTP1 相对无MTP的吞吐提升35.5%–65.0%，但存在TTFT代价。随后 P/D 都开启共享专家重叠，TPOT进一步下降3.5%–4.4%，吞吐几何均值仅+1.62%，hot1024 C1 TTFT恶化23.5%。原生角色指标显示该case P prefill多52–77ms，计算token数量不变，D prefill少约2ms。关闭P端重叠后，保留D实例的两轮对照中，该TTFT差异降至相对MTP1的+2.67%，TPOT改善保留。冷256 C1仍有+6.57%的观测差异，而原对照两轮TTFT差16.92%，不能将每个均值差均认作已确定的因果效应。

MTP热前缀比无MTP少复用一个128-token块；固定源码说明该块用于重建drafter需要的隐藏态。当前保留原生安全要求。D端原生指标的prefill KV computed tokens为0，不能由本地prefix hit减少推断D重新计算了整个prompt。

角色隔离的第二轮原生指标进一步显示：hot1024 C1 的P prefill从248.19ms降到205.06ms，D prefill为46.75→46.73ms。只改变P flag时，P耗时减少、D耗时保持，与前面的TTFT判断一致。证据为`node32-decode-only-role-metrics-r2-v1-2888766ddbf2/metrics.json`。

正确性检查分层保留：四种已测配置的正式HTTP/64-token/数值功能请求均通过；原12道算术检查每轮均11/12，所有原通过题未出现新增失败。原生目标和PD对同一个短few-shot `2+3=` 提示都输出4，该数值错误保留。新增24项数值、记录提取、2048/4096-token检索，通过相同token IDs分别走PD和native D，双方均24/24；这些有限任务不代表完整模型精度验收。

当前下一项为D-v6的MTP深度1→5，P-v5、D端重叠和其余设置固定。固定官方GLM5.3参考使用P1/D5，MTP1已有约94%的真实draft接受率；后续位置的接受率、target批次成本和端到端代价仍需实际对照。两轮正式请求及同一24项任务已排队；尚未将该候选计为收益。

证据：`evidence/20261017-official-pd-optimization/mtp1-comparison-v1.json`、`overlap-incremental-comparison-v1.json`、`decode-only-overlap-v5-vs-{mtp1,both-overlap,original}-v1.json`、`node32-official-protected-v5-r1-efce6a8b91ff/metrics.json`。原始逐请求记录、metrics前后快照、冻结脚本和生命周期保留；三个增量Mac归档已核验SHA。原32-rank SDMA/HCCS地图继续复用，所选新候选的最小实际执行profile待正式计时及扩展检查完成后进行。
