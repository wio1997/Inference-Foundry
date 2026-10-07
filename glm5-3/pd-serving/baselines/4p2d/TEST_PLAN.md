# 4P2D性能基线：并发与SLA测量计划

状态：PLANNED。先发布诊断，再切换服务；未执行不填结果。

本次建立用户指定的4P2D性能基线，不是q1修复实验。保持当前代码，不加入q1修复，记录C16/C32/C64/C96的吞吐、TTFT/TPOT与实际并发。现有2P2D诊断结果归独立issues专题，不作为本基线数值。后续任何修复必须相对同一4P2D基线做matched A/B。到达速率对照用于解释负载设置，不与代码收益混账。

## 拓扑与合同

P池DP4×TP16，预期EP64；D池DP16×TP2，EP32。四P统一util0.93、max batched tokens8192、max sequences8、eager、MTP1。D保持util0.92、MTP5、FULL_DECODE_ONLY及原传输配置；连接器prefill.dp_size改为4。

现场四P同一SPOD域、硬件域大小384；该规模与EP64软件进程组不同。启动后验证实际EP rank/group。默认通信按token规模选择MC2/AllToAll；prefill_mc2、fused_mc2、hierarchy、fullmesh_v2等属于独立模式，不凭EP64标签声称已启用。

SLA沿用MISSION严格小于门槛：

| 指标 | P50 | P75 | P90 | P99 |
|---|---:|---:|---:|---:|
| TTFT | 4秒 | 8秒 | 12秒 | 30秒 |
| TPOT | 18毫秒 | — | 40毫秒 | — |

主容量负载：input_len80000、output_len600、共享前缀比例0.93、temperature0、ignore_eos。实际tokenizer长度、cache hit、request_rate分别记账。历史81920输入不能伪称与80000完全一致。

“差距不大”无明确容差：只报告绝对和相对超标量，不把超标判为通过。

## 实验顺序与裁决

1. 六机服务及完整P→KV→D smoke、输出计数、EP64/EP32成立后再压测。
2. 固定公共前缀、覆盖全部四P预热，冷/暖状态分账。
3. 到达受限对照：C16/C32、相同request_rate0.93 req/s、相同输入集，初拟N48；验证实际并发是否未达到上限。
4. 容量对照：C16/C32/C64/C96，取消外部RPS限制、持续补充请求，初拟统一N384（至少4×最大并发）；具体调度参数经AISBench源码验证后冻结。
5. 只对接近门槛、明显波动或会改变判断的结果补针对性重复。不以单轮有限样本P99宣称长期稳定。

Hypothesis：C16/C32相近至少部分由到达速率/有效并发限制，而非C16已达设备容量。
Distinguishing evidence：POST/首chunk/结束时间、实际并发面积及峰值、P/D running/waiting、KV/cache计数、MTP统计、错误及有效token总数。
Decision table：固定RPS实际并发相近、无限速后TPS升高→支持到达受限；实际并发升高但TPS不升→继续分解P排队/KV/D执行与通信；C64/C96成功但分位超标→报告差距，不能称达标；失败/少输出/非PD路径→保留失败，不以成功子集验收。

每轮记录配置和代码身份、原始证据哈希、预热状态、有效工作量、request/s、output token/s、TTFT/TPOT分位数、E2E延迟、成功失败及实际并发。TPS按实际总有效输出/明确请求窗口，不用配置并发除以TPOT P50替代。P-only测试不能替代完整PD的TPOT/SLA。
