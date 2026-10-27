# 172 · 2026-10-27 首token优化

当前接受版本：**prefill-wide-v12**，**997.90 token/s，平均首token1.838秒**。
16/16完整1024-token输出与冻结vLLM一致；每轮实际前缀命中89.990234375%。

| 版本 | 合并输出token/s | 平均首token ms | 结论 |
|---|---:|---:|---|
| 冻结vLLM-Ascend | 518.42 | 831.25 | 冻结参照 |
| 此前prefill-v9c | 737.99 | 4315.80 | 本轮起点 |
| [fused-moe-v10](fused-moe-v10/README.md) | 884.57 | 2821.00 | 接受 |
| [prefill-sp-v11](prefill-sp-v11/README.md) | 963.08 | 2154.55 | 接受 |
| [prefill-wide-v12](prefill-wide-v12/README.md) | 997.90 | 1838.35 | 接受 |
| [prefill-rms-v13](prefill-rms-v13/README.md) | 1042.21 | 1868.80 | 未采用，TTFT收益未确认 |

相对本轮开始的v9c：首token下降57.40%，吞吐提升35.22%。
当前首token仍比冻结vLLM慢1007.10ms；输出吞吐高92.49%。
两轮范围保留在各版本记录中，不宣称统计显著性或全局最优。

所有版本保持同一172、只读W8A8权重、固定镜像、DP2×TP8/EP16和同一AISBench协议。
原vLLM、v8和v9c冻结结果均未覆盖。代码按接受顺序保存增量补丁，不推DeepInfer上游。
Native算法仍为输入prompt lookup63+64-query完整目标模型图验证，未把它称为DSpark/MTP。

本轮按融合MoE→序列并行→容量→RMS融合逐项验证，每项两轮；无盲扫。
4096可行性来自已接受的融合/分片实现及更小MC2专用窗口；此前v9的2048 OOM仍是其旧分离GMM路径的有效失败记录。

- [当前运行配置](prefill-wide-v12/config/current-resident.json)
- [当前完整比较](prefill-wide-v12/results/comparison.json)
- [当前独立验收](prefill-wide-v12/results/independent-validation.json)
- [此前v9c](../prefill-v9c/README.md)
- [冻结vLLM](../../../910c-deepseek-v4.1-flash/README.md)

RMS评估后已回切v12，并恢复标准前缀/验证图预热；没有重跑正式性能测量。当前运行快照与原测量配置分别保留。

记录时间：2026-10-27T18:11:57.742064+08:00。归档/发布未追加推理，也未为交付重新启动服务。
