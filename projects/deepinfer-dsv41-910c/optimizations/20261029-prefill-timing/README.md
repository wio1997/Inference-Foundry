# Prefill 第 20 层时标分段

## 目的与范围

同期对照中，DeepInfer 两轮 TTFT 存在明显波动。已有主机时标显示，第 20 层相关区间分别为 474.04/454.85 ms 和 107.87/108.44 ms（每轮列出两个 DP），但原区间同时包含层 forward 和 bounded 压缩，不能区分两者。

本候选增加可按 job 开关的主机时标，分别记录层 forward 结束与后续处理结束；不增加设备同步，不改变模型计算、权重、通信、DSpark 或 Engram。`post_layer_host_seconds` 包含压缩条件判断和执行，也可能包含前序异步工作的等待。

## 单因素测试

同一服务固定 learned DSpark、异步草稿预发、256 token bounded replay 和 prefill Engram UVA，只切换 `prefill_timing_split`。输入 32768 token、输出 1024 token、并发 8；各轮实际 KV 命中均为 235904/262144。按关、开、开、关顺序各测两轮：

| 时标开关 | 轮次 | 平均 TTFT (ms) | 平均 TPOT (ms) | 输出吞吐 (token/s) |
| --- | --- | ---: | ---: | ---: |
| 关闭 | 1 | 677.6 | 12.4 | 605.098 |
| 开启 | 1 | 582.3 | 12.4 | 607.879 |
| 开启 | 2 | 596.1 | 12.3 | 604.641 |
| 关闭 | 2 | 588.6 | 12.2 | 606.447 |

开启时，第 20 层 forward 的主机区间为 5.20–5.26 ms，后续区间为 104.25–106.16 ms。后续区间值得继续定位，但现有证据不能将其归因于压缩计算或通信，也不能将它当成设备算子独占耗时。

四轮共 32 条完整 1024 token 序列与既有参考一致；每轮分词缓存命中 8/8，16 个 rank 的模式、异步 DSpark、bounded 和 prefill UVA 均已核验。关闭两轮合并吞吐为 605.772，开启为 606.255 token/s；平均 TPOT 为 12.30/12.35 ms。第二轮开启时 TTFT 高 7.5 ms、TPOT 高 0.1 ms、吞吐低 1.806 token/s。两轮样本不足以确定观测开销及波动成因，不能认定时标没有影响。完整指标与审核 SHA 见 `results.json`。

## 使用方法

1. 对 `manifest.json` 指定的父源码执行 `python3 verify.py <源码目录> base`。
2. 在独立副本执行 `patch -p1 < candidate.patch`，再运行 `python3 verify.py <源码目录> candidate`。
3. 作业模式中的 `prefill_timing_split` 控制额外时标，默认关闭。启用后记录 `layer_forward_host_seconds` 和 `post_layer_host_seconds`，原有总区间保留。
4. 原始对照与候选源码位于 172 的 wio 项目目录。测试结束后时标开关及实验组合已关闭，服务保留。

该候选用于定位，不构成性能优化。测试只覆盖固定批次、greedy、ignore EOS 与当前缓存状态；设备事件、其他工作负载和通用正确性仍需独立验证。
