# Prefill 压缩阶段时标

## 目的与范围

前一轮分段结果显示，恢复前缀后的第 20 层处理区间约为 104–106 ms，包含 bounded replay 的条件判断和压缩。此候选在主机侧进一步记录压缩准备、metadata gather、trunk gather、previous gather 和其余准备，不改变计算、通信或权重，也不增加设备同步。`prefill_timing_detail` 默认关闭，开启时必须同时开启 `prefill_timing_split`。

## 验证条件和结果

同一服务固定 W8A8、DP2×TP8/EP16、learned DSpark、异步草稿预发、256 token bounded replay 和 prefill Engram UVA，仅切换细分时标。输入 32768 token、输出 1024 token、并发 8，每轮实际 KV 命中 235904/262144。按关闭、开启、开启、关闭的顺序各测两轮：

| 细分时标 | 轮次 | 平均 TTFT (ms) | 平均 TPOT (ms) | 输出吞吐 (token/s) |
| --- | --- | ---: | ---: | ---: |
| 关闭 | 1 | 681.6 | 12.4 | 604.442 |
| 开启 | 1 | 612.3 | 12.3 | 608.603 |
| 开启 | 2 | 585.6 | 12.2 | 606.391 |
| 关闭 | 2 | 591.7 | 12.2 | 606.231 |

四轮各 8 条完整 1024 token 序列与既有参考一致，各轮分词缓存命中 8/8，16 个 rank 的模式、bounded replay、Engram UVA 和异步 DSpark 路径已核验。关闭与开启时的合并输出吞吐分别为 605.335 和 607.495 token/s；样本不足以确定时标开销或波动原因，不能据此认定存在性能收益。

开启两轮、两个 DP 的压缩开始到 metadata 提交前为 98.15–100.67 ms；metadata、trunk 和 previous 的主机提交区间分别约 0.30–0.51、0.10–0.12 和 0.10–0.11 ms，后续准备约 3.73–3.92 ms。上述区间使用主机时钟，可能包含前序异步操作的等待；不能将其直接视为设备算子或通信的独占耗时，也不能判定具体原因。

完整逐轮数据、模式核验和审计摘要见 `results.json`。测试限于固定负载、greedy、ignore EOS 和当前缓存状态；设备缓存张量、其他请求形状及通用正确性未逐项验证，本地锁无法完全排除外部客户端。

## 应用与使用

1. 对 `manifest.json` 指定的父源码执行 `python3 verify.py <源码目录> base`。
2. 在独立副本执行 `patch -p1 < candidate.patch`，然后运行 `python3 verify.py <源码目录> candidate`。
3. 测试作业需要设置 `prefill_timing_split=true` 与 `prefill_timing_detail=true`；默认关闭细分时标。
4. 原始数据和候选源码保存在 172 的 `/data/tiankuan/wio/910c-deepinfer-172` 下。对照结束后服务已恢复安全模式；实验组合开关和时标已关闭。

此候选用于定位瓶颈，不作为性能优化启用。
