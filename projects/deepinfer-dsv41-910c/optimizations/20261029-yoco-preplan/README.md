# YOCO 压缩索引提前准备

## 背景与范围

固定官方源码 `553741557a57df44a46fb3ab72b6aafeaf945337` 的 YOCO 路径在输入 metadata 准备阶段生成 consumer 信息，在层边界使用重排索引。910C 路径原先在第 20 层边界生成选行和设备索引。前期主机时标中，压缩开始至 metadata 提交前的区间约 98–101 ms，包含 Python 计划、设备索引构造及可能的前序异步等待，不能据此认定具体耗时原因。

本候选将同一 forward 的 bounded replay 索引构造移到 producer 层之前，边界消费前检查 query 长度、keep、CP 布局、rank 和设备；不跨请求缓存。`yoco_preplan` 默认关闭，开启时要求 `yoco_bounded=true`。候选不改变模型计算、通信、W8A8 权重、DSpark、Engram 或 bounded replay 的保留长度。此实现借鉴官方准备与消费分离，尚未移植官方 CP mirror kernel，不能称为完整官方 YOCO。

## 验证结果

同一服务保持 DP2×TP8/EP16、异步 DSpark、256 token bounded replay、Engram CPU 表和 prefill UVA；关闭分层计时，只切换 `yoco_preplan`。输入 32768 token、输出 1024 token、并发 8，按关闭、开启、开启、关闭各测试两轮：

| 索引提前准备 | 轮次 | 平均 TTFT (ms) | 平均 TPOT (ms) | 输出吞吐 (token/s) |
| --- | --- | ---: | ---: | ---: |
| 关闭 | 1 | 609.8 | 12.4 | 608.215 |
| 开启 | 1 | 597.2 | 12.3 | 609.915 |
| 开启 | 2 | 584.9 | 12.2 | 606.016 |
| 关闭 | 2 | 596.4 | 12.2 | 606.751 |

平均 TTFT 为 603.10 → 591.05 ms，平均 TPOT 为 12.30 → 12.25 ms，两轮合并吞吐为 607.482 → 607.959 token/s。第二轮开启时吞吐低 0.735 token/s；仅两轮样本，不能判断细小差异的稳定性或原因。候选保留且默认关闭，不替换既有路径。

四轮各 8 条完整 1024 token 与既有参考逐 token 一致，每轮实际 KV 命中 235904/262144、分词缓存命中 8/8；16 个 rank 的准备和消费计数、bounded、Engram UVA 与异步 DSpark 路径均经核验。此前短诊断中，每个 rank 均准备并消费一次，8 条前 32 token 与参考一致。

开启后，第 20 层包含层执行和边界处理的主机区间仍约 104–109 ms，关闭约 109–110 ms；前处理区间均约 46–49 ms。这些区间含异步等待，既不能作为设备内核独占耗时，也不能证明索引构造是主要瓶颈。完整逐轮记录和审计 SHA 见 `results.json`。

## 应用与限制

1. 对 `manifest.json` 指定的父源码运行 `python3 verify.py <源码目录> base`。
2. 在独立源码副本执行 `patch -p1 < candidate.patch`，再运行 `python3 verify.py <源码目录> candidate`。
3. 通过作业模式设置 `yoco_preplan=true` 启用；其他已有优化按原配置控制。
4. 测试源码和原始数据保存在 172 的 `/data/tiankuan/wio/910c-deepinfer-172` 下。测试完成后候选开关关闭，异步 DSpark、bounded replay 和 prefill UVA 共同基线保持开启。

测试仅覆盖固定负载、greedy、ignore EOS 和当前缓存状态；设备缓存张量、其他批次及通用正确性未逐项验证，不能完全排除外部客户端干扰。
