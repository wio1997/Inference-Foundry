# 预填充 KV 通信与 Q 投影

## 范围与实现

固定的 DeepInfer 官方源码在 CP 注意力路径中先提交压缩输入的 all-gather，再执行独立的 Q 投影。现有 910C 路径在 token-sharded CP 预填充中先计算 Q，再汇集本地 KV；本次固定负载每个 rank 实际调用 34 次 KV 汇集。

候选将非 KV 源层、eager 预填充的 KV all-gather 提前至 Q 投影之前，以 HCCL `async_op=True` 提交，在使用结果前调用 `wait()` 并保留输入、输出和 work 对象。`prefill_kv_overlap` 默认关闭；关闭时保持原顺序。候选不改 KV 计算、通信组、decode 图、DSpark、Engram、bounded replay 或 Recipe 分词器。官方代码的压缩输入和 CP 布局不同，本候选借鉴其调度方式，未移植官方 CP kernel。

## 两轮单因素对照

同一服务仅切换 `prefill_kv_overlap`，两组共同启用 Recipe、异步 DSpark、256 token bounded replay 和 prefill Engram UVA。输入 32768 token、输出 1024 token、并发 8；每轮实际 KV 命中 235904/262144，分词缓存命中 8/8。按关闭、开启、开启、关闭顺序各测两轮：

| KV 重叠 | 轮次 | 平均 TTFT (ms) | 平均 TPOT (ms) | 输出吞吐 (token/s) |
| --- | --- | ---: | ---: | ---: |
| 关闭 | 1 | 608.7 | 12.4 | 605.095 |
| 开启 | 1 | 603.9 | 12.4 | 606.107 |
| 开启 | 2 | 588.3 | 12.3 | 600.725 |
| 关闭 | 2 | 593.9 | 12.3 | 602.222 |

两轮平均 TTFT 为 601.30 → 596.10 ms，平均 TPOT 均为 12.35 ms，合并吞吐为 603.655 → 603.404 token/s。开启的第二轮吞吐低 1.497 token/s，合并吞吐略低；两轮样本不足以判断变化的稳定性和原因。候选保留但默认关闭，已有优化继续启用。

四轮共 32 条完整 1024 token 序列与参考逐 token 一致；各轮 16 rank 的 DSpark、bounded replay、Engram UVA 和实际 Recipe 路径经核验。开启两轮每个 rank 各提交并等待 34 次，关闭为 0；TP rank 计数不能累加为独立用户操作。此前 8×32 token 短诊断同样验证了调用和输出。提交与等待计数不能证明设备上的通信和 Q 投影在时间上重叠；目前没有相应设备时间线证据。

## 应用与限制

父源码为 `src/DeepInfer-recipe-raw-20261029`，独立候选为 `src/DeepInfer-prefill-kv-overlap-20261029`。先运行 `python3 verify.py <父源码目录> base`，在独立副本执行 `patch -p1 < candidate.patch` 后运行 `python3 verify.py <候选源码目录> candidate`。作业配置 `prefill_kv_overlap` 控制该路径，切换前须确认请求空闲。

当前服务已恢复关闭候选的共同优化模式。完整逐轮数据、来源哈希和审核摘要见 `results.json`。本次只覆盖固定形状、greedy、ignore EOS 和现有缓存状态；设备缓存张量、其他输入与通用正确性未逐项验证，本地锁不能完全排除外部客户端。
