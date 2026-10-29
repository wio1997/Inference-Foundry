# 910C W8A8 同期性能对照

## 测试范围

2026 年 10 月 29 日，在 172 服务器使用同一 DeepSeek V4.1 Flash W8A8 权重和同一 AISBench 版本，对 vLLM Ascend 与 910C DeepInfer 适配进行阶段性对照。每侧使用 16 个逻辑设备，DP2×TP8、EP16；输入 32768 token、输出 1024 token、并发 8，greedy、ignore EOS、流式输出。每轮实际 KV 查询总量为 262144 token，命中总量为 235904 token。

测试顺序为 vLLM 第一轮、DeepInfer 第一轮、DeepInfer 第二轮、vLLM 第二轮；引擎切换时重新启动服务。vLLM 使用冻结配置，DeepInfer 启用 learned DSpark、异步草稿预发、256 token bounded replay、prefill Engram UVA、共享专家多流和精确 FFN 尾部裁剪。Engram 表位于 CPU，主干权重不使用 CPU prefetch。

## 结果

| 引擎 | 轮次 | 平均 TTFT (ms) | 平均 TPOT (ms) | 输出吞吐 (token/s) |
| --- | --- | ---: | ---: | ---: |
| vLLM Ascend | 1 | 2513.1 | 14.5 | 465.306 |
| DeepInfer | 1 | 1365.0 | 12.4 | 573.447 |
| DeepInfer | 2 | 744.2 | 12.2 | 599.891 |
| vLLM Ascend | 2 | 2523.2 | 14.5 | 459.160 |

两轮合并吞吐按输出 token 总数除以两轮总时长计算：vLLM 为 462.213 token/s，DeepInfer 为 586.371 token/s，高 26.86%。两轮平均 TTFT 分别为 2518.15 ms 和 1054.60 ms；平均 TPOT 分别为 14.50 ms 和 12.30 ms。结果限于本次负载与两轮样本，不能推广至其他输入长度、并发或采样设置。

vLLM 每轮 8 条完整输出文本与冻结参考一致；DeepInfer 每轮 8 条完整输出 token 序列与已有 B 参考一致。两侧各轮均有 8 个成功请求、8192 个输出 token。DeepInfer 的两个种子在各 DP 上均无既有 KV 命中；vLLM 种子的两个 engine 计数亦显示无既有命中。各项原始数据和审核摘要在服务器 wio 对照目录中，摘要及 SHA256 见 `results.json`。

## 首 token 时标

DeepInfer 两轮的分词缓存均未命中。服务端分词平均耗时为 149.94/156.55 ms；从 IPC 提交到收到 worker 首 token 的平均耗时为 1156.67/533.73 ms。该区间包含缓存恢复、prefill、首轮生成和 IPC，不能解释为单个算子耗时。

已记录的 prefill 主机时标中，第 20 层相关区间在第一轮两个 DP 分别为 474.04/454.85 ms，第二轮为 107.87/108.44 ms。该区间覆盖层计算和 bounded 压缩，可能包含等待先前异步操作；目前不能归因于其中任何一个环节。后续需拆分时标并结合设备事件验证。

## 证据边界

- vLLM 的逐请求 KV 命中、实际 DP/slot、原生生成 token ID 和分词缓存状态未取得。输出文本一致不等同于原生 token ID 逐项一致。
- DeepInfer 两轮均为分词缓存未命中，并启用了轻量时标；两引擎的观测开销未证明相同。
- 本地锁及空闲检查不能完全排除其他同机 loopback 请求；聚合计数的核验范围以实际快照为准。
- vLLM 已确认 decode 图捕获，正式请求的独立 replay 证据尚缺。DeepInfer 的 bounded replay 为近似路径，正确性证据限于本次固定负载；参考 token 文件在早期审计中未逐文件冻结 SHA。
- 测试顺序涉及服务重启，不排除时间和热态变化。当前对照不能证明 DeepInfer 已完整继承所有适用官方优化，也不能代替后续逐项验证。
