# 32K / 1K · C8 · 90% prefix hit · 调优后基线

2026-10-26，固定 DP2×TP8、EP16、16 逻辑设备 / 8 物理卡，使用原有 W8A8 权重与 vLLM-Ascend v0.30.0rc2-a3。最佳已验证参数为 **DSpark5 / 图 / batch=8192 / seq=4 / MC2=1**。

冻结基线合并总输出吞吐 **518.4172 token/s**，相对本次匹配数据的 eager 参考 49.8272 token/s 为 **10.404 倍**。这表示当前测试范围与固定负载下的最优结果。

| 轮次 | 总输出吞吐 token/s | 平均 TTFT s | 平均 TPOT ms | 实际前缀命中率 | 成功 |
|---|---:|---:|---:|---:|---:|
| 1 | 509.5203 | 0.8857 | 14.5 | 89.99023438% | 8/8 |
| 2 | 527.6304 | 0.7768 | 14.0 | 89.99023438% | 8/8 |

每轮 8 请求、8 并发，每请求实际输入 32768、输出 1024 token；每轮服务计数器增量为输入 262144、输出 8192、prefix queries 262144、hits 235904、成功请求 8、抢占 0。输出吞吐仅计算输出 token，采用 8192 / AISBench 正式测量时长，两轮合并采用 16384 / 两轮时长之和。

关键配置：max_model_len=34816、max_num_batched_tokens=8192、max_num_seqs=4/DP、block_size=128、gpu_memory_utilization=0.9。推测解码：DSpark，5 个草稿 token；draft eager，target Decode 图。Engram CPU offload 与 DP shared memory 开启；主干 CPU prefetch 关闭；异步调度、prefix caching、chunked prefill 开启。最终参数见 [selected-configuration.json](selected-configuration.json)，启动脚本为项目的 [serve.sh](../../scripts/serve.sh)。

| 候选 | 两轮合并总输出吞吐 token/s |
|---|---:|
| DSpark5 / 图 / batch=8192 / seq=4 / MC2=1 | 520.7285 |
| DSpark5 / 图 / batch=16384 / seq=4 / MC2=1 | 520.1173 |
| 最终复核：DSpark5 / 图 / batch=8192 / seq=4 / MC2=1 | 518.4172 |
| DSpark5 / 图 / batch=8192 / seq=4 / MC2=0 | 510.8504 |
| DSpark5 / 图 / batch=2048 / seq=4 / MC2=1 | 508.9863 |
| DSpark3 / 图 / batch=2048 / seq=4 / MC2=1 | 438.6832 |
| DSpark5 / 图 / batch=512 / seq=4 / MC2=1 | 416.0608 |
| DSpark1 / 图 / batch=2048 / seq=4 / MC2=1 | 253.8866 |
| 无推测解码 / 图 / batch=2048 / seq=8 / MC2=1 | 235.5920 |
| 无推测解码 / eager / batch=2048 / seq=4 / MC2=1 | 49.8272 |

最终复核为恢复选中参数后独立启动、完整预热并再次测量的结果，冻结值以该复核为准。8192 与 16384 预算若处于相同吞吐平台，优先保留较小的8192预算。草稿长度1/3/5对应每DP四请求时验证步token数8/16/24，适配TP8图捕获对齐约束。

仅通过逐请求输入/输出长度、服务计数器、健康检查、输出一致性及正式阶段无新 JIT 编译验收的候选参与排名。非 eager 候选的 16 个正式输出均与匹配提示词的 eager 输出逐字一致；这是本次固定样本的一致性检查，不是全面模型准确率评测。

图捕获需要同时满足 TP 和调度上限。TP8 会过滤捕获尺寸；无推测解码且 max_num_seqs=4 时，剩下的 size=8 又被 Decode 上限过滤，形成空捕获，不能把参数开启当作实际图回放。该候选已排除。普通有效图使用 max_num_seqs=8；DSpark 的 Decode query 长度为 1+草稿长度，按对应尺寸捕获，实际非空捕获日志已验收。

客户端使用 [AISBench 固定源码的 prefix_cache 插件](https://github.com/AISBench/benchmark/tree/7bcb6440da11bef020943b2af9ed7c1ac8362306/plugins/prefix_cache)，版本0.5.0。每候选先完整跑一轮 8×32K/1K 预热，不计正式成绩；随后两轮分别预热两个 DP 的目标前缀，AISBench 原生 num_warmups=0。跨候选使用相同提示词字节，并核对 SHA256；三阶段的命名空间互不相同。新服务 KV 缓存为空；reset 接口不可用，插件保留 ASSUME_EMPTY_CACHE 警告，实际89.990234375%由正式阶段前后计数器差值核实。

Fused MC2 与共享专家多流在当前镜像中互斥，按有效组合比较。CPU绑定、原有W8A8量化、设备与并行布局保持固定；没有修改权重或引入主干CPU卸载。原始 JSON/CSV、逐请求输出、指标快照、容器、NPU和捕获日志留在私有 wio 项目目录，公开仓库仅保存脱敏记录。汇总及原始性能文件 SHA256 见 [tuned-baseline-summary.json](tuned-baseline-summary.json)。
